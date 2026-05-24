#!/usr/bin/env python3
"""
Batch PDF → Markdown  [docext + Tesseract rotation + streaming + retry]
  - docext image pipeline + Tesseract OSD rotation
  - Streaming vLLM API (no timeouts)
  - Inline page retry + placeholder for permanent failures

Usage:
    nohup python batch_pdf2md.py > nohup.out 2>&1 &
    tail -f nohup.out
    tail -f conversion.log
    cat progress.txt
"""

import os, sys, time, signal, logging, subprocess, traceback
import requests, threading, shutil, json, re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import cv2
from docext.core.utils import convert_files_to_images, resize_images, encode_image

# ── STRICT OS-LEVEL CPU AFFINITY ──
# This physically restricts the Python process and all child processes (including vLLM)
def enforce_cpu_affinity(num_cores=8):
    try:
        import psutil
        p = psutil.Process(os.getpid())
        available_cores = p.cpu_affinity()
        if len(available_cores) > num_cores:
            p.cpu_affinity(available_cores[:num_cores])
    except ImportError:
        try:
            # Fallback for Windows using ctypes
            import ctypes
            mask = (1 << num_cores) - 1
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask)
        except Exception:
            pass

enforce_cpu_affinity(8)

# Limit OpenMP/OpenBLAS threads to 1 per instance to prevent aggressive CPU utilization.
# We will use a Semaphore to allow exactly 8 concurrent CPU tasks globally.
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Force OpenCV to use only 1 thread
cv2.setNumThreads(1)

# Global semaphore to limit CPU-intensive tasks (like rotation/Tesseract) to 7 cores total
# (leaving 1 core for the vLLM server to ensure strict 8 core maximum system-wide)
_cpu_semaphore = threading.Semaphore(7)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
VLLM_MODEL       = "nanonets/Nanonets-OCR2-3B"
CUDA_DEVICE      = "0"
VLM_PORT         = 8000
VLLM_API_URL     = f"http://127.0.0.1:{VLM_PORT}/v1/chat/completions"

# ── Paths (all relative to script dir) ────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).resolve().parent
DATA_DIR         = SCRIPT_DIR / "data"
OUTPUT_BASE_DIR  = SCRIPT_DIR / "markdown"
LOG_FILE         = SCRIPT_DIR / "conversion.log"
PROGRESS_FILE    = SCRIPT_DIR / "progress.txt"
FAILED_CSV       = SCRIPT_DIR / "failed_conversions.csv"
SERVER_LOG       = SCRIPT_DIR / "server.log"
TEMP_DIR         = SCRIPT_DIR / ".page_cache"

# ── H100 94GB optimized settings ─────────────────────────────────────────────
PARALLEL_WORKERS = 180       # total threads sending requests across PDFs
PDF_CONCURRENCY = 6          # number of PDFs processed in parallel
PER_PDF_WORKERS = max(1, PARALLEL_WORKERS // PDF_CONCURRENCY)
MAX_NUM_SEQS     = 220       # vLLM concurrent sequences
MAX_MODEL_LEN    = 15000     # context window
MAX_GEN_TOKENS   = 10000     # max output tokens (docext default)
MAX_IMG_SIZE     = 2048      # docext resize target
GPU_MEM_UTIL     = 0.95      # GPU memory utilization
MIN_PDF_SIZE     = 10 * 1024
PAGE_RETRY_COUNT = 2
PAGE_RETRY_DELAY = 5
PDF_RETRY_PASSES = 1
REQUEST_TIMEOUT  = 600       # streaming keeps connection alive
SERVER_TIMEOUT   = 600

PAGE_SEPARATOR   = "\n\n---\n\n"
PLACEHOLDER      = "[PAGE {n} - EXTRACTION FAILED - MANUAL REVIEW REQUIRED]"

OCR_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in HTML format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a small "
    "description of the image inside the <img></img> tag; otherwise, add the image caption "
    "inside <img></img>. Watermarks should be wrapped in brackets. "
    "Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "
    "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes. Only return HTML table within <table></table>."
)

_server_proc   = None
_csv_lock      = threading.Lock()
_progress_lock = threading.Lock()
_stats         = {"done": 0, "partial": 0, "failed": 0,
                  "pages": 0, "failed_pages": 0, "retried_ok": 0}
_stats_lock    = threading.Lock()
_server_lock   = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

class TqdmFileHandler(logging.Handler):
    def __init__(self, fp):
        super().__init__()
        self.file = open(fp, "a", encoding="utf-8", buffering=1)
    def emit(self, record):
        try: self.file.write(self.format(record) + "\n"); self.file.flush()
        except: pass

class TqdmStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try: tqdm.write(self.format(record))
        except: pass

def setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    logger = logging.getLogger("batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = TqdmFileHandler(LOG_FILE); fh.setFormatter(logging.Formatter(fmt))
    sh = TqdmStreamHandler();       sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh); logger.addHandler(sh)
    for n in ["httpx","httpcore","urllib3","requests"]:
        logging.getLogger(n).setLevel(logging.WARNING)
    return logger

def init_failed_csv():
    if not FAILED_CSV.exists():
        with open(FAILED_CSV, "w") as f:
            f.write("timestamp,pdf_path,sector,company,year,page,error\n")

def log_failure(pdf_path, page, error):
    parts = pdf_path.relative_to(DATA_DIR).parts
    sector  = parts[0] if len(parts) > 0 else "unknown"
    company = parts[1] if len(parts) > 1 else "unknown"
    year    = pdf_path.stem
    err = str(error).replace('"', "'").replace("\n", " ")[:400]
    with _csv_lock:
        with open(FAILED_CSV, "a") as f:
            f.write(f'"{datetime.now()}","{pdf_path}","{sector}","{company}","{year}",{page},"{err}"\n')

def save_progress(done_pdfs, total_pdfs, current):
    with _progress_lock:
        pct = 100 * done_pdfs / total_pdfs if total_pdfs > 0 else 0
        with open(PROGRESS_FILE, "w") as f:
            f.write(f"PDFs       : {done_pdfs}/{total_pdfs} ({pct:.1f}%)\n")
            with _stats_lock:
                f.write(f"Succeeded  : {_stats['done']}\n")
                f.write(f"Partial    : {_stats['partial']} "
                        f"(have placeholders)\n")
                f.write(f"Failed     : {_stats['failed']}\n")
                f.write(f"Pages done : {_stats['pages']}\n")
                f.write(f"Pages fail : {_stats['failed_pages']}\n")
                f.write(f"Retried ok : {_stats['retried_ok']}\n")
            f.write(f"Last update: {datetime.now()}\n")
            f.write(f"Current    : {current}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ROTATION DETECTION (Tesseract OSD — per-page)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_rotation(image_path):
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(str(image_path))
        match = re.search(r'Rotate:\s*(\d+)', osd)
        if match:
            angle = int(match.group(1))
            if angle in (90, 180, 270):
                return angle
    except: pass
    return 0

def rotate_image(image_path, angle):
    if angle == 0: return
    img = cv2.imread(str(image_path))
    if img is None: return
    if   angle == 90:  img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180: img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(str(image_path), img)



# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING OCR (matches docext's stream_request — no timeouts)
# ═══════════════════════════════════════════════════════════════════════════════

def stream_ocr(image_path):
    """Stream one page through vLLM. Returns markdown text."""
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
            {"type": "text", "text": OCR_PROMPT},
        ]}],
        "max_tokens": MAX_GEN_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {os.getenv('API_KEY', 'EMPTY')}"}
    result = []
    with requests.post(VLLM_API_URL, json=payload, headers=headers,
                       stream=True, timeout=REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line: continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]": break
                try:
                    chunk = json.loads(data)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        c = chunk["choices"][0].get("delta", {}).get("content", "")
                        if c: result.append(c)
                except json.JSONDecodeError: continue
    return "".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER
# ═══════════════════════════════════════════════════════════════════════════════

def is_vllm_healthy():
    try: return requests.get(f"http://127.0.0.1:{VLM_PORT}/health", timeout=5).status_code == 200
    except: return False

def start_vllm_server(logger):
    global _server_proc
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", VLLM_MODEL,
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--max-model-len",       str(MAX_MODEL_LEN),
        "--dtype",               "float16",
        "--max-num-seqs",        str(MAX_NUM_SEQS),
        "--limit-mm-per-prompt", "image=1",
        "--host", "127.0.0.1",
        "--port", str(VLM_PORT),
    ]
    log_fh = open(SERVER_LOG, "w", encoding="utf-8")
    logger.info(f"Starting vLLM — GPU {CUDA_DEVICE}, seqs={MAX_NUM_SEQS}")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    _server_proc = proc
    return proc

def wait_for_server(proc, logger, timeout=SERVER_TIMEOUT):
    logger.info(f"Waiting up to {timeout}s...")
    start = time.time()
    with tqdm(total=timeout, desc="vLLM startup", unit="s",
              bar_format="{l_bar}{bar}| {n:.0f}/{total}s",
              dynamic_ncols=True) as pbar:
        last = 0
        while time.time() - start < timeout:
            if proc.poll() is not None:
                logger.error("vLLM exited! Check server.log")
                if SERVER_LOG.exists():
                    for line in SERVER_LOG.read_text().splitlines()[-20:]:
                        logger.error(f"  {line}")
                return False
            if SERVER_LOG.exists():
                if "out of memory" in SERVER_LOG.read_text()[-3000:].lower():
                    logger.error("OOM detected — reduce workers or GPU util")
                    proc.terminate()
                    return False
            if is_vllm_healthy():
                pbar.update(timeout - last)
                logger.info(f"vLLM ready after {time.time()-start:.0f}s")
                return True
            now = int(time.time() - start)
            pbar.update(now - last)
            last = now
            time.sleep(5)
    logger.error(f"Timeout after {timeout}s")
    return False

def ensure_server(logger):
    if is_vllm_healthy():
        logger.info("vLLM already running"); return None
    proc = start_vllm_server(logger)
    if not wait_for_server(proc, logger):
        proc.terminate(); sys.exit(1)
    time.sleep(3); return proc

def restart_if_dead(proc, logger):
    if proc is None:
        return ensure_server(logger) if not is_vllm_healthy() else None
    if proc.poll() is not None:
        logger.warning("vLLM died — restarting"); return ensure_server(logger)
    return proc


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONVERSION WITH RETRY
# ═══════════════════════════════════════════════════════════════════════════════

def is_error_response(text):
    p = text[:120].lower()
    return any(e in p for e in ["❌","error processing","unable to get","traceback"])

def convert_page_with_retry(page_num, image_path):
    last_err = None
    for attempt in range(1 + PAGE_RETRY_COUNT):
        try:
            result = stream_ocr(image_path)
            if not result or len(result.strip()) < 3:
                raise ValueError("Empty response")
            if is_error_response(result):
                raise ValueError(f"Error: {result[:120]}")
            return page_num, result
        except ValueError: raise
        except Exception as e:
            last_err = e
            if attempt < PAGE_RETRY_COUNT: time.sleep(PAGE_RETRY_DELAY)
    raise RuntimeError(f"Page {page_num} failed {1+PAGE_RETRY_COUNT}x: {last_err}")


# ═══════════════════════════════════════════════════════════════════════════════
# PDF PIPELINE (docext images + rotation + streaming OCR)
# ═══════════════════════════════════════════════════════════════════════════════

def load_completed_pdfs():
    completed, has_placeholders = set(), set()
    for md in OUTPUT_BASE_DIR.rglob("*.md"):
        pdf_eq = DATA_DIR / md.relative_to(OUTPUT_BASE_DIR).with_suffix(".pdf")
        if not pdf_eq.exists(): continue
        try:
            content = md.read_text(encoding="utf-8")
            if "EXTRACTION FAILED" in content:
                has_placeholders.add(str(pdf_eq))
            else:
                completed.add(str(pdf_eq))
        except: pass
    return completed, has_placeholders

def get_output_path(pdf_path):
    return OUTPUT_BASE_DIR / pdf_path.relative_to(DATA_DIR).with_suffix(".md")

def process_pdf(pdf_path, logger, pdf_bar, is_retry_pass=False, worker_count=PER_PDF_WORKERS):
    out_path = get_output_path(pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rel = pdf_path.relative_to(DATA_DIR)

    global _server_proc
    with _server_lock:
        _server_proc = restart_if_dead(_server_proc, logger)

    try:
        # ── Handle retry pass ─────────────────────────────────────────────
        if is_retry_pass and out_path.exists():
            existing_parts = out_path.read_text(encoding="utf-8").split(PAGE_SEPARATOR)
            pages_to_retry = [i+1 for i, p in enumerate(existing_parts) if "EXTRACTION FAILED" in p]
            if not pages_to_retry: return "ok"
            logger.info(f"  Retry: {len(pages_to_retry)} pages in {rel}")
            # Re-render only failed pages using docext
            with _cpu_semaphore:
                image_paths = convert_files_to_images([str(pdf_path)])
            # Filter to only retry pages
            retry_images = [(i+1, p) for i, p in enumerate(image_paths) if (i+1) in pages_to_retry]
            # Rotation + resize
            for _, img_p in retry_images:
                with _cpu_semaphore:
                    angle = detect_rotation(img_p)
                    if angle != 0: rotate_image(img_p, angle)
            with _cpu_semaphore:
                resize_images([p for _, p in retry_images], MAX_IMG_SIZE)
            existing_results = {i+1: p for i, p in enumerate(existing_parts) if "EXTRACTION FAILED" not in p}
            pages = retry_images
            total_pages = len(existing_parts)
        else:
            # ── Fresh: render all pages via docext ─────────────────────────
            try:
                with _cpu_semaphore:
                    image_paths = convert_files_to_images([str(pdf_path)])
            except Exception as e:
                logger.error(f"  ✗ Render failed [{rel}]: {e}")
                log_failure(pdf_path, 0, f"Render: {e}"); return "failed"
            if not image_paths:
                log_failure(pdf_path, 0, "No pages"); return "failed"

            # Parallel Rotation detection + correction (per-page OSD)
            def preprocess_img(img_p):
                with _cpu_semaphore:
                    angle = detect_rotation(img_p)
                    if angle != 0:
                        rotate_image(img_p, angle)

            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                list(pool.map(preprocess_img, image_paths))

            # Resize (docext)
            with _cpu_semaphore:
                resize_images(image_paths, MAX_IMG_SIZE)

            pages = [(i+1, p) for i, p in enumerate(image_paths)]
            existing_results = {}
            total_pages = len(pages)

        num_pages   = len(pages)
        results     = dict(existing_results)
        perm_failed = []
        page_times  = []

        # ── Per-page tqdm bar ─────────────────────────────────────────────
        prefix = "↺ " if is_retry_pass else "  "
        page_bar = tqdm(
            total=num_pages,
            desc=f"{prefix}{rel.parts[-2]}/{rel.stem}" if len(rel.parts) > 1 else f"{prefix}{rel.stem}",
            unit="pg",
            dynamic_ncols=True,
            leave=False,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt}pg "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )
        page_bar.set_postfix(ok=0, fail=0, w=worker_count)

        # ── Parallel page OCR ─────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(convert_page_with_retry, pn, ip): pn
                for pn, ip in pages
            }
            t0 = time.time()
            for fut in as_completed(futures):
                pnum = futures[fut]
                try:
                    _, md = fut.result()
                    results[pnum] = md
                    with _stats_lock:
                        _stats["pages"] += 1
                        if is_retry_pass:
                            _stats["retried_ok"] += 1
                    page_times.append(time.time() - t0)
                    t0 = time.time()
                except Exception as e:
                    perm_failed.append(pnum)
                    results[pnum] = PLACEHOLDER.format(n=pnum)
                    with _stats_lock:
                        _stats["failed_pages"] += 1
                    log_failure(pdf_path, pnum, str(e))
                    tqdm.write(
                        f"  ✗ Page {pnum} permanently failed [{rel}]: "
                        f"{str(e)[:80]}"
                    )
                finally:
                    page_bar.set_postfix(
                        ok=len([r for r in results.values()
                                if "EXTRACTION FAILED" not in str(r)]),
                        fail=len(perm_failed),
                        w=worker_count,
                    )
                    page_bar.update(1)

        page_bar.close()

        # ── Stitch in order ───────────────────────────────────────────────
        final_md = PAGE_SEPARATOR.join(
            results.get(p, PLACEHOLDER.format(n=p))
            for p in range(1, total_pages + 1)
        )
        out_path.write_text(final_md, encoding="utf-8")
        size_kb = out_path.stat().st_size / 1024
        avg_t   = sum(page_times) / len(page_times) if page_times else 0

        # Cleanup temp images
        for _, ip in pages:
            try: os.remove(ip)
            except: pass

        if perm_failed:
            logger.warning(
                f"⚠  {rel} — {total_pages-len(perm_failed)}/"
                f"{total_pages}pg ok | "
                f"placeholders: {sorted(perm_failed)} | "
                f"{size_kb:.0f}KB | {avg_t:.1f}s/pg"
            )
            return "partial"
        else:
            logger.info(
                f"✓  {rel} — {total_pages}pg | "
                f"{avg_t:.1f}s/pg | {size_kb:.0f}KB"
            )
            if pdf_bar:
                pdf_bar.set_postfix(
                    last=f"{rel.parts[-2]}/{rel.stem}" if len(rel.parts) > 1 else rel.stem,
                    pg=total_pages,
                    kb=f"{size_kb:.0f}",
                )
            return "ok"

    except Exception as e:
        logger.error(f"  ✗ {rel}: {e}")
        log_failure(pdf_path, 0, str(e))
        return "failed"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def shutdown(sig, frame):
    tqdm.write("\nInterrupted. Re-run to resume.")
    if _server_proc and _server_proc.poll() is None: _server_proc.terminate()
    shutil.rmtree(TEMP_DIR, ignore_errors=True); sys.exit(0)

def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger = setup_logging()
    init_failed_csv()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  Batch PDF → Markdown [docext + rotation + streaming]")
    logger.info(
        "  H100 94GB | TotalWorkers=%s | PdfConcurrency=%s | PerPdfWorkers=%s | Seqs=%s"
        % (PARALLEL_WORKERS, PDF_CONCURRENCY, PER_PDF_WORKERS, MAX_NUM_SEQS)
    )
    logger.info(f"  Data: {DATA_DIR} | Output: {OUTPUT_BASE_DIR}")
    logger.info("=" * 70)

    server_proc = ensure_server(logger)

    all_pdfs = sorted(p for p in DATA_DIR.rglob("*.pdf") if p.stat().st_size >= MIN_PDF_SIZE)
    corrupt = sorted(p for p in DATA_DIR.rglob("*.pdf") if p.stat().st_size < MIN_PDF_SIZE)
    if corrupt:
        logger.warning(f"Skipping {len(corrupt)} corrupt PDFs (<{MIN_PDF_SIZE//1024}KB)")
        for p in corrupt: log_failure(p, 0, f"Too small: {p.stat().st_size}B")

    completed, placeholder_set = load_completed_pdfs()
    pending = [p for p in all_pdfs if str(p) not in completed and str(p) not in placeholder_set]

    logger.info(f"Total: {len(all_pdfs)} | Done: {len(completed)} | "
                f"Placeholders: {len(placeholder_set)} | Pending: {len(pending)}")

    # ── PASS 1: New PDFs ──────────────────────────────────────────────────
    pdf_bar = tqdm(
        total=len(pending),
        desc="Pass 1",
        unit="pdf",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} PDFs "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
    )
    pdf_bar.set_postfix(done=0, partial=0, failed=0, pages=0)
    newly_partial = []

    def _run_pdf(pdf_path):
        size_mb = pdf_path.stat().st_size / 1024 / 1024
        logger.info(
            f"── [{pdf_path.relative_to(DATA_DIR)}] ({size_mb:.1f}MB)"
        )
        return pdf_path, process_pdf(pdf_path, logger, None, worker_count=PER_PDF_WORKERS)

    with ThreadPoolExecutor(max_workers=PDF_CONCURRENCY) as pool:
        futures = [pool.submit(_run_pdf, pdf_path) for pdf_path in pending]
        done_count = 0
        for fut in as_completed(futures):
            pdf_path, status = fut.result()
            done_count += 1

            with _stats_lock:
                if status == "ok":
                    _stats["done"] += 1
                elif status == "partial":
                    _stats["partial"] += 1
                    newly_partial.append(pdf_path)
                else:
                    _stats["failed"] += 1

                pdf_bar.set_postfix(
                    done=_stats["done"],
                    partial=_stats["partial"],
                    failed=_stats["failed"],
                    pages=_stats["pages"],
                )

            save_progress(len(completed) + done_count, len(all_pdfs), pdf_path)
            pdf_bar.update(1)

            if done_count % 10 == 0:
                with _stats_lock:
                    logger.info(
                        f"── Stats [{done_count}/{len(pending)}]: "
                        f"{_stats['done']} ok | "
                        f"{_stats['partial']} partial | "
                        f"{_stats['failed']} failed | "
                        f"{_stats['pages']} pages done"
                    )

    pdf_bar.close()

    # ── PASS 2: Retry placeholders ────────────────────────────────────────
    retry_pdfs = newly_partial + [Path(p) for p in placeholder_set]
    for rpass in range(1, PDF_RETRY_PASSES + 1):
        if not retry_pdfs: break
        logger.info(f"\n{'='*70}\n  RETRY PASS {rpass} — {len(retry_pdfs)} PDFs\n{'='*70}")
        still_partial = []
        for pdf_path in tqdm(retry_pdfs, desc=f"Retry {rpass}", unit="pdf"):
            server_proc = restart_if_dead(server_proc, logger)
            status = process_pdf(
                pdf_path,
                logger,
                None,
                is_retry_pass=True,
                worker_count=PER_PDF_WORKERS,
            )
            with _stats_lock:
                if status == "ok": _stats["done"] += 1; _stats["partial"] = max(0, _stats["partial"]-1)
                else: still_partial.append(pdf_path)
        retry_pdfs = still_partial

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  BATCH COMPLETE")
    with _stats_lock:
        for k, v in _stats.items(): logger.info(f"  {k}: {v}")
    logger.info(f"  Failed log: {FAILED_CSV}")
    logger.info("=" * 70)

    save_progress(len(all_pdfs), len(all_pdfs), Path("DONE"))
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate(); _server_proc.wait()
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
