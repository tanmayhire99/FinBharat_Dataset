#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import time
import threading
import concurrent.futures
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from openai import OpenAI

# Insert parent directory to path to import qa_pipeline_v2
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from qa_pipeline_v2.chunking import Chunk, chunk_markdown_text
from qa_pipeline_v2.prompts import get_system_user_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure httpx logger to be silent on terminal (handler added in main)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.INFO)
httpx_logger.propagate = False

KEYWORDS = [
    "revenue", "profit", "loss", "cash flow", "dividend", "audit", "auditor",
    "board", "director", "risk", "remuneration", "share capital", "borrowings",
    "debt", "equity", "segment", "md&a", "management discussion", "contingent",
    "ebitda", "pat", "eps", "roce", "roe", "working capital", "inventory",
    "finance cost", "interest coverage", "related party", "pledge", "promoter",
    "note ", "schedule ", "exceptional item", "capital expenditure", "capex",
]

HIGH_PRIORITY_SECTIONS = {
    "management discussion", "directors' report", "directors report",
    "corporate governance", "notes to accounts", "financial statements",
    "profit and loss", "balance sheet", "cash flow", "auditor",
    "risk management", "related party", "shareholding pattern",
}

def _score_chunk(chunk: Chunk) -> int:
    score = 0
    if chunk.has_table: score += 2
    lower_text = chunk.text.lower()
    lower_section = (chunk.section or "").lower()
    if any(k in lower_text for k in KEYWORDS): score += 1
    if any(h in lower_section for h in HIGH_PRIORITY_SECTIONS): score += 2
    return score

class MultiKeyClient:
    def __init__(self, api_keys: List[str], model: str, base_url: str = "https://lightning.ai/api/v1", lightning_max_concurrent: int = 2):
        if not api_keys:
            raise ValueError("No API keys provided! Set LIGHTNING_API_KEYS (comma separated).")
        self.api_keys = api_keys
        self.model = model
        self.base_url = base_url
        self.clients = [OpenAI(base_url=base_url, api_key=k, max_retries=0) for k in api_keys]
        
        # NVIDIA NIM Fallback
        nvidia_keys_str = os.environ.get("NVIDIA_API_KEYS", "[ENCRYPTION_KEY]")
        self.nvidia_api_keys = [k.strip() for k in nvidia_keys_str.split(",") if k.strip()]
        self.fallback_clients = [OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=k, max_retries=0) for k in self.nvidia_api_keys]
        
        self.lock = threading.Lock()
        self.current_idx = 0
        self.nvidia_current_idx = 0
        
        # Protect Lightning AI from Rate Limits (Max concurrent)
        self.lightning_semaphore = threading.Semaphore(lightning_max_concurrent)

    def _get_client(self):
        with self.lock:
            client = self.clients[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.clients)
            return client

    def _get_fallback_client(self):
        with self.lock:
            client = self.fallback_clients[self.nvidia_current_idx]
            self.nvidia_current_idx = (self.nvidia_current_idx + 1) % len(self.fallback_clients)
            return client

    def chat(self, messages: List[Dict[str, str]], max_tokens=6000, temperature=0.3) -> str:
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            lightning_error = None
            
            # Try Lightning ONLY if a slot is available
            if self.lightning_semaphore.acquire(blocking=False):
                try:
                    client = self._get_client()
                    completion = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    lightning_error = str(e)
                    logger.warning(f"Lightning AI Failed (Attempt {attempt}). Instantly routing to NVIDIA NIM Fallback...")
                finally:
                    self.lightning_semaphore.release()
            else:
                lightning_error = "Lightning busy. Overflowing to NVIDIA."

            # If Lightning failed, immediately fallback to NVIDIA NIM
            nvidia_error = None
            try:
                fallback_client = self._get_fallback_client()
                completion = fallback_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                logger.info("Successfully recovered and generated questions using NVIDIA NIM!")
                return completion.choices[0].message.content
            except Exception as e:
                nvidia_error = str(e)
                logger.warning(f"NVIDIA NIM Fallback also failed: {nvidia_error}")

            # If we reach here, BOTH failed. Handle retry sleeping.
            if attempt == max_attempts:
                raise Exception(f"Failed after {max_attempts} attempts on both Lightning and NVIDIA.")
            
            err_str = lightning_error or ""
            if "429" in err_str or "Too Many Requests" in err_str:
                sleep_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                logger.warning(f"429 Too Many Requests. Both APIs failed. Retrying in {sleep_time:.2f}s... (Attempt {attempt}/{max_attempts})")
                time.sleep(sleep_time)
            else:
                sleep_time = (2 ** attempt)
                logger.warning(f"Both APIs failed. Retrying in {sleep_time:.2f}s... (Attempt {attempt}/{max_attempts})")
                time.sleep(sleep_time)

def _parse_and_log_qa(raw_text: str, debug_log_file: Path, bundle_id: str) -> Optional[List[Dict]]:
    if raw_text is None:
        with debug_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"bundle_id": bundle_id, "error": "LLM returned None (Empty Response)", "raw_text": None}) + "\n")
        return None
        
    text = raw_text.strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    json_str = match.group(0) if match else text
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        with debug_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"bundle_id": bundle_id, "error": f"JSON Decode Error: {e}", "raw_text": raw_text}) + "\n")
        return None

    if not isinstance(data, list):
        with debug_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"bundle_id": bundle_id, "error": "Not a JSON array", "raw_text": raw_text}) + "\n")
        return None

    valid_pairs = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        e = str(item.get("evidence", "")).strip()
        if not q or not a:
            with debug_log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"bundle_id": bundle_id, "error": "Missing question or answer", "item": item}) + "\n")
            continue
        valid_pairs.append({
            "question": q, 
            "answer": a, 
            "evidence": e,
            "question_type": str(item.get("question_type", "")).strip(),
            "requires_calculation": bool(item.get("requires_calculation", False)),
            "verification_anchors": item.get("verification_anchors", {})
        })
    
    return valid_pairs

def build_bundles(chunks: List[Chunk], min_bundle_size: int = 4) -> List[Dict]:
    if not chunks: return []
    bundles = []
    current = []
    for i, c in enumerate(chunks):
        current.append(c)
        if len(current) < min_bundle_size:
            continue
        if i == len(chunks) - 1:
            should_close = True
        else:
            next_sec = chunks[i+1].section or "root"
            curr_sec = c.section or "root"
            should_close = next_sec != curr_sec
        if should_close:
            bundles.append(_assemble_bundle(current))
            current = []
    if current:
        bundles.append(_assemble_bundle(current))
    return bundles

def _assemble_bundle(chunks: List[Chunk]) -> Dict:
    secs = list(dict.fromkeys(c.section or "root" for c in chunks))
    pages = set()
    has_table = False
    for c in chunks:
        pages.update(c.page_numbers)
        if c.has_table:
            has_table = True

    return {
        "bundle_id": f"{chunks[0].chunk_id}_to_{chunks[-1].chunk_id}",
        "chunk_count": len(chunks),
        "chunks": chunks,
        "section": " | ".join(secs),
        "score": sum(c.score for c in chunks),
        "text": "\n\n---\n\n".join(c.text for c in chunks),
        "page_numbers": sorted(list(pages)),
        "has_table": has_table
    }

def filter_and_sort_bundles(bundles: List[Dict]) -> List[Dict]:
    if not bundles: return []
    threshold = 8
    
    # Adaptive Context Window / Dynamic Bundling
    merged_bundles = []
    i = 0
    while i < len(bundles):
        b1 = bundles[i]
        
        # Check if we can merge two adjacent weak bundles (score between 8 and 10)
        if 8 <= b1["score"] <= 10 and i + 1 < len(bundles):
            b2 = bundles[i + 1]
            if 8 <= b2["score"] <= 10:
                # Merge them
                combined_chunks = b1["chunks"] + b2["chunks"]
                merged_b = _assemble_bundle(combined_chunks)
                merged_b["merged_weak_bundle"] = True
                merged_bundles.append(merged_b)
                i += 2
                continue
        
        # If no merge, just add b1
        b1["merged_weak_bundle"] = False
        merged_bundles.append(b1)
        i += 1
        
    filtered = [b for b in merged_bundles if b["score"] >= threshold]
    filtered.sort(key=lambda b: b["chunks"][0].chunk_id)
    return filtered

def _chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "sector": chunk.sector,
        "company": chunk.company,
        "year": chunk.year,
        "section": chunk.section,
        "page_numbers": chunk.page_numbers,
        "has_table": chunk.has_table,
        "token_count": chunk.token_count,
        "oversized_table": chunk.oversized_table,
        "score": getattr(chunk, 'score', 0),
        "text": chunk.text,
    }

def process_company(md_path: Path, sector: str, company: str, year: str, out_dir: Path, client: MultiKeyClient, difficulties: List[str], min_bundle_size: int, max_workers: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_path = out_dir / "chunks.jsonl"
    chunk_scores_path = out_dir / "chunk_scoring_details.jsonl"
    
    if chunks_path.exists():
        logger.info(f"Loading existing chunks from {chunks_path}")
        chunks = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                c_dict = json.loads(line)
                chunks.append(Chunk(**c_dict))
    else:
        logger.info(f"Reading markdown and chunking {md_path.name}")
        with md_path.open("r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_markdown_text(doc_id=f"{sector}/{company}/{year}", sector=sector, company=company, year=year, markdown_text=text, target_tokens=800, max_tokens=900)
        
        with chunk_scores_path.open("w", encoding="utf-8") as fs:
            for c in chunks: 
                c.score = _score_chunk(c)
                
                # Write debug scoring details
                details = {"chunk_id": c.chunk_id, "section": c.section, "total_score": c.score}
                if c.has_table: details["table_bonus"] = 2
                fk = [k for k in KEYWORDS if k in c.text.lower()]
                if fk: details["found_keywords"] = fk
                fhp = [h for h in HIGH_PRIORITY_SECTIONS if h in (c.section or "").lower()]
                if fhp: details["found_high_priority_sections"] = fhp
                fs.write(json.dumps(details, ensure_ascii=True) + "\n")
                
        with chunks_path.open("w", encoding="utf-8") as f:
            for c in chunks: 
                f.write(json.dumps(_chunk_to_dict(c)) + "\n")

    bundles = build_bundles(chunks, min_bundle_size)
    filtered = filter_and_sort_bundles(bundles)
    
    bundle_details_path = out_dir / "semantic_bundles_details.jsonl"
    with bundle_details_path.open("w", encoding="utf-8") as f:
        for b in filtered:
            f.write(json.dumps({"bundle_id": b["bundle_id"], "score": b["score"], "chunk_count": b["chunk_count"], "section": b["section"]}) + "\n")

    for difficulty in difficulties:
        diff_dir = out_dir / difficulty
        diff_dir.mkdir(parents=True, exist_ok=True)
        
        qa_file = diff_dir / f"{difficulty}_qa_pairs.jsonl"
        processed_log = diff_dir / ".processed_bundles.log"
        debug_log = diff_dir / "debug_hallucinations.log"

        processed = set()
        if processed_log.exists():
            with processed_log.open("r", encoding="utf-8") as f:
                for line in f: processed.add(line.strip())

        write_lock = threading.Lock()

        def worker(bundle):
            if bundle["bundle_id"] in processed:
                return
            
            score = bundle["score"]
            
            if bundle.get("merged_weak_bundle", False):
                max_q = 2
            elif score >= 18:
                max_q = 4
            elif score >= 11:
                max_q = 2
            else:
                max_q = 1

            sys_msg, usr_msg = get_system_user_messages(
                difficulty=difficulty,
                sector=sector,
                company=company,
                year=year,
                section=bundle["section"],
                chunk_text=bundle["text"],
                max_questions=max_q
            )
            try:
                max_json_retries = 3
                valid_pairs = None
                
                for json_attempt in range(1, max_json_retries + 1):
                    raw = client.chat([{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}])
                    valid_pairs = _parse_and_log_qa(raw, debug_log, bundle["bundle_id"])
                    
                    if valid_pairs is not None:
                        logger.info(f"Successfully generated {difficulty} QA for bundle {bundle['bundle_id']}")
                        break
                    else:
                        logger.warning(f"JSON Hallucination on {bundle['bundle_id']}. Retrying ({json_attempt}/{max_json_retries})...")
                
                if valid_pairs is None:
                    logger.error(f"Failed to generate valid JSON for {bundle['bundle_id']} after {max_json_retries} attempts.")
                    valid_pairs = []

                with write_lock:
                    if valid_pairs:
                        with qa_file.open("a", encoding="utf-8") as f:
                            for pair in valid_pairs:
                                out = {
                                    "question": pair["question"],
                                    "answer": pair["answer"],
                                    "evidence": pair["evidence"],
                                    "question_type": pair.get("question_type", ""),
                                    "requires_calculation": pair.get("requires_calculation", False),
                                    "verification_anchors": pair.get("verification_anchors", {}),
                                    "sector": sector,
                                    "company": company,
                                    "year": year,
                                    "section": bundle["section"],
                                    "page_numbers": bundle.get("page_numbers", []),
                                    "bundle_id": bundle["bundle_id"],
                                    "difficulty": difficulty
                                }
                                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    # Log as processed whether valid pairs were returned or not
                    with processed_log.open("a", encoding="utf-8") as f:
                        f.write(bundle["bundle_id"] + "\n")
            except Exception as e:
                logger.error(f"Failed on bundle {bundle['bundle_id']} ({difficulty}): {e}")

        remaining = [b for b in filtered if b["bundle_id"] not in processed]
        logger.info(f"Generating QA for {difficulty} level. Processing {len(remaining)} remaining bundles concurrently...")
        if remaining:
            bundle_pbar = tqdm(total=len(remaining), desc=f"  Bundles ({difficulty})", unit="bundle", leave=False)
            completed_lock = threading.Lock()

            def tracked_worker(bundle):
                worker(bundle)
                with completed_lock:
                    bundle_pbar.update(1)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(tracked_worker, b) for b in remaining]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            bundle_pbar.close()

    # Calculate total questions and hallucinations for this specific FY
    total_company_questions = 0
    total_company_hallucinations = 0
    
    questions_by_diff = {}
    hallucinations_by_diff = {}
    
    questions_by_type = {}
    
    for difficulty in difficulties:
        qa_file = out_dir / difficulty / f"{difficulty}_qa_pairs.jsonl"
        questions_by_diff[difficulty] = 0
        if qa_file.exists():
            with qa_file.open("r", encoding="utf-8") as f:
                for line in f:
                    total_company_questions += 1
                    questions_by_diff[difficulty] += 1
                    try:
                        data = json.loads(line)
                        qt = data.get("question_type", "Unknown")
                        questions_by_type[qt] = questions_by_type.get(qt, 0) + 1
                    except json.JSONDecodeError:
                        pass
                
        debug_file = out_dir / difficulty / "debug_hallucinations.log"
        hallucinations_by_diff[difficulty] = 0
        if debug_file.exists():
            with debug_file.open("r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
                total_company_hallucinations += count
                hallucinations_by_diff[difficulty] = count
                
    # Write local count log
    local_count_file = out_dir / "question_count_hard_multihop.log"
    with local_count_file.open("w", encoding="utf-8") as f:
        f.write(f"Total Hard/Multihop Questions for {year}: {total_company_questions}\n")
        f.write("\nBy Difficulty:\n")
        for d, c in questions_by_diff.items():
            f.write(f"- {d}: {c}\n")
        f.write("\nBy Question Type:\n")
        for qt, c in questions_by_type.items():
            f.write(f"- {qt}: {c}\n")

    local_hal_file = out_dir / "hallucination_count_hard_multihop.log"
    with local_hal_file.open("w", encoding="utf-8") as f:
        f.write(f"Total Hard/Multihop Hallucinations for {year}: {total_company_hallucinations}\n")
        for d, c in hallucinations_by_diff.items():
            f.write(f"- {d}: {c}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default="lightning-ai/gpt-oss-120b")
    parser.add_argument("--base-url", default="https://lightning.ai/api/v1")
    parser.add_argument("--difficulty", default="all", choices=["hard", "multihop", "all"])
    parser.add_argument("--bundle-size", type=int, default=4)
    args = parser.parse_args()

    # Determine keys
    keys_str = os.environ.get("LIGHTNING_API_KEYS", "")
    if keys_str:
        api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    else:
        single_key = os.environ.get("LIGHTNING_API_KEY", "")
        if single_key:
            api_keys = [single_key.strip()]
        else:
            raise ValueError("No API keys found. Please set LIGHTNING_API_KEYS (comma separated).")

    logger.info(f"Loaded {len(api_keys)} Lightning API keys for Hard/Multihop generation.")
    
    # Configurable Variables to Decide API Worker Allocation
    lightning_workers = len(api_keys)  # Automatically match number of lightning keys
    
    # Initialize the client
    client = MultiKeyClient(api_keys=api_keys, model=args.model, base_url=args.base_url, lightning_max_concurrent=lightning_workers)
    
    # Automatically scale NVIDIA workers based on the number of keys (8 workers per key)
    nvidia_keys_count = len(client.nvidia_api_keys)
    nvidia_workers = nvidia_keys_count * 8                 
    max_workers = lightning_workers + nvidia_workers
    
    logger.info(f"Loaded {nvidia_keys_count} NVIDIA API keys. Auto-scaling NVIDIA workers to {nvidia_workers}.")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Set up server_requests.log in the output root
    server_log_path = output_root / "server_requests_hard_multihop.log"
    _httpx_fh = logging.FileHandler(server_log_path, encoding="utf-8")
    _httpx_fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logging.getLogger("httpx").addHandler(_httpx_fh)
    
    difficulties = ["hard", "multihop"] if args.difficulty == "all" else [args.difficulty]

    all_reports = []
    for sector_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        sector = sector_dir.name
        if sector.startswith((".", "_")): continue

        for company_dir in sorted(p for p in sector_dir.iterdir() if p.is_dir()):
            company = company_dir.name
            
            for md_path in sorted(company_dir.glob("*.md")):
                all_reports.append((md_path, sector, company))

    pbar = tqdm(all_reports, desc="Annual Reports (Hard/Multihop)", unit="report")
    for md_path, sector, company in pbar:
        report_start = time.time()

        # Assuming naming is FY2025.md or 2025.md
        year = md_path.stem
        if re.match(r"^(19|20)\d{2}$", year):
            year = f"FY{year}"

        pbar.set_postfix_str(f"{company}/{year}", refresh=True)

        out_dir = output_root / sector / company / year
        
        logger.info(f"Processing Hard/Multihop QA for {sector} / {company} / {year}")
        process_company(
            md_path=md_path,
            sector=sector,
            company=company,
            year=year,
            out_dir=out_dir,
            client=client,
            difficulties=difficulties,
            min_bundle_size=args.bundle_size,
            max_workers=max_workers
        )

        elapsed = time.time() - report_start
        pbar.set_postfix_str(f"{company}/{year} done in {elapsed:.1f}s", refresh=True)
        
        # Update global count dynamically
        global_total = 0
        global_questions_by_type = {}
        for qa_file in output_root.rglob("*_qa_pairs.jsonl"):
            # Only count hard and multihop in this global log if wanted, but standard keeps all files
            if "hard" in qa_file.name or "multihop" in qa_file.name:
                with qa_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        global_total += 1
                        try:
                            data = json.loads(line)
                            qt = data.get("question_type", "Unknown")
                            global_questions_by_type[qt] = global_questions_by_type.get(qt, 0) + 1
                        except json.JSONDecodeError:
                            pass
        with (output_root / "global_question_count_hard_multihop.log").open("w", encoding="utf-8") as f:
            f.write(f"Global Total Hard/Multihop Questions Generated: {global_total}\n")
            f.write("\nBy Question Type:\n")
            for qt, count in global_questions_by_type.items():
                f.write(f"- {qt}: {count}\n")
            
        # Update global hallucination count dynamically
        global_hal_total = 0
        for hal_file in output_root.rglob("debug_hallucinations.log"):
            if "hard" in str(hal_file) or "multihop" in str(hal_file):
                with hal_file.open("r", encoding="utf-8") as f:
                    global_hal_total += sum(1 for _ in f)
        with (output_root / "global_hallucination_count_hard_multihop.log").open("w", encoding="utf-8") as f:
            f.write(f"Global Total Hard/Multihop Hallucinations: {global_hal_total}\n")
                
    logger.info("Hard/Multihop QA Generation Complete.")

if __name__ == "__main__":
    main()
