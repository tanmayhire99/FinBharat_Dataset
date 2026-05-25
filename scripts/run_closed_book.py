#!/usr/bin/env python3
"""
FinBharat — Closed-Book Evaluation: All Models
================================================
Runs closed-book evaluation across all 18 verified NIM models,
4 difficulty tiers, and saves incremental progress so a crash
never loses completed work.

Logging:
  logs/run_master.log          — one line per completed run (TSV)
  logs/{model}_{diff}.log      — detailed per-question log with
                                  sector, company, gold, pred, all metrics
  results/aggregates/          — JSON aggregate per model×diff×regime
  results/eval_results/        — per-question JSONL scores
  results/generations/         — cached model outputs (resume-safe)

Usage:
  uv run python scripts/run_closed_book.py
  uv run python scripts/run_closed_book.py --max-per-company 5
  uv run python scripts/run_closed_book.py --models llama3.1-8b llama3.3-70b
  uv run python scripts/run_closed_book.py --difficulty easy medium
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finbharat.data.loader import FinBharatDataset, SAMPLE_COMPANIES
from finbharat.eval.evaluate import (
    evaluate_qa_records, aggregate_results,
    save_results, save_aggregate, run_llm_judge_pass,
)
from finbharat.models.runner import ModelRunner, PREDEFINED_MODELS
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_ROOT    = Path(__file__).parent.parent / "dataset"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
LOGS_DIR     = Path(__file__).parent.parent / "logs"
MASTER_LOG   = LOGS_DIR / "run_master.log"

REGIME = "closed_book"

# All 18 confirmed working models (as of May 2026)
ALL_MODELS = [
    # Small
    "llama3.1-8b",
    "llama3.2-3b",
    "nemotron-nano-8b",
    # phi4-mini removed — consistently times out on this NIM account
    "gpt-oss-20b",
    # Medium
    "llama4-maverick",
    "nemotron-super-49b",
    "gemma4-31b",
    "mistral-nemotron",
    # Large
    "llama3.3-70b",
    "nemotron-120b",
    "gpt-oss-120b",
    "mistral-small-119b",
    # Very Large / MoE
    "qwen3.5-122b",
    "qwen3.5-397b",
    "mistral-large-675b",
    # DeepSeek
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]

ALL_DIFFICULTIES = ["easy", "medium", "hard", "multihop"]


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_master_log():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not MASTER_LOG.exists():
        with open(MASTER_LOG, "w") as f:
            f.write("timestamp\tmodel\tdifficulty\tregime\ttotal_q\terrors\t"
                    "exact_match\trelaxed_em\ttoken_f1\tnum_exact\tmape\t"
                    "abstain_rate\tentailment\ttraceability\tstatus\n")


def log_master(model_key: str, difficulty: str, agg: dict, status: str):
    with open(MASTER_LOG, "a") as f:
        f.write("\t".join([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_key, difficulty, REGIME,
            str(agg.get("total", 0)),
            str(agg.get("num_errors", 0)),
            str(agg.get("exact_match", "")),
            str(agg.get("relaxed_em", "")),
            str(agg.get("token_f1", "")),
            str(agg.get("num_exact", "")),
            str(agg.get("mape", "")),
            str(agg.get("abstain_rate", "")),
            str(agg.get("entailment_ratio", "")),
            str(agg.get("evidence_traceability", "")),
            status,
        ]) + "\n")


def setup_run_logger(model_key: str, difficulty: str) -> logging.Logger:
    """Create a per-run logger that writes to logs/{model}_{diff}.log"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"{model_key}_{difficulty}_{REGIME}"
    log_file = LOGS_DIR / f"{log_name}.log"

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Also echo INFO+ to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


# ── Skip-if-done check ─────────────────────────────────────────────────────────

def already_done(model_key: str, difficulty: str) -> bool:
    """Return True if aggregate JSON already exists for this run."""
    tag = f"{model_key}_{difficulty}_{REGIME}"
    agg_path = RESULTS_DIR / "aggregates" / f"{tag}.json"
    return agg_path.exists()


# ── Per-company / per-sector detailed log ──────────────────────────────────────

def write_detailed_log(
    logger: logging.Logger,
    qa_records,
    eval_results,
    model_key: str,
    difficulty: str,
):
    """Write one log line per question with sector, company, metrics."""
    logger.info("=" * 100)
    logger.info(f"DETAILED RESULTS  model={model_key}  difficulty={difficulty}  regime={REGIME}")
    logger.info("=" * 100)

    # Group by sector → company
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for qa, res in zip(qa_records, eval_results):
        groups[qa.sector][qa.company].append((qa, res))

    for sector in sorted(groups):
        logger.info(f"\n── SECTOR: {sector} {'─'*60}")
        for company in sorted(groups[sector]):
            pairs = groups[sector][company]
            em_avg = sum(r.exact_match for _, r in pairs) / len(pairs)
            f1_avg = sum(r.token_f1    for _, r in pairs) / len(pairs)
            ne_avg = sum(r.num_exact   for _, r in pairs) / len(pairs)
            ab_cnt = sum(1 for _, r in pairs if r.abstained)
            logger.info(
                f"  COMPANY: {company:30s}  "
                f"n={len(pairs):3d}  EM={em_avg:.3f}  F1={f1_avg:.3f}  "
                f"Num-EM={ne_avg:.3f}  Abstains={ab_cnt}"
            )
            for qa, res in pairs:
                icon = "✅" if res.exact_match else ("🤷" if res.abstained else "❌")
                logger.debug(
                    f"    {icon} [{qa.question_type:20s}] "
                    f"Q: {qa.question[:60]!r}  "
                    f"Gold: {qa.answer[:40]!r}  "
                    f"Pred: {res.predicted_answer[:40]!r}  "
                    f"EM={res.exact_match} F1={res.token_f1:.3f} "
                    f"NE={res.num_exact} Abs={res.abstained}"
                )


# ── Single run ─────────────────────────────────────────────────────────────────

def run_one(
    model_key: str,
    difficulty: str,
    companies: list[dict],
    max_per_company: int | None,
) -> bool:
    """
    Run one (model, difficulty) pair. Returns True on success.
    Skips silently if already done. Catches all errors and logs them.
    """
    tag = f"{model_key}_{difficulty}_{REGIME}"

    if already_done(model_key, difficulty):
        print(f"  ⏭  SKIP  {tag}  (already done)")
        return True

    logger = setup_run_logger(model_key, difficulty)
    logger.info(f"\n{'='*80}")
    logger.info(f"START  {tag}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  companies={[c['company'] for c in companies]}  max_per_company={max_per_company}")

    config = PREDEFINED_MODELS.get(model_key)
    if config is None:
        logger.error(f"Unknown model key: {model_key}")
        return False

    try:
        dataset = FinBharatDataset(DATA_ROOT)
        qa_records = dataset.load_sample(
            companies=companies,
            difficulty=difficulty,
            max_per_company=max_per_company,
        )
        logger.info(f"  Loaded {len(qa_records)} QA records")

        # Build contexts (in closed_book, still build but runner ignores them)
        contexts = []
        for qa in qa_records:
            ctx = dataset.build_context_for_qa(qa)
            if not ctx:
                ctx = qa.evidence
            contexts.append(ctx)

        cache_path  = RESULTS_DIR / "generations"  / f"{tag}.jsonl"
        eval_path   = RESULTS_DIR / "eval_results" / f"{tag}.jsonl"
        agg_path    = RESULTS_DIR / "aggregates"   / f"{tag}.json"

        t0 = time.time()
        runner = ModelRunner(config, regime=REGIME)
        try:
            generations = runner.generate_batch(qa_records, contexts, cache_path=cache_path)
        finally:
            runner.close()

        elapsed_gen = time.time() - t0
        errors = sum(1 for g in generations if g.error)
        logger.info(f"  Generation done in {elapsed_gen:.1f}s  errors={errors}")

        eval_results = evaluate_qa_records(qa_records, generations, regime=REGIME)
        agg = aggregate_results(eval_results)
        agg["regime"] = REGIME

        save_results(eval_results, eval_path)
        save_aggregate(agg, agg_path)

        # Detailed per-company/sector log
        write_detailed_log(logger, qa_records, eval_results, model_key, difficulty)

        # Summary
        logger.info(
            f"\n  SUMMARY  EM={agg['exact_match']:.4f}  "
            f"Rel.EM={agg['relaxed_em']:.4f}  F1={agg['token_f1']:.4f}  "
            f"Num-EM={agg['num_exact']:.4f}  Abstain={agg['abstain_rate']:.4f}  "
            f"NLI={agg['entailment_ratio']:.4f}  Trace={agg['evidence_traceability']:.4f}"
        )
        logger.info(f"  Results → {agg_path}")
        logger.info(f"  Log     → {LOGS_DIR / f'{tag}.log'}")

        log_master(model_key, difficulty, agg, "OK")
        return True

    except Exception as e:
        import traceback
        logger.error(f"FAILED: {e}")
        logger.error(traceback.format_exc())
        log_master(model_key, difficulty, {}, f"ERROR: {e}")
        return False


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run closed-book evaluation for all models")
    parser.add_argument("--models",           nargs="+", default=ALL_MODELS,
                        help="Model keys to run (default: all 17 NIM models)")
    parser.add_argument("--difficulty",       nargs="+", default=ALL_DIFFICULTIES,
                        dest="difficulties",
                        help="Difficulty tiers (default: all 4)")
    parser.add_argument("--max-per-company",  type=int,  default=10,
                        help="Max QA pairs per company (default: 10)")
    parser.add_argument("--all-companies",    action="store_true",
                        help="Run on all 187 companies instead of 3 sample")
    # vLLM local model support
    parser.add_argument("--vllm-model",       type=str,  default=None,
                        help="Run against a local vLLM server instead of NIM. "
                             "Pass the model ID as served by vLLM "
                             "(e.g. 'fingpt' or 'meta-llama/Meta-Llama-3-8B').")
    parser.add_argument("--vllm-host",        type=str,  default="localhost",
                        help="Host where vLLM is running (default: localhost).")
    parser.add_argument("--vllm-port",        type=int,  default=8000,
                        help="Port where vLLM listens (default: 8000).")
    parser.add_argument("--vllm-completion",  action="store_true", default=False,
                        help="Use /completions endpoint with Alpaca format (e.g. for FinMA).")
    parser.add_argument("--vllm-max-context", type=int,  default=0,
                        help="Truncate context to N chars (e.g. 2000 for FinMA's 2048-token limit).")
    parser.add_argument("--workers",          type=int,  default=1,
                        help="Parallel workers. Each worker uses a different NVIDIA_API_KEY "
                             "(e.g. --workers 3 uses KEY, KEY_1, KEY_2 simultaneously). "
                             "Safe max = number of API keys you have. Default: 1 (sequential).")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_master_log()

    # vLLM: register model dynamically and replace the model list
    if args.vllm_model:
        from finbharat.models.runner import make_vllm_config, PREDEFINED_MODELS
        cfg = make_vllm_config(
            args.vllm_model,
            host=args.vllm_host,
            port=args.vllm_port,
            use_completion=args.vllm_completion,
            max_context_chars=args.vllm_max_context,
        )
        PREDEFINED_MODELS["__vllm__"] = cfg
        args.models = ["__vllm__"]
        print(f"Using local vLLM: {args.vllm_model} at {cfg.api_base}")

    dataset = FinBharatDataset(DATA_ROOT)
    if args.all_companies:
        companies = dataset.list_available_companies("e_m")
        print(f"Running on all {len(companies)} companies")
    else:
        companies = SAMPLE_COMPANIES
        print(f"Running on {len(companies)} sample companies: {[c['company'] for c in companies]}")

    # Build flat list of (model, difficulty) pairs — skip already-done
    all_pairs = [
        (model_key, difficulty)
        for model_key in args.models
        for difficulty in args.difficulties
        if not already_done(model_key, difficulty)
    ]
    skipped   = len(args.models) * len(args.difficulties) - len(all_pairs)
    total     = len(all_pairs)
    workers   = min(args.workers, total) if total > 0 else 1

    print(f"\n{'='*70}")
    print(f"  FinBharat Closed-Book Evaluation")
    print(f"  {len(args.models)} models × {len(args.difficulties)} tiers")
    print(f"  Pending: {total} runs  |  Skipped (done): {skipped}")
    print(f"  Workers: {workers}  (each uses a separate API key)")
    print(f"  max_per_company={args.max_per_company}")
    print(f"  Logs   → {LOGS_DIR}/")
    print(f"  Results → {RESULTS_DIR}/")
    print(f"{'='*70}\n")

    if total == 0:
        print("Nothing to run — all done!")
        return

    t_start  = time.time()
    done     = 0
    failed   = 0
    counter  = {"n": 0}
    lock     = __import__("threading").Lock()

    def _worker(pair):
        model_key, difficulty = pair
        with lock:
            counter["n"] += 1
            idx = counter["n"]
        print(f"\n[{idx}/{total}] {model_key}_{difficulty}_{REGIME}")
        ok = run_one(model_key, difficulty, companies, args.max_per_company)
        return ok

    if workers == 1:
        # Sequential — simpler output
        for pair in all_pairs:
            ok = _worker(pair)
            if ok: done += 1
            else:  failed += 1
    else:
        # Parallel — assign each worker a distinct API key index via env
        import concurrent.futures, os
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")

        # Collect available keys
        base_keys = []
        base = os.environ.get("NVIDIA_API_KEY","").strip()
        if base: base_keys.append(base)
        for i in range(1, 10):
            k = os.environ.get(f"NVIDIA_API_KEY_{i}","").strip()
            if k: base_keys.append(k)

        if not base_keys:
            print("  ⚠  No API keys found — falling back to sequential")
            for pair in all_pairs:
                ok = _worker(pair)
                if ok: done += 1
                else:  failed += 1
        else:
            print(f"  API keys available: {len(base_keys)} → using {min(workers,len(base_keys))} in parallel")
            workers = min(workers, len(base_keys))

            def _worker_with_key(args_tuple):
                pair, worker_idx = args_tuple
                # Set env key for this thread so the runner picks it up
                key = base_keys[worker_idx % len(base_keys)]
                os.environ["NVIDIA_API_KEY"] = key
                return _worker(pair)

            indexed_pairs = [(pair, i % workers) for i, pair in enumerate(all_pairs)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_worker_with_key, p): p for p in indexed_pairs}
                for fut in concurrent.futures.as_completed(futures):
                    ok = fut.result()
                    if ok: done += 1
                    else:  failed += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  DONE in {elapsed/60:.1f} min")
    print(f"  Completed: {done}  Skipped: {skipped}  Failed: {failed}")
    print(f"  Master log: {MASTER_LOG}")
    print(f"  Run: uv run python main.py analyze --results-dir results")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
