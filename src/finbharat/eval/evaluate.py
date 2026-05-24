import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from finbharat.data.loader import QARecord, FinBharatDataset, SAMPLE_COMPANIES
from finbharat.metrics.numeric import compute_numeric_metrics, compute_tolerance_accuracy, compute_mape
from finbharat.metrics.text import (
    compute_exact_match, compute_token_f1, compute_relaxed_em,
    compute_directional_accuracy, compute_rouge_l,
)
from finbharat.metrics.faithfulness import compute_nli_entailment, compute_evidence_traceability
from finbharat.models.runner import ModelRunner, GenerationResult, PREDEFINED_MODELS, VALID_REGIMES


@dataclass
class EvalResult:
    question_id: str
    model_name: str
    difficulty: str
    question_type: str
    sector: str
    company: str
    gold_answer: str
    predicted_answer: str
    exact_match: int
    relaxed_em: int
    token_f1: float
    rouge_l: float
    bertscore_f1: float
    num_exact: int
    num_f1: float
    tol1_acc: int
    tol5_acc: int
    tol10_acc: int
    directional_acc: Optional[int]
    entailment_ratio: float
    evidence_traceability: float
    latency_ms: float
    regime: str = "zero_shot"
    error: Optional[str] = None


def compute_bertscore_batch(golds: list[str], preds: list[str]) -> list[float]:
    """Return per-pair BERTScore F1. Falls back to 0.0 if bert_score not installed."""
    try:
        from bert_score import score as bs_score
        _, _, F = bs_score(preds, golds, lang="en", verbose=False, device="cpu")
        return [round(float(f), 4) for f in F]
    except ImportError:
        return [0.0] * len(golds)
    except Exception:
        return [0.0] * len(golds)


def evaluate_single(gold: str, pred: str, evidence: str, gold_evidence: str) -> dict:
    text_res = compute_token_f1(gold, pred)
    rouge = compute_rouge_l(gold, pred)
    num_res = compute_numeric_metrics(gold, pred)
    tol1 = compute_tolerance_accuracy(gold, pred, tolerance_pct=1.0)
    tol5 = compute_tolerance_accuracy(gold, pred, tolerance_pct=5.0)
    tol10 = compute_tolerance_accuracy(gold, pred, tolerance_pct=10.0)
    dir_acc = compute_directional_accuracy(gold, pred)
    rlx_em = compute_relaxed_em(gold, pred)
    nli_res = compute_nli_entailment(evidence, pred)
    trace = compute_evidence_traceability(evidence, gold_evidence)
    return {
        "exact_match": text_res.exact_match,
        "relaxed_em": rlx_em,
        "token_f1": text_res.f1,
        "rouge_l": rouge,
        "bertscore_f1": 0.0,  # filled in batch after generation
        "num_exact": num_res.num_exact,
        "num_f1": num_res.num_f1,
        "tol1_acc": tol1,
        "tol5_acc": tol5,
        "tol10_acc": tol10,
        "directional_acc": dir_acc,
        "entailment_ratio": nli_res.entailment_ratio,
        "evidence_traceability": trace,
    }


def evaluate_qa_records(
    qa_records: list[QARecord],
    generations: list[GenerationResult],
    regime: str = "zero_shot",
) -> list[EvalResult]:
    results = []
    for qa, gen in zip(qa_records, generations):
        if gen.error:
            results.append(EvalResult(
                question_id=gen.question_id,
                model_name=gen.model_name,
                difficulty=qa.difficulty,
                question_type=qa.question_type,
                sector=qa.sector,
                company=qa.company,
                gold_answer=qa.answer,
                predicted_answer="",
                exact_match=0, relaxed_em=0, token_f1=0.0, rouge_l=0.0, bertscore_f1=0.0,
                num_exact=0, num_f1=0.0, tol1_acc=0, tol5_acc=0, tol10_acc=0,
                directional_acc=None, entailment_ratio=0.0,
                evidence_traceability=0.0,
                latency_ms=gen.latency_ms,
                regime=regime,
                error=gen.error,
            ))
            continue
        metrics = evaluate_single(qa.answer, gen.predicted_answer, qa.evidence, qa.evidence)
        results.append(EvalResult(
            question_id=gen.question_id,
            model_name=gen.model_name,
            difficulty=qa.difficulty,
            question_type=qa.question_type,
            sector=qa.sector,
            company=qa.company,
            gold_answer=qa.answer,
            predicted_answer=gen.predicted_answer,
            latency_ms=gen.latency_ms,
            regime=regime,
            **metrics,
        ))

    # Batch BERTScore — much faster than per-call
    valid = [r for r in results if not r.error]
    if valid:
        bs_scores = compute_bertscore_batch(
            [r.gold_answer for r in valid],
            [r.predicted_answer for r in valid],
        )
        for r, bs in zip(valid, bs_scores):
            r.bertscore_f1 = bs

    return results


def aggregate_results(results: list[EvalResult]) -> dict:
    if not results:
        return {}
    n = len(results)
    valid_dir = [r for r in results if r.directional_acc is not None]
    mape = compute_mape(
        [r.gold_answer for r in results],
        [r.predicted_answer for r in results],
    )
    agg = {
        "total": n,
        "num_errors": sum(1 for r in results if r.error),
        "exact_match": round(sum(r.exact_match for r in results) / n, 4),
        "relaxed_em": round(sum(r.relaxed_em for r in results) / n, 4),
        "token_f1": round(sum(r.token_f1 for r in results) / n, 4),
        "rouge_l": round(sum(r.rouge_l for r in results) / n, 4),
        "bertscore_f1": round(sum(r.bertscore_f1 for r in results) / n, 4),
        "num_exact": round(sum(r.num_exact for r in results) / n, 4),
        "num_f1": round(sum(r.num_f1 for r in results) / n, 4),
        "tol1_acc": round(sum(r.tol1_acc for r in results) / n, 4),
        "tol5_acc": round(sum(r.tol5_acc for r in results) / n, 4),
        "tol10_acc": round(sum(r.tol10_acc for r in results) / n, 4),
        "mape": round(mape, 4) if mape is not None else None,
        "entailment_ratio": round(sum(r.entailment_ratio for r in results) / n, 4),
        "evidence_traceability": round(sum(r.evidence_traceability for r in results) / n, 4),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
    }
    if valid_dir:
        agg["directional_accuracy"] = round(sum(r.directional_acc for r in valid_dir) / len(valid_dir), 4)
        agg["directional_applicable"] = len(valid_dir)

    def _init_bucket() -> dict:
        return {"total": 0, "exact_match": 0, "relaxed_em": 0, "token_f1": 0.0,
                "rouge_l": 0.0, "bertscore_f1": 0.0, "num_exact": 0, "num_f1": 0.0}

    def _acc_bucket(b: dict, r: "EvalResult"):
        b["total"] += 1
        b["exact_match"] += r.exact_match
        b["relaxed_em"] += r.relaxed_em
        b["token_f1"] += r.token_f1
        b["rouge_l"] += r.rouge_l
        b["bertscore_f1"] += r.bertscore_f1
        b["num_exact"] += r.num_exact
        b["num_f1"] += r.num_f1

    def _avg_bucket(b: dict):
        t = b["total"]
        for k in ("exact_match", "relaxed_em", "token_f1", "rouge_l", "bertscore_f1", "num_exact", "num_f1"):
            b[k] = round(b[k] / t, 4)

    by_qtype: dict[str, dict] = {}
    for r in results:
        qt = r.question_type
        if qt not in by_qtype:
            by_qtype[qt] = _init_bucket()
        _acc_bucket(by_qtype[qt], r)
    for d in by_qtype.values():
        _avg_bucket(d)
    agg["by_question_type"] = by_qtype

    by_difficulty: dict[str, dict] = {}
    for r in results:
        diff = r.difficulty
        if diff not in by_difficulty:
            by_difficulty[diff] = _init_bucket()
        _acc_bucket(by_difficulty[diff], r)
    for d in by_difficulty.values():
        _avg_bucket(d)
    agg["by_difficulty"] = by_difficulty

    by_company: dict[str, dict] = {}
    for r in results:
        c = r.company
        if c not in by_company:
            by_company[c] = _init_bucket()
        _acc_bucket(by_company[c], r)
    for d in by_company.values():
        _avg_bucket(d)
    agg["by_company"] = by_company

    return agg


def save_results(results: list[EvalResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), default=str) + "\n")


def save_aggregate(agg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(agg, f, indent=2, default=str)


def run_evaluation(
    data_root: Path,
    output_dir: Path,
    model_keys: list[str] = ("llama3.1-8b",),
    difficulty: str = "easy",
    regime: str = "zero_shot",
    companies: list[dict] | None = None,
    all_companies: bool = False,
    max_per_company: int | None = None,
    api_key: str | None = None,
):
    dataset = FinBharatDataset(data_root)

    if all_companies:
        source = "e_m" if difficulty in ("easy", "medium") else "h_m"
        companies = dataset.list_available_companies(source=source)
        print(f"Running on all {len(companies)} companies ({source})")

    qa_records = dataset.load_sample(companies=companies, difficulty=difficulty, max_per_company=max_per_company)

    print(f"Loaded {len(qa_records)} QA records for difficulty={difficulty}")
    _print_qa_stats(qa_records)

    contexts = []
    for qa in tqdm(qa_records, desc="Building contexts", unit="q"):
        ctx = dataset.build_context_for_qa(qa)
        if not ctx:
            ctx = qa.evidence
        contexts.append(ctx)

    for model_key in model_keys:
        config = PREDEFINED_MODELS.get(model_key)
        if config is None:
            print(f"Unknown model key: {model_key}. Available: {list(PREDEFINED_MODELS.keys())}")
            continue

        print(f"\nRunning model: {config.name} | regime: {regime}")
        tag = f"{model_key}_{difficulty}_{regime}"
        cache_path = output_dir / "generations" / f"{tag}.jsonl"
        eval_path = output_dir / "eval_results" / f"{tag}.jsonl"
        agg_path = output_dir / "aggregates" / f"{tag}.json"

        runner = ModelRunner(config, api_key=api_key, regime=regime)
        try:
            generations = runner.generate_batch(qa_records, contexts, cache_path=cache_path)
            eval_results = evaluate_qa_records(qa_records, generations, regime=regime)
            agg = aggregate_results(eval_results)
            agg["regime"] = regime

            save_results(eval_results, eval_path)
            save_aggregate(agg, agg_path)

            _print_aggregate(config.name, agg)
        finally:
            runner.close()

    print(f"\nResults saved to {output_dir}")


def _print_qa_stats(records: list[QARecord]):
    from collections import Counter
    qtypes = Counter(r.question_type for r in records)
    companies = Counter(r.company for r in records)
    print(f"  Companies: {dict(companies)}")
    print(f"  Question types: {dict(qtypes)}")


def _print_aggregate(model_name: str, agg: dict):
    print(f"\n  === {model_name} Results ===")
    print(f"  Total: {agg['total']} | Errors: {agg.get('num_errors', 0)}")
    print(f"  EM: {agg['exact_match']:.4f} | Relaxed EM: {agg['relaxed_em']:.4f} | Token F1: {agg['token_f1']:.4f} | ROUGE-L: {agg['rouge_l']:.4f} | BERTScore: {agg['bertscore_f1']:.4f}")
    mape_str = f"{agg['mape']:.2f}%" if agg.get("mape") is not None else "N/A"
    print(f"  Num-Exact: {agg['num_exact']:.4f} | Num-F1: {agg['num_f1']:.4f} | MAPE: {mape_str}")
    print(f"  Tol-1: {agg['tol1_acc']:.4f} | Tol-5: {agg['tol5_acc']:.4f} | Tol-10: {agg['tol10_acc']:.4f}")
    print(f"  Entailment: {agg['entailment_ratio']:.4f} | Traceability: {agg['evidence_traceability']:.4f}")
    if "directional_accuracy" in agg:
        print(f"  Directional Acc: {agg['directional_accuracy']:.4f} (n={agg['directional_applicable']})")
    print(f"  Avg Latency: {agg['avg_latency_ms']:.1f}ms")
    if "by_question_type" in agg:
        print("  By Question Type:")
        for qt, d in agg["by_question_type"].items():
            print(f"    {qt}: EM={d['exact_match']:.4f} NumExact={d['num_exact']:.4f} F1={d['token_f1']:.4f}")
