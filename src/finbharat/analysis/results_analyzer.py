"""
Analyze and summarize evaluation results.

Usage:
    uv run python -m finbharat.analysis.results_analyzer --results-dir results
    uv run python -m finbharat.analysis.results_analyzer --results-dir results --detailed
    uv run python -m finbharat.analysis.results_analyzer --results-dir results --compare
    uv run python -m finbharat.analysis.results_analyzer --results-dir results --errors
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


METRIC_COLS = [
    "exact_match", "relaxed_em", "token_f1", "rouge_l", "bertscore_f1",
    "num_exact", "num_f1", "mape", "tol1_acc", "tol5_acc", "tol10_acc",
    "entailment_ratio", "directional_accuracy",
]

DISPLAY_NAMES = {
    "exact_match":          "EM",
    "relaxed_em":           "Rel.EM",
    "token_f1":             "F1",
    "rouge_l":              "ROUGE-L",
    "bertscore_f1":         "BERTScore",
    "num_exact":            "Num-EM",
    "num_f1":               "Num-F1",
    "mape":                 "MAPE%",
    "tol1_acc":             "Tol-1",
    "tol5_acc":             "Tol-5",
    "tol10_acc":            "Tol-10",
    "entailment_ratio":     "NLI-Ent",
    "directional_accuracy": "Dir-Acc",
}


def load_aggregates(results_dir: Path) -> list[dict]:
    agg_dir = results_dir / "aggregates"
    if not agg_dir.exists():
        return []
    records = []
    for f in sorted(agg_dir.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Filename: {model_key}_{difficulty}_{regime}.json
            # Difficulty is always one of easy|medium|hard|multihop — use as anchor
            stem = f.stem
            model_key, difficulty, regime = stem, "?", "?"
            for diff in ("easy", "medium", "hard", "multihop"):
                marker = f"_{diff}_"
                if marker in stem:
                    idx = stem.index(marker)
                    model_key = stem[:idx]
                    regime = stem[idx + len(marker):]
                    difficulty = diff
                    break
            data["_model_key"] = model_key
            data["_difficulty"] = difficulty
            data["_regime"] = regime
            data["_file"] = f.name
            records.append(data)
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")
    return records


def fmt(val, key: str) -> str:
    if val is None:
        return "  -   "
    if key == "mape":
        return f"{float(val):6.2f}%"
    return f"{float(val):6.4f}"


def print_summary_table(records: list[dict]):
    if not records:
        print("No results found.")
        return

    core_metrics = [
        "exact_match", "relaxed_em", "token_f1", "rouge_l", "bertscore_f1",
        "num_exact", "num_f1", "mape", "entailment_ratio",
    ]

    col_w = 10
    header_parts = [f"{'Model':<22}", f"{'Diff':<10}", f"{'Regime':<14}", f"{'N':>5}"]
    for m in core_metrics:
        header_parts.append(f"{DISPLAY_NAMES[m]:>{col_w}}")
    sep = "  "
    print(sep.join(header_parts))
    print("-" * (22 + 10 + 14 + 5 + len(core_metrics) * (col_w + 2) + 10))

    for r in records:
        n = r.get("total", 0)
        row = [
            f"{r['_model_key']:<22}",
            f"{r['_difficulty']:<10}",
            f"{r['_regime']:<14}",
            f"{n:>5}",
        ]
        for m in core_metrics:
            val = r.get(m)
            row.append(f"{fmt(val, m):>{col_w}}")
        print(sep.join(row))


def print_by_difficulty(records: list[dict]):
    """For each model+regime combo show difficulty breakdown."""
    groups: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        key = (r["_model_key"], r["_regime"])
        groups[key][r["_difficulty"]] = r

    difficulties = ["easy", "medium", "hard", "multihop"]
    metrics = ["exact_match", "token_f1", "rouge_l", "num_exact", "num_f1", "entailment_ratio"]

    for (model, regime), diffs in sorted(groups.items()):
        present_diffs = [d for d in difficulties if d in diffs]
        if not present_diffs:
            continue
        print(f"\n  {model} | {regime}")
        header = f"  {'Metric':<22}" + "".join(f"{d:>12}" for d in present_diffs)
        print(header)
        print("  " + "-" * (22 + 12 * len(present_diffs)))
        for m in metrics:
            vals = [diffs.get(d, {}).get(m) for d in present_diffs]
            row = f"  {DISPLAY_NAMES.get(m, m):<22}" + "".join(f"{fmt(v, m):>12}" for v in vals)
            print(row)


def print_by_question_type(records: list[dict]):
    for r in records:
        by_qt = r.get("by_question_type", {})
        if not by_qt:
            continue
        print(f"\n  {r['_model_key']} | {r['_difficulty']} | {r['_regime']}")
        print(f"  {'Type':<35} {'N':>5} {'EM':>8} {'F1':>8} {'NumEM':>8} {'ROUGE-L':>8}")
        print("  " + "-" * 75)
        for qt, d in sorted(by_qt.items(), key=lambda x: -x[1]["total"]):
            print(
                f"  {qt:<35} {d['total']:>5} "
                f"{d['exact_match']:>8.4f} {d['token_f1']:>8.4f} "
                f"{d['num_exact']:>8.4f} {d.get('rouge_l', 0.0):>8.4f}"
            )


def print_by_difficulty_breakdown(records: list[dict]):
    for r in records:
        by_diff = r.get("by_difficulty", {})
        if not by_diff:
            continue
        print(f"\n  {r['_model_key']} | {r['_difficulty']} | {r['_regime']}")
        print(f"  {'Difficulty':<15} {'N':>5} {'EM':>8} {'F1':>8} {'NumEM':>8}")
        print("  " + "-" * 50)
        for diff in ["easy", "medium", "hard", "multihop"]:
            d = by_diff.get(diff)
            if not d:
                continue
            print(f"  {diff:<15} {d['total']:>5} {d['exact_match']:>8.4f} {d['token_f1']:>8.4f} {d['num_exact']:>8.4f}")


def print_regime_comparison(records: list[dict]):
    """Compare regimes for same model+difficulty."""
    groups: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        key = (r["_model_key"], r["_difficulty"])
        groups[key][r["_regime"]] = r

    regimes_order = ["zero_shot", "few_shot", "closed_book", "few_shot_closed"]
    metrics = ["exact_match", "relaxed_em", "token_f1", "rouge_l", "num_exact", "entailment_ratio"]

    for (model, diff), regime_map in sorted(groups.items()):
        if len(regime_map) < 2:
            continue
        present = [rg for rg in regimes_order if rg in regime_map]
        print(f"\n  {model} | {diff}")
        header = f"  {'Metric':<22}" + "".join(f"{rg:>18}" for rg in present)
        print(header)
        print("  " + "-" * (22 + 18 * len(present)))
        for m in metrics:
            vals = [regime_map.get(rg, {}).get(m) for rg in present]
            row = f"  {DISPLAY_NAMES.get(m, m):<22}" + "".join(f"{fmt(v, m):>18}" for v in vals)
            print(row)

        # Delta: closed_book vs zero_shot
        if "zero_shot" in regime_map and "closed_book" in regime_map:
            print(f"\n  Delta (closed_book - zero_shot):")
            for m in ["exact_match", "token_f1", "num_exact"]:
                zs = regime_map["zero_shot"].get(m, 0) or 0
                cb = regime_map["closed_book"].get(m, 0) or 0
                delta = cb - zs
                sign = "+" if delta >= 0 else ""
                print(f"    {DISPLAY_NAMES.get(m, m):<20}: {sign}{delta:+.4f}")


def print_error_summary(results_dir: Path):
    eval_dir = results_dir / "eval_results"
    if not eval_dir.exists():
        print("  No eval_results directory found.")
        return
    total_errors = 0
    for f in sorted(eval_dir.glob("*.jsonl")):
        errors = []
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    if d.get("error"):
                        errors.append(d)
                except Exception:
                    pass
        if errors:
            print(f"  {f.stem}: {len(errors)} errors")
            for e in errors[:3]:
                print(f"    qid={e.get('question_id','?')} | {str(e.get('error',''))[:90]}")
            total_errors += len(errors)
    if total_errors == 0:
        print("  No API errors.")
    else:
        print(f"  Total errors across all runs: {total_errors}")


def print_top_failures(results_dir: Path, n: int = 10):
    """Show examples where EM=0 but token_f1 is high (near-misses)."""
    eval_dir = results_dir / "eval_results"
    if not eval_dir.exists():
        return
    near_misses = []
    for f in sorted(eval_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    if d.get("exact_match") == 0 and d.get("token_f1", 0) >= 0.7 and not d.get("error"):
                        d["_file"] = f.stem
                        near_misses.append(d)
                except Exception:
                    pass
    near_misses.sort(key=lambda x: -x.get("token_f1", 0))
    print(f"\n  Top {min(n, len(near_misses))} near-misses (EM=0, F1>=0.7):")
    print(f"  {'Model/Diff':<30} {'F1':>6} {'Gold Answer':<40} {'Predicted':<40}")
    print("  " + "-" * 120)
    for d in near_misses[:n]:
        tag = d["_file"][:28]
        gold = str(d.get("gold_answer", ""))[:38]
        pred = str(d.get("predicted_answer", ""))[:38]
        f1 = d.get("token_f1", 0)
        print(f"  {tag:<30} {f1:>6.4f} {gold:<40} {pred:<40}")


def main():
    parser = argparse.ArgumentParser(description="Analyze FinBharat evaluation results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--detailed", action="store_true", help="Show by-question-type breakdown")
    parser.add_argument("--compare", action="store_true", help="Regime comparison table")
    parser.add_argument("--errors", action="store_true", help="Show API error summary")
    parser.add_argument("--failures", action="store_true", help="Show near-miss failures")
    parser.add_argument("--all", action="store_true", help="Show everything")
    args = parser.parse_args()

    records = load_aggregates(args.results_dir)
    if not records:
        print(f"No aggregate files found in {args.results_dir / 'aggregates'}")
        return

    print(f"\n{'='*100}")
    print(f"  FinBharat Results  |  {len(records)} run(s)  |  {args.results_dir}")
    print(f"{'='*100}\n")
    print_summary_table(records)

    if len({r["_difficulty"] for r in records}) > 1 or args.all:
        print(f"\n{'='*100}")
        print("  By Difficulty Tier")
        print(f"{'='*100}")
        print_by_difficulty(records)

    if args.detailed or args.all:
        print(f"\n{'='*100}")
        print("  By Question Type")
        print(f"{'='*100}")
        print_by_question_type(records)

    if args.compare or args.all:
        print(f"\n{'='*100}")
        print("  Regime Comparison  (open-book vs closed-book vs few-shot)")
        print(f"{'='*100}")
        print_regime_comparison(records)

    if args.errors or args.all:
        print(f"\n{'='*100}")
        print("  API Error Summary")
        print(f"{'='*100}")
        print_error_summary(args.results_dir)

    if args.failures or args.all:
        print(f"\n{'='*100}")
        print("  Near-Miss Failure Analysis")
        print(f"{'='*100}")
        print_top_failures(args.results_dir)


if __name__ == "__main__":
    main()
