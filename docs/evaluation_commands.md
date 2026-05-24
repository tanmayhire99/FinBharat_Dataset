# FinBharat — Evaluation Command Reference

> All commands run from the project root: `/Users/tanmayhire/Documents/Projects/EMNLP2026`
> Prefix every command with `uv run`

---

## 0. Setup

```bash
# Create venv and install all dependencies (run once)
uv venv --python 3.14
uv pip install -e ".[dev]"          # core + dev tools
uv pip install -e ".[ml]"           # torch, transformers, bert-score (NLI + BERTScore)

# Verify keys are in .env
cat .env | grep NVIDIA_API_KEY      # should show nvapi-... values
```

---

## 1. Explore the Dataset

```bash
# List available models
uv run python main.py models

# Sample easy questions (default 3 companies)
uv run python main.py sample --difficulty easy
uv run python main.py sample --difficulty medium
uv run python main.py sample --difficulty hard
uv run python main.py sample --difficulty multihop

# Sample from a specific dataset root
uv run python main.py sample --data-root dataset --difficulty hard
```

---

## 2. Quick Smoke Tests (development / debugging)

Run a very small batch to check the pipeline works end-to-end before scaling.

```bash
# 3 companies × 5 QA = 15 records — fastest sanity check
uv run python main.py evaluate \
  --difficulty easy \
  --max-per-company 5 \
  --models llama3.1-8b

# 3 companies × 10 QA = 30 records — standard dev run
uv run python main.py evaluate \
  --difficulty easy \
  --max-per-company 10 \
  --models llama3.1-8b

# Quick hard-tier check (3 companies, 5 QA each)
uv run python main.py evaluate \
  --difficulty hard \
  --max-per-company 5 \
  --models llama3.1-8b

# Quick multihop check
uv run python main.py evaluate \
  --difficulty multihop \
  --max-per-company 5 \
  --models llama3.1-8b
```

---

## 3. Regime Ablations (Pilot Scale)

Run all three regimes on the same 30 QA pilot to produce the comparison table.

```bash
# Open-book zero-shot (baseline)
uv run python main.py evaluate \
  --difficulty easy \
  --max-per-company 10 \
  --models llama3.1-8b \
  --regime zero_shot

# Closed-book (proves dataset is not memorized)
uv run python main.py evaluate \
  --difficulty easy \
  --max-per-company 10 \
  --models llama3.1-8b \
  --regime closed_book

# Few-shot open-book (3 generic examples in prompt)
uv run python main.py evaluate \
  --difficulty easy \
  --max-per-company 10 \
  --models llama3.1-8b \
  --regime few_shot

# Few-shot closed-book
uv run python main.py evaluate \
  --difficulty easy \
  --max-per-company 10 \
  --models llama3.1-8b \
  --regime few_shot_closed
```

---

## 4. Table Format Ablation

Tests how models handle different table serializations (all else equal).

```bash
# Default: raw HTML tables
uv run python main.py evaluate \
  --difficulty easy \
  --max-per-company 10 \
  --models llama3.1-8b \
  --table-format html

# Markdown pipe tables
uv run python main.py evaluate \
  --difficulty easy \
  --max-per-company 10 \
  --models llama3.1-8b \
  --table-format markdown

# Linearized key-value rows
uv run python main.py evaluate \
  --difficulty easy \
  --max-per-company 10 \
  --models llama3.1-8b \
  --table-format linearized
```

---

## 5. Full-Scale Single-Difficulty Runs (all 187 companies)

These are the primary results for the paper. Run one difficulty tier at a time.

```bash
# Easy — all 187 companies, all QA
uv run python main.py evaluate \
  --difficulty easy \
  --all-companies \
  --models llama3.1-8b

# Medium
uv run python main.py evaluate \
  --difficulty medium \
  --all-companies \
  --models llama3.1-8b

# Hard (forensic)
uv run python main.py evaluate \
  --difficulty hard \
  --all-companies \
  --models llama3.1-8b

# Multihop
uv run python main.py evaluate \
  --difficulty multihop \
  --all-companies \
  --models llama3.1-8b

# Capped run (faster iteration: max 20 QA/company)
uv run python main.py evaluate \
  --difficulty easy \
  --all-companies \
  --max-per-company 20 \
  --models llama3.1-8b
```

---

## 6. Multi-Model Runs (paper Table 2)

Run multiple models in a single command for comparison.

```bash
# Small model tier — for difficulty degradation curve
uv run python main.py evaluate \
  --difficulty easy \
  --all-companies --max-per-company 20 \
  --models llama3.1-8b \
  --models qwen3-8b \
  --models gemma3-4b

# Large model tier
uv run python main.py evaluate \
  --difficulty easy \
  --all-companies --max-per-company 20 \
  --models llama3.3-70b \
  --models qwen3-72b

# Full paper model panel (all 4 difficulties, runs sequentially)
for DIFF in easy medium hard multihop; do
  uv run python main.py evaluate \
    --difficulty $DIFF \
    --all-companies --max-per-company 20 \
    --models llama3.1-8b \
    --models qwen3-8b \
    --models llama3.3-70b \
    --models qwen3-72b
done
```

---

## 7. Full Ablation Matrix (paper Table 8)

Generates results for the complete ablation section. Run after pilots confirm the pipeline is stable.

```bash
# All 4 regimes × 2 models × easy difficulty
for REGIME in zero_shot few_shot closed_book few_shot_closed; do
  for MODEL in llama3.1-8b llama3.3-70b; do
    uv run python main.py evaluate \
      --difficulty easy \
      --all-companies --max-per-company 20 \
      --models $MODEL \
      --regime $REGIME
  done
done

# Table format ablation (easy, zero_shot, 2 models)
for FMT in html markdown linearized; do
  uv run python main.py evaluate \
    --difficulty easy \
    --all-companies --max-per-company 20 \
    --models llama3.1-8b \
    --models llama3.3-70b \
    --table-format $FMT
done
```

---

## 8. BRSR/ESG Subset Run

The BRSR filter is automatic — records from BRSR sections are flagged and the
`brsr_subset` metrics appear in every aggregate JSON automatically.
To run a BRSR-only evaluation explicitly:

```bash
# Easy + Medium together captures most BRSR QA
uv run python main.py evaluate \
  --difficulty easy \
  --all-companies --max-per-company 20 \
  --models llama3.1-8b

uv run python main.py evaluate \
  --difficulty medium \
  --all-companies --max-per-company 20 \
  --models llama3.1-8b

# Then inspect brsr_subset in the aggregate JSON:
cat results/aggregates/llama3.1-8b_easy_zero_shot.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('brsr_subset',{}), indent=2))"
```

---

## 9. Dataset Splits

```bash
# Generate 60/20/20 stratified splits (run once, already committed as splits.json)
uv run python -m finbharat.data.split \
  --data-root dataset \
  --output splits.json

# Show which companies are in each split
uv run python -m finbharat.data.split \
  --data-root dataset \
  --output splits.json \
  --show

# Run evaluation on test-set companies only (from splits.json)
python3 - << 'EOF'
import json, subprocess
splits = json.load(open("splits.json"))
test_companies = splits["test"]   # 38 companies
# Pass as repeated --models-style arg... use run_evaluation() directly from Python
EOF
```

---

## 10. Analyze Results

```bash
# Summary table of all completed runs
uv run python main.py analyze

# Full breakdown: question type + regime comparison + error summary + near-misses
uv run python main.py analyze --all

# Just regime comparison (open-book vs closed-book vs few-shot)
uv run python main.py analyze --compare

# Per question-type breakdown (Text Only / Table Only / Table+Text / Numerical Calc)
uv run python main.py analyze --detailed

# Error summary (API failures)
uv run python main.py analyze --errors

# Near-miss failures (EM=0 but F1 >= 0.7) — useful for error taxonomy
uv run python main.py analyze --failures

# Point to a different results directory
uv run python main.py analyze --results-dir results_prod

# Inspect a single aggregate JSON (includes CI, sector, BRSR, hop-count)
python3 -c "import json; d=json.load(open('results/aggregates/llama3.1-8b_easy_zero_shot.json')); print(json.dumps(d, indent=2))" | head -80
```

---

## 11. Statistical Significance (paired bootstrap)

Use from Python directly when you have two runs to compare:

```python
import json
from pathlib import Path
from finbharat.metrics.stats import compute_significance_table
from finbharat.eval.evaluate import EvalResult

def load_eval_results(path: str) -> list[EvalResult]:
    results = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            results.append(EvalResult(**{k: v for k, v in d.items()
                                          if k in EvalResult.__dataclass_fields__}))
    return results

# Compare Llama-3.3-70B vs Llama-3.1-8B on easy zero_shot
a = load_eval_results("results/eval_results/llama3.3-70b_easy_zero_shot.jsonl")
b = load_eval_results("results/eval_results/llama3.1-8b_easy_zero_shot.jsonl")
sig = compute_significance_table(a, b)
for metric, r in sig.items():
    star = "* " if r.significant else "  "
    print(f"{star}{metric:<20}: delta={r.delta:+.4f}  p={r.p_value:.4f}  A={r.ci_a}  B={r.ci_b}")
```

---

## 12. Run the Full Test Suite

```bash
# All tests
uv run pytest tests/ -v

# Quick (no output capture, stops on first failure)
uv run pytest tests/ -x -q

# With coverage report
uv run pytest tests/ --cov=src/finbharat --cov-report=term-missing
```

---

## Available Models

| Key | Model | Size | Provider |
|-----|-------|------|----------|
| `llama3.2-3b` | Llama-3.2-3B-Instruct | 3B | NIM |
| `gemma3-4b` | Gemma-3-4B-Instruct | 4B | NIM |
| `llama3.1-8b` | Llama-3.1-8B-Instruct | 8B | NIM |
| `qwen3-8b` | Qwen3-8B-Instruct | 8B | NIM |
| `qwen3-14b` | Qwen3-14B-Instruct | 14B | NIM |
| `mistral-small` | Mistral-Small-3.2-24B | 24B | NIM |
| `llama3.3-70b` | Llama-3.3-70B-Instruct | 70B | NIM |
| `qwen3-72b` | Qwen3-72B-Instruct | 72B | NIM |
| `deepseek-r1-70b` | DeepSeek-R1-Distill-70B | 70B | NIM |
| `qwen3-235b` | Qwen3-235B-A22B | 235B MoE | NIM |
| `gpt-4o` | GPT-4o | — | OpenAI |

---

## Result File Naming Convention

```
results/
  aggregates/        {model_key}_{difficulty}_{regime}[_{table_format}].json
  eval_results/      {model_key}_{difficulty}_{regime}[_{table_format}].jsonl
  generations/       {model_key}_{difficulty}_{regime}[_{table_format}].jsonl  ← cached API outputs

Examples:
  aggregates/llama3.1-8b_easy_zero_shot.json
  aggregates/llama3.3-70b_hard_few_shot.json
  aggregates/llama3.1-8b_medium_zero_shot_markdown.json   ← table format ablation
```

Each aggregate JSON contains:
- Top-level metrics (EM, F1, ROUGE-L, BERTScore, Num-Exact, Num-F1, MAPE, Tol-1/5/10, NLI, Traceability)
- `*_ci` keys for every metric: `{"mean": 0.67, "lower": 0.58, "upper": 0.76, "n": 1870}`
- `by_question_type`: Text Only / Table Only / Table with Text / Numerical Calculation
- `by_difficulty`: breakdown by tier (when running all-companies with multiple tiers)
- `by_sector`: per-sector metrics
- `by_company`: per-company metrics
- `brsr_subset`: BRSR/ESG-specific metrics (if any BRSR records present)
- `red_flag_detection`: hard QA contradiction detection stats
- `by_hop_count`: `2-hop`, `3-hop`, `4-hop`, `5+-hop` breakdown
- `regime`: which evaluation mode was used
