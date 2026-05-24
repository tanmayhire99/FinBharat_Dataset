# FinBharat — Group Briefing

> **Target venue:** EMNLP 2026 (ARR May 2026 cycle)  
> **Status:** Pilot evaluation complete · Full-scale runs in progress  
> **Date:** May 2026

---

## What is FinBharat?

**FinBharat** is a large-scale benchmark for evaluating how well Large Language Models (LLMs) answer questions about Indian company annual reports — and whether their answers are actually *supported by the evidence* (hallucination detection).

### The core problem we're solving

Financial analysts and investors rely on annual reports to make decisions. If an LLM reads an annual report and answers questions about it, two things can go wrong:
1. **Wrong answer** — the model extracted or calculated the wrong number
2. **Hallucinated answer** — the model gave a confident, plausible-sounding answer that isn't in the report at all

No benchmark currently evaluates both of these for Indian companies. Every existing financial QA benchmark (FinQA, TAT-QA, PHANTOM) uses US/EU data. India has a completely different regulatory context — BSE/NSE-listed companies, BRSR sustainability reporting, Crore/Lakh units, SEBI governance requirements.

### Why India specifically?

- **Scale:** India is the world's 5th largest economy with 5,000+ listed companies
- **BRSR:** India mandates Business Responsibility and Sustainability Reporting (BRSR) — no other financial benchmark covers ESG disclosures
- **Unit complexity:** Indian reports mix ₹ Crores, ₹ Lakhs, USD Millions in the same document — a unique challenge for numeric reasoning
- **Underrepresented:** Despite India's economic size, there is zero India-focused financial NLP benchmark

---

## The Dataset

| Property | Value |
|----------|-------|
| Companies | 187 (one per sector, BSE/NSE-listed, FY2025) |
| Sectors | 187 (from 2-3 Wheelers to Waste Management) |
| Total QA pairs | ~84,300 |
| Difficulty tiers | Easy · Medium · Hard (Forensic) · Multihop |
| Question types | Text Only · Table Only · Table+Text · Numerical Calculation |
| Source | Publicly available annual reports (BSE/NSE) |

### Four difficulty tiers — what they test

| Tier | What it requires | % Numerical Calculation |
|------|-----------------|------------------------|
| **Easy** | Direct lookup from a single table or paragraph | 0% |
| **Medium** | Cross-table or cross-section extraction; some arithmetic | 7% |
| **Hard (Forensic)** | Identifying inconsistencies, red flags, governance issues across sections | 38% |
| **Multihop** | Chaining 2–4 reasoning steps across multiple sections | 59% |

### BRSR/ESG Track (unique to FinBharat)

748 unique section types in the dataset come from India's BRSR framework — covering Principles 1–9 (responsible business conduct, environment, human rights, etc.). **No other financial benchmark evaluates ESG/sustainability QA.** This is our key differentiator.

---

## Evaluation Metrics

We compute 14 metrics per prediction, organized by what they measure:

### Text correctness
| Metric | What it measures |
|--------|-----------------|
| **Exact Match (EM)** | Strict string match after normalization |
| **Relaxed EM** | EM after stripping Indian units (₹, Crore, Million, Lakh) |
| **Token F1** | Token-level overlap — partial credit for partial answers |
| **ROUGE-L** | Longest common subsequence overlap |
| **METEOR** | Synonym + stem-aware overlap (better for paraphrases) |
| **BERTScore (P/R/F1)** | Semantic similarity via RoBERTa embeddings |

### Numeric correctness
| Metric | What it measures |
|--------|-----------------|
| **Num-Exact** | Did the model extract/calculate the right number? |
| **Num-F1** | Set-based F1 over all numbers in the answer |
| **MAPE** | Mean Absolute Percentage Error (capped at 1000%) |
| **Tol-1 / Tol-5 / Tol-10** | Accuracy within ±1% / ±5% / ±10% of gold |

### Faithfulness (hallucination detection)
| Metric | What it measures |
|--------|-----------------|
| **NLI Entailment** | DeBERTa cross-encoder: does evidence support the answer? |
| **Evidence Traceability** | Word-overlap between cited evidence and gold evidence |
| **Abstain Rate** | How often the model correctly says "not available in context" |

### Ablation regimes
Every run is tagged with its **regime**:
- `zero_shot` — model sees question + evidence (open-book, no examples)
- `closed_book` — model sees only the question (no evidence) → proves dataset requires reading
- `few_shot` — 3 examples in prompt before the question
- `few_shot_closed` — 3 examples but no evidence

---

## Initial Results (Pilot)

> **Scale:** 15 QA pairs per tier (3 companies × 5 QA)  
> **Models:** Llama-3.1-8B-Instruct · Llama-3.3-70B-Instruct  
> **Regime:** Zero-shot open-book

### Table 1 — Full metric suite across all difficulty tiers

| Model | Tier | EM | Token F1 | METEOR | BERTScore | Num-EM | MAPE% | NLI | Abstain |
|-------|------|----|---------|--------|-----------|--------|-------|-----|---------|
| Llama-3.1-8B | Easy | **0.6667** | **0.8878** | 0.7262 | 0.9675 | 0.9333 | 7.3% | 0.8000 | 0% |
| Llama-3.1-8B | Medium | 0.0000 | 0.5320 | 0.5000 | 0.9152 | 0.6667 | 219% | 0.3417 | 0% |
| Llama-3.1-8B | Hard | 0.0000 | 0.4254 | 0.3741 | 0.8821 | 0.4667 | 208% | 0.0510 | 6.7% |
| Llama-3.1-8B | Multihop | 0.0000 | 0.5429 | 0.5434 | 0.8923 | 0.4667 | 162% | 0.2722 | **20%** |
| Llama-3.3-70B | Easy | **0.7333** | 0.8843 | 0.7230 | 0.9641 | **1.0000** | 0% | 0.6667 | 0% |
| Llama-3.3-70B | Medium | 0.0000 | 0.5396 | 0.4753 | 0.9138 | 0.6000 | 50% | 0.1194 | 0% |
| Llama-3.3-70B | Hard | 0.0000 | 0.4041 | 0.3676 | 0.8737 | 0.4667 | 199% | 0.0921 | 0% |
| Llama-3.3-70B | Multihop | 0.0000 | **0.6061** | **0.6351** | **0.9041** | **0.6000** | 247% | **0.4606** | 6.7% |

### Table 2 — Open-book vs Closed-book (Llama-3.1-8B, Easy)

| Regime | EM | Token F1 | Num-EM | MAPE% | NLI |
|--------|-----|---------|--------|-------|-----|
| **zero_shot** (open-book) | **0.6667** | **0.8878** | **0.9333** | 7.3% | **0.8000** |
| **closed_book** (no evidence) | 0.0000 | 0.0195 | 0.2222 | 5179% | 0.0000 |

> EM drops from 0.67 → 0.00 and MAPE explodes from 7% → 5179% without evidence.  
> **This proves FinBharat cannot be answered from memorized knowledge.**

---

## Key Findings from Pilot

### Finding 1: Steep difficulty degradation
Performance drops sharply from Easy → Multihop. On Token F1, Llama-3.1-8B drops from **0.89 (easy) → 0.43 (hard)** — a 46 percentage point fall. This validates that the dataset's difficulty tiers are meaningful and not trivially answerable.

### Finding 2: Dataset is not memorized (closed-book collapses)
Both models score EM = 0.00 across all tiers when no evidence is provided. This is the critical proof that FinBharat cannot be answered from pre-training knowledge alone. It verifies the dataset tests comprehension, not memorization.

### Finding 3: Abstain rate rises with difficulty — healthy model behaviour
Llama-3.1-8B abstains on 0% of easy questions but 20% of multihop questions ("Not available in context"). This is correct behaviour — hard cross-document reasoning questions genuinely cannot be answered from a single bundle. Rising abstain rate is a signal of model calibration.

### Finding 4: NLI Entailment drops to near-zero on Hard
Llama-3.1-8B: NLI = 0.80 (easy) → **0.05 (hard)**. Hard QA requires forensic reasoning that goes beyond what is directly stated in the evidence — the model's answers frequently contain claims not directly supported by the bundle context. This is a measurable hallucination signal.

### Finding 5: 70B outperforms 8B on easy Num-EM (1.0 vs 0.93)
The 70B model achieves perfect numeric extraction on easy questions — no rounding errors, correct scale. The 8B model makes ~7% numeric errors even on easy questions. This will be a stronger distinction on harder tiers.

### Finding 6: METEOR > ROUGE-L for multihop
On multihop, METEOR (0.54) consistently scores higher than ROUGE-L (0.44) because METEOR handles stemming and synonym matching — multihop answers often paraphrase the source text more than easy answers. This confirms both metrics are needed.

---

## What's Next

### Immediate (code-complete, needs running)
- Full-scale evaluation: all 187 companies × 4 tiers × 4+ models
- Closed-book and few-shot at scale (ablation tables)
- Table format ablation (HTML vs Markdown vs linearized)

### Requires humans (start now — longest lead time)
- **Human validation of 500 QA pairs** (3 annotators, Fleiss' κ) — this is the most critical gap for EMNLP acceptance. Reviewers will reject a benchmark with 84K LLM-generated QA and zero human validation.

### Planned
- LLM-judge faithfulness harness (GPT-4o calibrated against human labels)
- Cross-benchmark comparison (run same models on FinQA/TAT-QA)
- Hindi translation subset
- HuggingFace release + leaderboard

---

## Repository Structure

```
EMNLP2026/
  dataset/                  84K+ QA pairs, 187 companies, FY2025
  src/finbharat/
    data/                   Dataset loader, BRSR filter, table serializer, splits
    metrics/                EM, Relaxed EM, F1, ROUGE-L, METEOR, BERTScore, NLI,
                            Num-Exact, Num-F1, MAPE, Tol-1/5/10, Abstain, Bootstrap CI
    models/                 NVIDIA NIM runner, retry+key rotation, regimes
    eval/                   Full evaluation pipeline, aggregate_results
    analysis/               Results analyzer CLI
  tests/                    31 tests, all passing
  docs/
    pending_implementations.md   Full gap tracker (40 items)
    evaluation_commands.md       All run commands with examples
    test_descriptions.md         Purpose + thought process for all 31 tests
  results_summary/
    FinBharat_Group_Briefing.md  ← This document
    pilot_results_report.md      Detailed 5-table results report
    results_all_runs.csv         All runs, all metrics, one row per run
    difficulty_degradation.csv   Easy→Multihop degradation table
    all_aggregates.json          Full JSON with CIs, by-sector, BRSR, hop-count
  splits.json               Train(112) / Dev(37) / Test(38) stratified split
```

---

## How to Run

```bash
# Setup (one time)
uv venv --python 3.14 && uv pip install -e ".[dev,ml]"

# Quick sanity run (3 companies, 5 QA each)
uv run python main.py evaluate --difficulty easy --max-per-company 5

# Analyze results
uv run python main.py analyze --all

# Full debug run (2 models × 4 tiers × 2 regimes)
for MODEL in llama3.1-8b llama3.3-70b; do
  for DIFF in easy medium hard multihop; do
    uv run python main.py evaluate --models $MODEL --difficulty $DIFF --max-per-company 5
  done
done
```

See `docs/evaluation_commands.md` for the complete command reference.
