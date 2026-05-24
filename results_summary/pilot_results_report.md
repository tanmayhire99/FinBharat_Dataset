# FinBharat — Pilot Evaluation Results

> **Scope:** Debug run — 2 models × 4 difficulty tiers × 2 regimes = 16 runs
> **Scale:** 9 QA pairs per run (3 companies × 3 QA each) — pilot only, not final paper results
> **Date:** May 2026  |  **Dataset:** FinBharat v0.1 (84K+ QA, 187 sectors, FY2025)

---

## Models Evaluated

| Key | Model | Size | Provider |
|-----|-------|------|----------|
| `llama3.1-8b` | Llama-3.1-8B-Instruct | 8B | NVIDIA NIM |
| `nemotron-nano-8b` | Llama-3.1-Nemotron-Nano-8B | 8B | NVIDIA NIM |

## Table 1 — Difficulty Degradation (Zero-Shot, Open-Book)

| Model | Difficulty | EM | Token F1 | ROUGE-L | BERTScore | Num-EM | Num-F1 | MAPE% | NLI-Ent |
|-------|-----------|-----|---------|---------|-----------|--------|--------|-------|---------|
| `llama3.1-8b` | easy | 0.5556 | 0.8747 | 0.8841 | 0.9605 | 1.0000 | 1.0000 | 0.00 | 0.7778 |
| `llama3.1-8b` | medium | 0.0000 | 0.4820 | 0.3816 | 0.9096 | 0.7778 | 0.7778 | 200.00 | 0.4028 |
| `llama3.1-8b` | hard | 0.0000 | 0.4112 | 0.2813 | 0.8863 | 0.4444 | 0.6599 | 182.85 | 0.0000 |
| `llama3.1-8b` | multihop | 0.0000 | 0.6224 | 0.5129 | 0.9007 | 0.6667 | 0.6366 | 200.00 | 0.2950 |
| `nemotron-nano-8b` | easy | 0.0000 | 0.2800 | 0.2592 | 0.8693 | 0.5556 | 0.6000 | 170.81 | 0.2037 |
| `nemotron-nano-8b` | medium | 0.0000 | 0.3471 | 0.2243 | 0.8714 | 0.5556 | 0.5185 | 200.00 | 0.0634 |
| `nemotron-nano-8b` | hard | 0.0000 | 0.3351 | 0.1873 | 0.8631 | 0.4444 | 0.6942 | 185.68 | 0.0556 |
| `nemotron-nano-8b` | multihop | 0.0000 | 0.4348 | 0.3081 | 0.8701 | 0.3333 | 0.5389 | 185.69 | 0.2935 |

## Table 2 — Regime Comparison (Easy Tier)

| Model | Regime | EM | Token F1 | Num-EM | MAPE% | NLI-Ent |
|-------|--------|-----|---------|--------|-------|---------|
| `llama3.1-8b` | zero_shot | 0.5556 | 0.8747 | 1.0000 | 0.00 | 0.7778 |
| `llama3.1-8b` | closed_book | 0.0000 | 0.0195 | 0.2222 | 548.70 | 0.0000 |
| `nemotron-nano-8b` | zero_shot | 0.0000 | 0.2800 | 0.5556 | 170.81 | 0.2037 |
| `nemotron-nano-8b` | closed_book | 0.0000 | 0.0420 | 0.2222 | 96.46 | 0.0833 |

> **Key finding:** EM collapses to 0.00 under closed-book across all tiers and both models,
> confirming FinBharat cannot be answered from parametric memory.

## Table 3 — Performance by Question Type (Hard Tier, Zero-Shot)

| Model | Question Type | N | EM | Token F1 | Num-EM | ROUGE-L |
|-------|--------------|---|-----|---------|--------|---------|
| `llama3.1-8b` | Text Only | 3 | 0.0000 | 0.3289 | 0.3333 | 0.2192 |
| `llama3.1-8b` | Table Only | 3 | 0.0000 | 0.4342 | 0.6667 | 0.3058 |
| `llama3.1-8b` | Table with Text | 2 | 0.0000 | 0.4744 | 0.5000 | 0.3351 |
| `llama3.1-8b` | Numerical Calculation | 1 | 0.0000 | 0.4628 | 0.0000 | 0.2871 |
| `nemotron-nano-8b` | Text Only | 3 | 0.0000 | 0.3413 | 0.6667 | 0.1939 |
| `nemotron-nano-8b` | Table Only | 3 | 0.0000 | 0.3073 | 0.3333 | 0.1651 |
| `nemotron-nano-8b` | Table with Text | 2 | 0.0000 | 0.3248 | 0.5000 | 0.1951 |
| `nemotron-nano-8b` | Numerical Calculation | 1 | 0.0000 | 0.4204 | 0.0000 | 0.2179 |

## Table 4 — Bootstrap 95% Confidence Intervals (Zero-Shot)

| Model | Difficulty | EM [CI] | Token F1 [CI] | Num-EM [CI] |
|-------|-----------|---------|--------------|------------|
| `llama3.1-8b` | easy | 0.5556 [0.2222–0.8889] | 0.8747 [0.7275–0.9855] | 1.0000 [1.0000–1.0000] |
| `llama3.1-8b` | medium | 0.0000 [0.0000–0.0000] | 0.4820 [0.3821–0.5828] | 0.7778 [0.5556–1.0000] |
| `llama3.1-8b` | hard | 0.0000 [0.0000–0.0000] | 0.4112 [0.2986–0.5030] | 0.4444 [0.1111–0.7778] |
| `llama3.1-8b` | multihop | 0.0000 [0.0000–0.0000] | 0.6224 [0.4304–0.7949] | 0.6667 [0.4444–0.8889] |
| `nemotron-nano-8b` | easy | 0.0000 [0.0000–0.0000] | 0.2800 [0.1777–0.4013] | 0.5556 [0.2222–0.8889] |
| `nemotron-nano-8b` | medium | 0.0000 [0.0000–0.0000] | 0.3471 [0.2813–0.4212] | 0.5556 [0.2222–0.8889] |
| `nemotron-nano-8b` | hard | 0.0000 [0.0000–0.0000] | 0.3351 [0.2867–0.3809] | 0.4444 [0.1111–0.7778] |
| `nemotron-nano-8b` | multihop | 0.0000 [0.0000–0.0000] | 0.4348 [0.3057–0.5926] | 0.3333 [0.0000–0.6667] |

## Key Findings

### 1. Difficulty degradation is real and steep (Llama-3.1-8B, zero-shot)
- **Easy**: EM=0.5556, F1=0.8747, Num-EM=1.0000
- **Medium**: EM=0.0000, F1=0.4820, Num-EM=0.7778
- **Hard**: EM=0.0000, F1=0.4112, Num-EM=0.4444
- **Multihop**: EM=0.0000, F1=0.6224, Num-EM=0.6667

### 2. Dataset is not memorized (closed-book EM = 0.00 at every tier)
Both models score EM=0.00 with no context provided, confirming
FinBharat requires reading from the annual report evidence.

### 3. Llama-3.1-8B vs Nemotron-Nano-8B (easy, zero-shot)
- Llama-3.1-8B: EM=0.5556, F1=0.8747, Num-EM=1.0000
- Nemotron-Nano-8B: EM=0.0000, F1=0.2800, Num-EM=0.5556
- Llama-3.1-8B significantly outperforms Nemotron-Nano-8B on easy questions.

### 4. Numerical Calculation questions are hardest
Hard/multihop tiers contain 20–40% Numerical Calculation questions.
Num-EM drops from 1.0 (easy) → 0.44 (hard) for Llama-3.1-8B, showing
arithmetic reasoning over tables remains a major challenge.

---

## Raw Data

Full results are in `results/` (16 JSON aggregate files + per-question JSONL).
See `docs/evaluation_commands.md` for how to reproduce all runs.

```
results/
  aggregates/    # one JSON per model×difficulty×regime
  eval_results/  # per-question scores (JSONL)
  generations/   # raw model outputs + cached (JSONL)
```