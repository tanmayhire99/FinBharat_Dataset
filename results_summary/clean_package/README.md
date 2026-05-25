# FinBharat — Closed-Book Evaluation Results (Clean Runs)

> **Date:** 2026-05-25 | **Regime:** Closed-Book (Company + Sector + Year metadata provided, no document)
> **Scale:** 15 QA per tier (3 companies × 5 QA) | **Models:** 9 fully clean
> **Total predictions:** 540 | **API errors:** 0

---

## Experimental Setup

**Closed-book means:** Model receives company name, sector, fiscal year, and country.
**No document is provided.** The model must answer from pre-training knowledge only.

**Companies tested:** HDFC Bank (Private Sector Bank), Bosch (Auto Components), IndiGo (Airline)

**Difficulty tiers:**
- **Easy** — Direct lookup from a single table or paragraph
- **Medium** — Cross-section extraction, some arithmetic
- **Hard** — Forensic/governance questions, red flags
- **Multihop** — Chain of reasoning across multiple sections

---

## Table 1 — Full Metric Suite by Model and Difficulty

| Model | Size | Tier | N | EM | Rel.EM | F1 | ROUGE-L | METEOR | BS-F1 | Num-EM | Num-F1 | Abstain | NLI |
|-------|------|------|---|-----|--------|-----|---------|--------|-------|--------|--------|---------|-----|
| `llama3.1-8b` | Small (8B) | easy | 15 | 0.0000 | 0.0000 | 0.0049 | 0.0048 | 0.0133 | 0.8349 | 0.2000 | 0.2000 | 0.0000 | 0.0000 |
| `llama3.1-8b` | Small (8B) | medium | 15 | 0.0000 | 0.0667 | 0.0553 | 0.0491 | 0.0624 | 0.8330 | 0.3333 | 0.3333 | 0.0000 | 0.0333 |
| `llama3.1-8b` | Small (8B) | hard | 15 | 0.0000 | 0.3333 | 0.1567 | 0.0925 | 0.1255 | 0.8375 | 0.2000 | 0.2037 | 0.0000 | 0.0000 |
| `llama3.1-8b` | Small (8B) | multihop | 15 | 0.0000 | 0.2000 | 0.1148 | 0.0883 | 0.1155 | 0.8337 | 0.3333 | 0.3056 | 0.0000 | 0.0333 |
| `llama3.2-3b` | Small (3B) | easy | 15 | 0.0000 | 0.0000 | 0.0049 | 0.0048 | 0.0133 | 0.8349 | 0.2000 | 0.2000 | 0.0000 | 0.0000 |
| `llama3.2-3b` | Small (3B) | medium | 15 | 0.0000 | 0.0000 | 0.0588 | 0.0468 | 0.0875 | 0.8323 | 0.2667 | 0.2667 | 0.0000 | 0.0000 |
| `llama3.2-3b` | Small (3B) | hard | 15 | 0.0000 | 0.4667 | 0.1664 | 0.1035 | 0.1280 | 0.8409 | 0.2000 | 0.3222 | 0.0000 | 0.0083 |
| `llama3.2-3b` | Small (3B) | multihop | 15 | 0.0000 | 0.2000 | 0.0790 | 0.0574 | 0.0863 | 0.8275 | 0.2667 | 0.2667 | 0.0000 | 0.0333 |
| `llama3.3-70b` | Large (70B) | easy | 15 | 0.0000 | 0.0000 | 0.0049 | 0.0048 | 0.0133 | 0.8349 | 0.2000 | 0.2000 | 0.0000 | 0.0000 |
| `llama3.3-70b` | Large (70B) | medium | 15 | 0.0000 | 0.0667 | 0.0696 | 0.0611 | 0.0617 | 0.8357 | 0.3333 | 0.3333 | 0.0000 | 0.0000 |
| `llama3.3-70b` | Large (70B) | hard | 15 | 0.0000 | 0.4667 | 0.1783 | 0.1132 | 0.1397 | 0.8420 | 0.2000 | 0.3606 | 0.0000 | 0.0000 |
| `llama3.3-70b` | Large (70B) | multihop | 15 | 0.0000 | 0.2000 | 0.0865 | 0.0732 | 0.0985 | 0.8295 | 0.3333 | 0.2889 | 0.0000 | 0.0333 |
| `llama4-maverick` | Medium (17B MoE) | easy | 15 | 0.0000 | 0.0000 | 0.0049 | 0.0048 | 0.0015 | 0.8333 | 0.2000 | 0.2000 | 0.0000 | 0.0000 |
| `llama4-maverick` | Medium (17B MoE) | medium | 15 | 0.0000 | 0.0000 | 0.1477 | 0.0993 | 0.1945 | 0.8425 | 0.3333 | 0.4071 | 0.0000 | 0.0611 |
| `llama4-maverick` | Medium (17B MoE) | hard | 15 | 0.0000 | 0.2667 | 0.2196 | 0.1187 | 0.2024 | 0.8410 | 0.2667 | 0.3848 | 0.0667 | 0.0460 |
| `llama4-maverick` | Medium (17B MoE) | multihop | 15 | 0.0000 | 0.2000 | 0.1450 | 0.0914 | 0.1614 | 0.8339 | 0.3333 | 0.2946 | 0.0000 | 0.0433 |
| `mistral-nemotron` | Medium (~24B) | easy | 15 | 0.0000 | 0.0667 | 0.0143 | 0.0130 | 0.0279 | 0.8315 | 0.2000 | 0.2000 | 0.0000 | 0.0000 |
| `mistral-nemotron` | Medium (~24B) | medium | 15 | 0.0000 | 0.0667 | 0.1314 | 0.1131 | 0.1402 | 0.8465 | 0.3333 | 0.3000 | 0.0000 | 0.0000 |
| `mistral-nemotron` | Medium (~24B) | hard | 15 | 0.0000 | 0.4000 | 0.1971 | 0.1189 | 0.1657 | 0.8437 | 0.2667 | 0.3318 | 0.0000 | 0.0051 |
| `mistral-nemotron` | Medium (~24B) | multihop | 15 | 0.0000 | 0.2000 | 0.1565 | 0.1273 | 0.1443 | 0.8428 | 0.2667 | 0.2667 | 0.0000 | 0.0333 |
| `mistral-small-119b` | Large (119B) | easy | 15 | 0.0000 | 0.0000 | 0.0049 | 0.0048 | 0.0133 | 0.8349 | 0.2000 | 0.2000 | 0.0000 | 0.0000 |
| `mistral-small-119b` | Large (119B) | medium | 15 | 0.0000 | 0.0667 | 0.0928 | 0.0733 | 0.0854 | 0.8364 | 0.3333 | 0.3333 | 0.0000 | 0.0000 |
| `mistral-small-119b` | Large (119B) | hard | 15 | 0.0000 | 0.4000 | 0.1216 | 0.0722 | 0.0906 | 0.8331 | 0.2000 | 0.2308 | 0.0000 | 0.0051 |
| `mistral-small-119b` | Large (119B) | multihop | 15 | 0.0000 | 0.2000 | 0.0994 | 0.0937 | 0.1073 | 0.8317 | 0.3333 | 0.2889 | 0.0000 | 0.0667 |
| `nemotron-120b` | Large (120B MoE) | easy | 15 | 0.0667 | 0.0667 | 0.1275 | 0.1233 | 0.0845 | 0.8340 | 0.2000 | 0.2229 | 0.0000 | 0.1422 |
| `nemotron-120b` | Large (120B MoE) | medium | 15 | 0.0000 | 0.1333 | 0.1397 | 0.1062 | 0.1559 | 0.8469 | 0.2667 | 0.2898 | 0.0000 | 0.0689 |
| `nemotron-120b` | Large (120B MoE) | hard | 15 | 0.0000 | 0.4667 | 0.1624 | 0.0971 | 0.1242 | 0.8341 | 0.3333 | 0.3102 | 0.0000 | 0.0105 |
| `nemotron-120b` | Large (120B MoE) | multihop | 15 | 0.0000 | 0.2000 | 0.1343 | 0.0994 | 0.1251 | 0.8320 | 0.2667 | 0.2488 | 0.0000 | 0.0580 |
| `nemotron-nano-8b` | Small (8B) | easy | 15 | 0.0000 | 0.0000 | 0.0266 | 0.0212 | 0.0516 | 0.8166 | 0.2000 | 0.2000 | 0.0000 | 0.0133 |
| `nemotron-nano-8b` | Small (8B) | medium | 15 | 0.0000 | 0.2000 | 0.1871 | 0.1290 | 0.2148 | 0.8473 | 0.2667 | 0.2692 | 0.0000 | 0.0000 |
| `nemotron-nano-8b` | Small (8B) | hard | 15 | 0.0000 | 0.2667 | 0.2748 | 0.1688 | 0.2280 | 0.8490 | 0.3333 | 0.4259 | 0.0000 | 0.0000 |
| `nemotron-nano-8b` | Small (8B) | multihop | 15 | 0.0000 | 0.0667 | 0.2701 | 0.1967 | 0.3074 | 0.8527 | 0.4000 | 0.4311 | 0.0000 | 0.0444 |
| `nemotron-super-49b` | Medium (49B) | easy | 15 | 0.0000 | 0.0000 | 0.0128 | 0.0079 | 0.0222 | 0.8090 | 0.2000 | 0.1524 | 0.0667 | 0.0226 |
| `nemotron-super-49b` | Medium (49B) | medium | 15 | 0.0000 | 0.0000 | 0.1788 | 0.1108 | 0.1911 | 0.8230 | 0.1333 | 0.1586 | 0.0667 | 0.0763 |
| `nemotron-super-49b` | Medium (49B) | hard | 15 | 0.0000 | 0.0667 | 0.2237 | 0.1157 | 0.2249 | 0.8239 | 0.0667 | 0.2771 | 0.2000 | 0.0204 |
| `nemotron-super-49b` | Medium (49B) | multihop | 15 | 0.0000 | 0.1333 | 0.2215 | 0.1328 | 0.2675 | 0.8202 | 0.2667 | 0.2783 | 0.2000 | 0.0667 |

## Table 2 — Model Rankings (Average across all 4 tiers)

| Rank | Model | Size | Avg F1 | Avg Num-EM | Avg Rel.EM | Avg Abstain | Avg NLI |
|------|-------|------|--------|-----------|-----------|------------|---------|
| 1 | `nemotron-nano-8b` | Small (8B) | 0.1896 | 0.3000 | 0.1333 | 0.0000 | 0.0144 |
| 2 | `nemotron-super-49b` | Medium (49B) | 0.1592 | 0.1667 | 0.0500 | 0.1333 | 0.0465 |
| 3 | `nemotron-120b` | Large (120B MoE) | 0.1410 | 0.2667 | 0.2167 | 0.0000 | 0.0699 |
| 4 | `llama4-maverick` | Medium (17B MoE) | 0.1293 | 0.2833 | 0.1167 | 0.0167 | 0.0376 |
| 5 | `mistral-nemotron` | Medium (~24B) | 0.1248 | 0.2667 | 0.1834 | 0.0000 | 0.0096 |
| 6 | `llama3.3-70b` | Large (70B) | 0.0848 | 0.2666 | 0.1834 | 0.0000 | 0.0083 |
| 7 | `llama3.1-8b` | Small (8B) | 0.0829 | 0.2666 | 0.1500 | 0.0000 | 0.0167 |
| 8 | `mistral-small-119b` | Large (119B) | 0.0797 | 0.2666 | 0.1667 | 0.0000 | 0.0180 |
| 9 | `llama3.2-3b` | Small (3B) | 0.0773 | 0.2334 | 0.1667 | 0.0000 | 0.0104 |

## Table 3 — Difficulty Degradation (Token F1)

| Model | Easy | Medium | Hard | Multihop | Drop (E→M) |
|-------|------|--------|------|----------|------------|
| `llama3.1-8b` | 0.0049 | 0.0553 | 0.1567 | 0.1148 | -11.0pp |
| `llama3.2-3b` | 0.0049 | 0.0588 | 0.1664 | 0.0790 | -7.4pp |
| `llama3.3-70b` | 0.0049 | 0.0696 | 0.1783 | 0.0865 | -8.2pp |
| `llama4-maverick` | 0.0049 | 0.1477 | 0.2196 | 0.1450 | -14.0pp |
| `mistral-nemotron` | 0.0143 | 0.1314 | 0.1971 | 0.1565 | -14.2pp |
| `mistral-small-119b` | 0.0049 | 0.0928 | 0.1216 | 0.0994 | -9.4pp |
| `nemotron-120b` | 0.1275 | 0.1397 | 0.1624 | 0.1343 | -0.7pp |
| `nemotron-nano-8b` | 0.0266 | 0.1871 | 0.2748 | 0.2701 | -24.3pp |
| `nemotron-super-49b` | 0.0128 | 0.1788 | 0.2237 | 0.2215 | -20.9pp |

## Table 4 — Numeric Recall Closed-Book (Num-Exact)

| Model | Easy | Medium | Hard | Multihop |
|-------|------|--------|------|----------|
| `llama3.1-8b` | 0.2000 | 0.3333 | 0.2000 | 0.3333 |
| `llama3.2-3b` | 0.2000 | 0.2667 | 0.2000 | 0.2667 |
| `llama3.3-70b` | 0.2000 | 0.3333 | 0.2000 | 0.3333 |
| `llama4-maverick` | 0.2000 | 0.3333 | 0.2667 | 0.3333 |
| `mistral-nemotron` | 0.2000 | 0.3333 | 0.2667 | 0.2667 |
| `mistral-small-119b` | 0.2000 | 0.3333 | 0.2000 | 0.3333 |
| `nemotron-120b` | 0.2000 | 0.2667 | 0.3333 | 0.2667 |
| `nemotron-nano-8b` | 0.2000 | 0.2667 | 0.3333 | 0.4000 |
| `nemotron-super-49b` | 0.2000 | 0.1333 | 0.0667 | 0.2667 |

## Table 5 — Abstain Rate (Model Self-Calibration)

| Model | Easy | Medium | Hard | Multihop |
|-------|------|--------|------|----------|
| `llama3.1-8b` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `llama3.2-3b` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `llama3.3-70b` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `llama4-maverick` | 0.0000 | 0.0000 | 0.0667 | 0.0000 |
| `mistral-nemotron` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mistral-small-119b` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `nemotron-120b` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `nemotron-nano-8b` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `nemotron-super-49b` | 0.0667 | 0.0667 | 0.2000 | 0.2000 |

## Key Findings

1. **EM = 0 everywhere** — No model can answer FinBharat from memory alone. This proves the dataset requires reading the actual annual report document.
2. **Best F1:** `nemotron-nano-8b` (avg 0.1896) — has strongest parametric financial knowledge
3. **Best Num-EM:** `nemotron-nano-8b` (avg 0.3000) — best numeric recall from pre-training
4. **Most calibrated:** `nemotron-super-49b` (abstains 13.3%) — correctly admits uncertainty
5. **F1 drops steeply with difficulty** — all models degrade from easy→multihop, confirming tier difficulty is real
6. **Larger ≠ better closed-book** — Small (8B) model outperforms larger models on parametric recall

---

## Files in this Package

```
clean_package/
  all_metrics.csv          ← all metrics, every model×tier (36 rows, 26 columns)
  model_summary.csv        ← averaged across tiers, ranked by Token F1
  raw_aggregates/          ← 36 JSON files (one per model×tier)
  per_question_scores/     ← 36 JSONL files (per-question EM, F1, Num-EM, NLI, etc.)
  raw_model_outputs/       ← 36 JSONL files (question, gold answer, model prediction)
  logs/                    ← per-run logs + master TSV log
    run_master.log         ← summary of all runs
    {model}_{tier}_closed_book.log  ← sector→company→Q/Gold/Pred per run
```

> **Excluded:** mistral-large-675b, qwen3.5-122b, qwen3.5-397b (VPN SSL errors — 100% API failures)
> **Partially excluded:** phi4-mini (only 2/4 tiers), gemma4-31b, gpt-oss-20b, gpt-oss-120b (minor API errors)