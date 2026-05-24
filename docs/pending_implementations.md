# FinBharat — Pending Implementations

> Last updated: May 2026 (synced with actual codebase state)
> Track against: `finBharat_plan_updated.md`

---

## Status Legend
- ✅ Done & tested
- ⚠️ Partial
- ❌ Not started
- 🔴 Blocker (paper cannot be submitted without this)
- 🟠 High priority
- 🟡 Medium priority
- 🟢 Low / nice-to-have

---

## 1. Metrics & Evaluation

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 1.1 | EM, Relaxed EM, Token F1 | ✅ | — | Includes sign-direction equivalence, cross-unit conversion |
| 1.2 | ROUGE-L | ✅ | — | Wired into all runs |
| 1.3 | BERTScore (P / R / F1, batched) | ✅ | — | roberta-large, truncation fix for long answers |
| 1.4 | METEOR | ✅ | — | NLTK wordnet, lazy download, graceful fallback |
| 1.5 | Num-Exact, Num-F1, MAPE | ✅ | — | Indian unit map, MAPE capped at 1000%, small-denom guard |
| 1.6 | Tol-1, Tol-5, Tol-10 | ✅ | — | Done |
| 1.7 | Directional accuracy | ✅ | — | Direction words + signed numbers (-30%) + accounting parens (30%) |
| 1.8 | DeBERTa NLI entailment | ✅ | — | cross-encoder/nli-deberta-v3-small, correct (premise, hypothesis) usage |
| 1.9 | Evidence traceability | ✅ | — | Done |
| 1.10 | Abstain rate | ✅ | — | 10-phrase detector; tracks model self-refusals |
| 1.11 | LLM Numerical Judge | ✅ | — | NumericalJudge class, direct httpx, disk cache, --llm-judge CLI flag |
| 1.12 | Bootstrap 95% CI | ✅ | — | All metrics in every aggregate JSON; scipy, 1000 iterations |
| 1.13 | Paired bootstrap test | ✅ | — | compute_significance_table() in metrics/stats.py |
| 1.14 | Cross-unit canonical comparison | ✅ | — | extract_canonical_value(), are_numerically_equivalent() |
| 1.15 | Sector-wise aggregation | ✅ | — | by_sector in every aggregate JSON |
| 1.16 | BRSR/ESG subset filter | ✅ | — | is_brsr_section(), load_brsr_subset(), brsr_subset in aggregate |
| 1.17 | verification_anchors parsed | ✅ | — | VerificationAnchors: calculation_inputs, hop_count, is_red_flag |
| 1.18 | Red-flag detection (heuristic) | ✅ | — | alignment_status == "contradiction" → is_red_flag; heuristic detection rate |
| 1.19 | Hop-count breakdown | ✅ | — | by_hop_count in aggregate (2-hop, 3-hop, 4-hop, 5+-hop) |
| 1.20 | **MAPE/Tol filtered to numerical questions** | ❌ | 🔴 | MAPE and Tol-1/5/10 currently averaged over ALL questions including Text Only (where they're meaningless 0s). Must only aggregate requires_calculation=True questions. |
| 1.21 | **Hallucination type taxonomy** | ❌ | 🔴 | Classify NLI=CONTRADICTION cases as: Numeric Fabrication / Miscalculation / Misattribution / Over-interpretation. This is the key faithfulness finding in the paper. |
| 1.22 | **Num-Exact by operation type** | ❌ | 🟠 | Direct Lookup vs YoY vs Ratio vs Multivariate. Derivable from len(calculation_inputs) in verification_anchors. |
| 1.23 | LLM faithfulness judge (full-answer) | ❌ | 🟡 | Beyond numerical: judge whether the entire answer is faithful. GPT-4o or llama3.3-70b. Calibrate against human labels. |
| 1.24 | Atomic claim decomposition | ❌ | 🟡 | FinGround-style: decompose answer into claims, verify each against evidence. |
| 1.25 | Citation / attribution metrics | ❌ | 🟡 | Citation precision/recall/F1. ARQA-style. |

---

## 2. Dataset & Data Pipeline

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 2.1 | Data loader (all 4 difficulties) | ✅ | — | Easy, Medium, Hard, Multihop |
| 2.2 | Chunk cache | ✅ | — | Identity-based cache, one disk read per company |
| 2.3 | Train/dev/test splits | ✅ | — | splits.json: train=112, dev=37, test=38 (stratified, no leakage) |
| 2.4 | Table serializer (HTML→Markdown→Linearized) | ✅ | — | _serialize_tables(), --table-format CLI flag |
| 2.5 | calculation_inputs for Num-Exact | ❌ | 🟠 | Hard/multihop has gold operands. Use them instead of extracting from free-text answer. |
| 2.6 | Oversized table handling | ❌ | 🟠 | oversized_table field exists; no fallback strategy yet. |
| 2.7 | Global QA index | ❌ | 🟡 | No single index over all 84K QA pairs. |
| 2.8 | **Human validation (500 QA, κ scores)** | ❌ | 🔴 | CANNOT BE CODED. Recruit 3 annotators. Annotate 125/tier. Fleiss' κ > 0.7. Biggest reviewer concern. |
| 2.9 | Evidence grounding audit (1000 QA) | ❌ | 🟠 | Number hallucination rate, table grounding rate. Section 3 of paper. |
| 2.10 | BRSR KPI extraction accuracy | ❌ | 🟡 | Emissions, water, energy, gender ratios exact match. |

---

## 3. Model Evaluation

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 3.1 | Llama-3.1-8B | ✅ | — | Tested, all 4 tiers, pilot scale |
| 3.2 | Llama-3.3-70B | ✅ | — | Tested, all 4 tiers, pilot scale |
| 3.3 | Nemotron-Nano-8B | ✅ | — | Tested, all 4 tiers, pilot scale |
| 3.4 | **Run all 4 tiers at scale (187 companies)** | ❌ | 🔴 | Pilot only (15–30 QA). Full run needed for paper. |
| 3.5 | **5+ models across size tiers** | ❌ | 🔴 | Need: ~4B, ~8B, ~49B, ~70B, GPT-4o. Currently 3 tested models. |
| 3.6 | **Full ablation at scale** | ❌ | 🔴 | zero_shot + closed_book + few_shot across all companies, all tiers. |
| 3.7 | OpenAI provider path (gpt-4o) | ⚠️ | 🟠 | Defined in PREDEFINED_MODELS but untested. |
| 3.8 | Context window size ablation | ❌ | 🟡 | Full bundle vs 2k/4k/8k truncation. |
| 3.9 | Program-of-Thought prompting | ❌ | 🟡 | Code generation for numerical questions. |

---

## 4. Ablation Studies

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 4.1 | Open-book vs Closed-book | ✅ | — | EM 0.67→0.00. Strong result proving dataset is not memorized. |
| 4.2 | Zero-shot vs Few-shot | ✅ | — | MAPE 3.8%→0.54%. Format-learning effect confirmed. |
| 4.3 | Modality (Table/Text/Table+Text/NumCalc) | ✅ | — | by_question_type in aggregates. Need scale for NumCalc. |
| 4.4 | **Difficulty degradation curve** | ❌ | 🔴 | Core Figure 2. Needs full-scale runs all 4 tiers. |
| 4.5 | Table format ablation | ✅ | — | --table-format html/markdown/linearized implemented. Not run at scale yet. |
| 4.6 | Context window size ablation | ❌ | 🟡 | Truncate bundle to 2k/4k/8k tokens. |
| 4.7 | Reasoning mode (CoT vs PoT) | ❌ | 🟡 | Chain-of-Thought vs Program-of-Thought. |

---

## 5. Infrastructure

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 5.1 | CLI: evaluate, sample, models, analyze | ✅ | — | Full CLI working |
| 5.2 | Retry + key rotation (3 NIM keys) | ✅ | — | Exponential backoff |
| 5.3 | Generation cache | ✅ | — | JSONL per model/difficulty/regime |
| 5.4 | Judge cache | ✅ | — | MD5-keyed JSONL, re-runs are free |
| 5.5 | tqdm progress bars | ✅ | — | All loops: context-building, generation, judge |
| 5.6 | Results analyzer CLI | ✅ | — | finbharat analyze --all |
| 5.7 | **Parallel model evaluation** | ❌ | 🟠 | 5+ models × 4 tiers runs sequentially now. ThreadPoolExecutor for speed. |
| 5.8 | RAG evaluation track | ❌ | 🟡 | Dense/hybrid/hierarchical/agentic retrieval. |

---

## 6. Paper Requirements

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 6.1 | **Human validation (κ > 0.7)** | ❌ | 🔴 | Most critical. Cannot be coded. |
| 6.2 | **Cross-benchmark comparison (FinQA/TAT-QA)** | ❌ | 🔴 | Shows FinBharat is harder. Reviewers will ask for this. |
| 6.3 | **Error taxonomy (200 cases, manual)** | ❌ | 🟠 | 50 per tier, classify: Calc Error / Unit Mismatch / Misattribution / Evidence Ignored. |
| 6.4 | Hindi evaluation subset | ❌ | 🟡 | IndicTrans2 + 2000 QA pairs. |
| 6.5 | HuggingFace dataset release | ❌ | 🟠 | Dataset card, test/dev splits, data loader. |
| 6.6 | GitHub: eval scripts + prompts + raw outputs | ⚠️ | 🟠 | Repo is public. Raw outputs and prompts not exported yet. |
| 6.7 | **Limitations section** | ❌ | 🔴 | EMNLP desk-rejects without it. Single FY, LLM QA, one company/sector, no vision. |
| 6.8 | **Ethics statement** | ❌ | 🔴 | EMNLP desk-rejects without it. BSE/NSE public data, no PII, financial disclaimer. |

---

## Implementation Order (Updated)

### This sprint — code-only, no external dependency
1. **Fix MAPE/Tol aggregation** — filter to requires_calculation=True only (1.20)
2. **Hallucination type taxonomy** — classify NLI contradiction cases (1.21)
3. **Num-Exact by operation type** — from verification_anchors.calculation_inputs (1.22)
4. **verification_anchors → calculation_inputs for Num-Exact** — use gold operands (2.5)

### Requires running (compute)
5. Full-scale runs: 5 models × 4 tiers × 187 companies
6. Closed-book + few-shot ablations at scale
7. Table format ablation run
8. LLM judge on all numerical questions (--llm-judge)

### Requires humans — START NOW (longest lead time)
9. **Human validation** (2.8) — longest lead time item
10. **Error taxonomy** (6.3) — can start after full model runs

### Requires additional work
11. Cross-benchmark comparison (6.2)
12. Hindi translation (6.4)
13. HuggingFace release (6.5)
14. Limitations + Ethics sections (6.7, 6.8)
