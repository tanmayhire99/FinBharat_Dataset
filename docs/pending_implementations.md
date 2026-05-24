# FinBharat — Pending Implementations

> Last updated: May 2026  
> Track against: `finBharat_plan_updated.md`

---

## Status Legend
- ✅ Done
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
| 1.1 | EM, Relaxed EM, Token F1 | ✅ | — | Done, tested |
| 1.2 | ROUGE-L | ✅ | — | Done, wired |
| 1.3 | BERTScore (batched) | ✅ | — | Done, uses roberta-large |
| 1.4 | Num-Exact, Num-F1, MAPE | ✅ | — | Done, Indian unit map |
| 1.5 | Tol-1, Tol-5, Tol-10 | ✅ | — | Done |
| 1.6 | Directional accuracy | ✅ | — | Done |
| 1.7 | DeBERTa NLI entailment | ✅ | — | cross-encoder/nli-deberta-v3-small |
| 1.8 | Evidence traceability | ✅ | — | Done |
| 1.9 | **Statistical significance** | ❌ | 🔴 | Bootstrap 95% CI + paired bootstrap p-values for all main result tables. Use `scipy.stats`. Required at EMNLP. |
| 1.10 | **Numerical Calculation routing** | ⚠️ | 🟠 | `question_type=="Numerical Calculation"` and `requires_calculation=True` are logged but MAPE/Tol only make sense for these. Should skip non-numeric gold answers from Tol/MAPE aggregation. |
| 1.11 | **verification_anchors → Num-Exact improvement** | ❌ | 🟠 | Hard/multihop QA has `calculation_inputs` (list of gold numbers). Use these for stricter numeric matching instead of extracting from free-text answer. |
| 1.12 | **Red-flag detection metric (Hard)** | ❌ | 🟠 | `alignment_status: contradiction/consistent` in hard QA verification_anchors. Binary classification: precision/recall/F1 for issue detection. |
| 1.13 | **Hop-count analysis (Multihop)** | ❌ | 🟠 | `cross_section_sources` in verification_anchors tells hop count. Break multihop results by 2-hop/3-hop/4+. |
| 1.14 | **Sector-wise aggregation** | ❌ | 🟠 | `sector` field exists on every EvalResult. Add `by_sector` breakdown to `aggregate_results`. |
| 1.15 | LLM-judge harness | ❌ | 🟡 | GPT-4o / Llama-70B judge for faithfulness. Calibrate against human labels. |
| 1.16 | Atomic claim decomposition | ❌ | 🟡 | Decompose answers into atomic claims, verify each against evidence. FinGround-style. |
| 1.17 | Citation / attribution metrics | ❌ | 🟡 | Citation precision/recall/F1. ARQA-style. |
| 1.18 | **METEOR metric** | ❌ | 🟡 | Add alongside ROUGE-L for backward compatibility with older literature. |

---

## 2. Dataset & Data Pipeline

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 2.1 | Data loader (all 4 difficulties) | ✅ | — | Done |
| 2.2 | Chunk cache | ✅ | — | Done |
| 2.3 | Train/dev/test splits | ✅ | — | splits.json: train=112, dev=37, test=38 |
| 2.4 | **BRSR/ESG subset filter** | ❌ | 🔴 | 748 BRSR sections exist (keywords: BRSR, Sustainability, Business Responsibility, ESG, Principle 1–9). Needs keyword filter to produce BRSR-QA subset. The paper's key differentiator. |
| 2.5 | **verification_anchors parsed and used** | ❌ | 🟠 | `calculation_inputs`, `cross_section_sources`, `alignment_status` exist on hard/multihop. Currently ignored. |
| 2.6 | **Oversized table handling** | ❌ | 🟠 | `oversized_table: true` field on chunks. Large tables cause context truncation. Need detection + fallback strategy. |
| 2.7 | **Table format serialization** | ⚠️ | 🟠 | Evidence passes raw HTML to models. No Markdown/linearized/key-value alternative. Table format ablation (HTML vs Markdown vs linearized) is a required ablation study. |
| 2.8 | Global QA index with unique IDs | ❌ | 🟡 | No single global index over all 84K+ QA pairs. Makes sampling/splits harder. |
| 2.9 | Human validation (500 QA) | ❌ | 🔴 | Recruit 3 domain experts (CAs/finance researchers). Annotate 125/tier. Compute Fleiss' κ. **Cannot be automated — needs real people.** |
| 2.10 | Evidence grounding audit (1000 QA) | ❌ | 🟠 | Number hallucination rate, table grounding rate. Intrinsic quality metrics for Section 3 of paper. |
| 2.11 | BRSR KPI extraction accuracy | ❌ | 🟡 | For BRSR Core KPIs (emissions, water, energy, gender ratios), measure exact match of numeric values. |

---

## 3. Model Evaluation

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 3.1 | Llama-3.1-8B (NIM) | ✅ | — | Tested, working |
| 3.2 | **Run all 4 difficulty tiers at scale** | ❌ | 🔴 | Currently only easy/30 QA tested. Need all 187 companies × 4 tiers. |
| 3.3 | **Multiple models (5+ minimum)** | ❌ | 🔴 | Paper needs size tier comparison: ~8B, ~14B, ~70B, MoE, GPT-4o. |
| 3.4 | **Zero-shot + few-shot + closed-book at scale** | ⚠️ | 🔴 | Implemented (regimes work), but only run on 30 QA pilot. Need full scale. |
| 3.5 | Model runner: OpenAI provider support | ⚠️ | 🟠 | GPT-4o defined in PREDEFINED_MODELS but uses NIM base URL. OpenAI provider path untested. |
| 3.6 | **Context window size ablation** | ❌ | 🟡 | Full bundle vs 2k/4k/8k token truncation. Shows impact of doc length. |
| 3.7 | Program-of-Thought (PoT) prompting | ❌ | 🟡 | Code generation for numeric questions. FinAgent-RAG showed significant gains. |
| 3.8 | Financial-specific models | ❌ | 🟡 | FinMA-7B, FinGPT as domain baselines. |

---

## 4. Ablation Studies

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 4.1 | Open-book vs Closed-book | ✅ | — | Implemented, pilot shows EM 0.67→0.00 — strong result |
| 4.2 | Zero-shot vs Few-shot | ✅ | — | Implemented, MAPE 3.8%→0.54% — strong format-learning result |
| 4.3 | Modality analysis (Table/Text/Table+Text/NumCalc) | ⚠️ | 🟠 | `by_question_type` breakdown exists in aggregates. But only 0% of easy QA is Numerical Calculation — need medium/hard/multihop runs to see the real breakdown. |
| 4.4 | **Difficulty degradation curve** | ❌ | 🔴 | Easy→Medium→Hard→Multihop performance degradation. Core Figure 2 in paper. Needs all 4 tiers run. |
| 4.5 | **Table format ablation** | ❌ | 🟠 | HTML vs Markdown vs linearized key-value. Requires table serializer. |
| 4.6 | Context window size ablation | ❌ | 🟡 | Truncate bundle to 2k/4k/8k tokens, measure degradation. |
| 4.7 | Reasoning mode: CoT vs PoT | ❌ | 🟡 | Chain-of-Thought vs Program-of-Thought for numeric questions. |

---

## 5. Infrastructure

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 5.1 | CLI: evaluate, sample, models, analyze | ✅ | — | Done |
| 5.2 | Retry + key rotation | ✅ | — | Done |
| 5.3 | Result caching (generation cache) | ✅ | — | Done |
| 5.4 | tqdm progress bars | ✅ | — | Done |
| 5.5 | Results analyzer + regime comparison | ✅ | — | Done |
| 5.6 | **Parallel model evaluation** | ❌ | 🟠 | Currently models run sequentially. Running 5+ models × 4 tiers = slow. Add `--parallel` flag with ThreadPoolExecutor. |
| 5.7 | **RAG evaluation track** | ❌ | 🟡 | 5 retrieval architectures (dense, hybrid, hierarchical, graph, agentic). Full pipeline. |
| 5.8 | OpenAI provider path | ⚠️ | 🟠 | Defined but untested. Needs separate base_url handling. |

---

## 6. Paper Requirements

| # | Item | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 6.1 | Human validation (κ > 0.7) | ❌ | 🔴 | **Cannot be coded — needs human annotators.** Biggest reviewer concern. |
| 6.2 | Cross-benchmark comparison | ❌ | 🔴 | Run same models on FinQA / TAT-QA test set. Shows FinBharat is harder. |
| 6.3 | Hindi evaluation subset | ❌ | 🟡 | Translate 2000 QA pairs. IndicTrans2 + human post-edit. |
| 6.4 | Error taxonomy (200 cases, manual) | ❌ | 🟠 | Sample 50 errors/tier from best model. Classify: Calculation Error, Unit Mismatch, Misattribution, Evidence Ignored, etc. |
| 6.5 | HuggingFace dataset release | ❌ | 🟠 | Dataset card, test/dev splits, data loader script. |
| 6.6 | GitHub repo + leaderboard | ❌ | 🟡 | Eval scripts, prompts, raw outputs, HF Spaces leaderboard. |
| 6.7 | Limitations section | ❌ | 🔴 | EMNLP desk-rejects without it. Single FY, LLM-generated QA, one company/sector, no vision. |
| 6.8 | Ethics statement | ❌ | 🔴 | EMNLP desk-rejects without it. BSE/NSE data is public, no PII, financial disclaimer. |

---

## Implementation Order (Recommended)

### This sprint (code-only, no external dependency)
1. `verification_anchors` integration (1.11, 1.12, 1.13, 2.5)
2. BRSR/ESG subset filter (2.4)
3. Statistical significance — bootstrap CI + paired tests (1.9)
4. Sector-wise aggregation (1.14)
5. Table → Markdown serializer + table format ablation (2.7, 4.5)
6. OpenAI provider path fix (5.8, 3.5)

### Requires running (compute/API)
7. All 4 difficulties × 5+ models × all companies
8. Closed-book ablation at scale
9. Difficulty degradation curve (Figure 2)

### Requires humans
10. Human validation (2.9) — START NOW, longest lead time
11. Error taxonomy (6.4) — can start after full model runs

### Requires additional work
12. Hindi translation (6.3)
13. Cross-benchmark comparison (6.2)
14. RAG evaluation (5.7)
15. HuggingFace + leaderboard release (6.5, 6.6)
