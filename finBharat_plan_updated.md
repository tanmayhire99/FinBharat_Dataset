# FinBharat — Updated Benchmark & Paper Plan for EMNLP 2026

> **Updated:** May 2026 | **Target:** EMNLP 2026 (ARR May 2026 cycle, commitment phase)
> **Original plan reviewed:** `finBharat_plan.md`

---

## 0. Executive Summary of Changes

The original plan is solid in scope, task definition, and metric design. However, after analyzing the actual dataset and surveying the 2025–2026 financial NLP landscape (PHANTOM, FAITH, FinLongDocQA, ARQA, FABRIC, BhashaBench-Finance, FinGround, MULTIFINBEN, etc.), the following gaps must be addressed for EMNLP acceptance:

| # | Issue | Severity |
|---|-------|----------|
| 1 | **No human validation** of LLM-generated QA pairs — biggest reviewer concern | Critical |
| 2 | **No multilingual / Indic-language coverage** — missed differentiator for India-focused benchmark | Critical |
| 3 | **No RAG pipeline evaluation** — all existing 2025–26 financial benchmarks include RAG tracks | High |
| 4 | **Model panel is vague** — "family A-7B" placeholders; needs concrete, current model names | High |
| 5 | **No cross-benchmark comparison** — reviewers will ask "how does this compare to FinQA / PHANTOM / FAITH?" | High |
| 6 | **Dataset quality analysis missing** — no inter-annotator agreement, no answerability audit, no evidence-grounding verification | High |
| 7 | **Missing BRSR/ESG-specific evaluation track** — the data is rich in BRSR content but the plan treats it as generic "section" analysis | Medium |
| 8 | **No statistical significance testing** | Medium |
| 9 | **Paper framing too narrow** — "hallucination benchmark" is not the strongest hook; need India-specific + table-heavy + BRSR narrative | Medium |
| 10 | **Missing Limitations and Ethics sections** — desk-reject risk at EMNLP | Medium |
| 11 | **No leaderboard / HuggingFace release plan** — expected for benchmark papers in 2026 | Medium |
| 12 | **No conversational / multi-turn QA** — ConvFinQA-style extension | Low-Medium |

Each issue is addressed in the sections below.

---

## 1. Revised High-Level Goal & Positioning

### 1.1 Rename to FinBharat

"FinAR-Hall" is generic. **FinBharat** signals:
- **India-focused** (Bharat = India in Hindi/Sanskrit)
- **Financial** domain
- Covers both **QA accuracy** and **hallucination/faithfulness**
- Echos naming convention of recent benchmarks (FinBen, FinMTEB, BhashaBench-Finance, FABRIC)

### 1.2 Revised Contribution Bullets

1. **FinBharat**: the first large-scale, India-focused financial QA and hallucination benchmark built from 187 annual reports (BSE/NSE-listed companies) across 187 sectors, with **84K+ QA pairs** across 4 difficulty tiers, structured evidence links, and rich BRSR/ESG content.

2. **Multi-dimensional evaluation suite**: numeric correctness (Num-Exact/Num-F1/MAPE), evidence grounding, NLI-based faithfulness, LLM-judge faithfulness with hallucination type taxonomy, and BRSR-specific compliance metrics — all with **human-validated gold subsets**.

3. **RAG evaluation track**: the first Indian financial benchmark to include a full RAG pipeline evaluation (retrieval → generation → faithfulness) with multiple retrieval architectures.

4. **Comprehensive model evaluation**: 12+ models (open + closed), with cross-benchmark comparisons to FinQA, TAT-QA, and PHANTOM, plus a detailed error taxonomy and sector-wise analysis.

5. **Bilingual extension (English + Hindi)**: a translated subset demonstrating cross-lingual performance gaps for Indian financial QA.

### 1.3 Key Differentiators from Existing Benchmarks

| Feature | FinQA | TAT-QA | PHANTOM | FAITH | FinLongDocQA | ARQA | FABRIC | **FinBharat** |
|---------|-------|--------|---------|-------|--------------|------|--------|---------------|
| India-focused annual reports | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (advice) | ✓ |
| BRSR/ESG content | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 4 difficulty tiers | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (3) | ✗ | ✓ |
| Full RAG pipeline eval | ✗ | ✗ | ✗ | ✗ | Partial | ✗ | ✗ | ✓ |
| Hallucination detection | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| Forensic/hard QA | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Multilingual (Indic) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (6 lang) | ✓ (EN+HI) |
| Cross-benchmark comparison | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Human-validated subset | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 187 sectors | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

---

## 2. Dataset: Consolidated Statistics & Quality Assurance

### 2.1 Actual Dataset Statistics (from analysis)

| Metric | Count |
|--------|-------|
| Sectors | 187 |
| Companies | 187 (1 per sector, FY2025) |
| Chunks | ~182,000 (across h_m + e_m) |
| Semantic bundles | ~22,700 |
| Easy QA pairs | ~21,660 |
| Medium QA pairs | ~21,340 |
| Hard QA pairs | ~19,940 |
| Multihop QA pairs | ~21,350 |
| **Total QA pairs** | **~84,300** |
| Question types | Text Only, Table Only, Table with Text, Numerical Calculation |

### 2.2 NEW: Human Validation (Critical Addition)

**Problem:** All QA pairs are LLM-generated. Reviewers at EMNLP will flag this as a major weakness without human validation.

**Solution:**

1. **Stratified sampling for human validation:**
   - Sample 500 QA pairs (125 per difficulty tier), stratified by question type and sector.
   - Recruit 3 domain experts (CAs / finance professionals / academic researchers) for annotation.
   
2. **Human validation tasks:**
   - **Answer correctness:** Is the gold answer actually correct given the evidence? (Yes/No/Partially)
   - **Evidence sufficiency:** Does the evidence fully support the answer? (Yes/No)
   - **Answerability:** Is the question answerable from the provided evidence? (Yes/No)
   - **Hallucination labeling:** For model-generated answers, label as {faithful, intrinsic hallucination, extrinsic hallucination} with subtype flags.
   
3. **Inter-annotator agreement:**
   - Report Fleiss' κ for correctness and Cohen's κ for pairwise.
   - Target κ > 0.7. If below, resolve via adjudication and report both raw and adjudicated scores.

4. **Dataset quality report (Section 3 of paper):**
   - Gold answer accuracy rate (% correct after human review).
   - Evidence sufficiency rate.
   - Error types in LLM-generated QA (fabrication, wrong extraction, misattribution, etc.).
   - Revision protocol: correct or discard flagged items.

### 2.3 NEW: Evidence Grounding Audit

- For 1,000 random QA pairs, compute:
  - **Evidence traceability score** (already in plan) — but also report aggregate statistics.
  - **Number hallucination rate**: fraction of numbers in gold answers NOT found in evidence text.
  - **Table grounding rate**: for Table/Table-with-Text QAs, what fraction of answer numbers can be traced to specific table cells?
  
- Report these as **intrinsic dataset quality metrics** in the paper. This demonstrates rigor that reviewers expect.

### 2.4 Data Split (Revised)

- **Test:** 20% of reports (38 companies) — held out, never used in any development.
- **Dev:** 20% of reports (38 companies).
- **Train:** 60% of reports (111 companies) — optional for fine-tuning experiments.

**Stratification constraints:**
- Each split must contain: ≥5 companies with BRSR-heavy content, ≥3 financial-sector companies, ≥3 manufacturing companies, ≥3 services companies.
- Ensure all question types and difficulty tiers are represented in each split.

---

## 3. Tasks (Revised & Extended)

### 3.1 Task 1: Core QA (FinBharat-QA) — Same as original

No changes needed. Metrics are well-defined.

### 3.2 Task 2: Faithfulness & Hallucination (FinBharat-Hall) — Enhanced

**Additions:**

1. **Hallucination type taxonomy** (aligned with PHANTOM and FinGround):
   - **Intrinsic hallucination:** Answer contradicts evidence.
     - Numeric fabrication (number not in evidence)
     - Numeric miscalculation (arithmetic error)
     - Misattribution (wrong row/year/segment)
   - **Extrinsic hallucination:** Answer includes information not in evidence.
     - Over-interpretation / speculation
     - External knowledge injection
   - **Cherry-picking:** Selective evidence usage that distorts meaning.

2. **Atomic claim decomposition** (inspired by FinGround):
   - Decompose each model answer into atomic claims.
   - Verify each claim against evidence.
   - Report claim-level faithfulness (fraction of entailed claims).

3. **Human-labeled hallucination detection test set:**
   - For 500 QA pairs, generate answers from 3 models.
   - Human-annotate 1,500 (answer, evidence) pairs for faithfulness.
   - Use this as gold standard for hallucination detection evaluation.
   - Report accuracy, precision, recall, F1 of NLI and LLM-judge against human labels.

### 3.3 Task 3: Evidence Quality & Retrieval — Same as original

No changes needed.

### 3.4 NEW: Task 4 — RAG Pipeline Evaluation (FinBharat-RAG)

**Motivation:** Every major 2025–26 financial benchmark (FinLongDocQA, FinAgentBench, FinDoc-RAG, OmniEval) includes RAG evaluation. Without this track, the paper will look outdated.

**Task definition:**

Given `(question, full document chunks)`, execute a RAG pipeline: retrieve → generate → verify.

**Pipelines to evaluate:**

1. **Dense retrieval + LLM:** BGE-base / GTE-Qwen2-1.5B + LLM generator.
2. **Hybrid retrieval (BM25 + dense) + LLM:** Standard hybrid with reciprocal rank fusion.
3. **Raptor-style hierarchical retrieval + LLM:** Cluster-based summary retrieval.
4. **Graph-RAG:** Entity-graph enhanced retrieval.
5. **Agentic RAG (multi-round):** Iterative retrieve-reason-verify loop (inspired by FinLongDocAgent).

**Metrics:**

- Retrieval: Recall@k, MRR@k, nDCG@k (chunk level).
- End-to-end QA: EM, Num-Exact, BERTScore.
- Faithfulness: NLI entailment ratio + LLM-judge score on RAG-generated answers.
- Latency and cost per query.

**Key comparison:** Gold-bundle context vs. retrieved context — show the retrieval gap.

### 3.5 Task 5: Numeric & Table QA (FinBharat-Numeric) — Enhanced

**Additions:**

1. **Operation-type breakdown** (aligned with FAITH taxonomy):
   - Direct Lookup
   - Comparative Calculation (YoY, QoQ)
   - Bivariate Calculation (ratios, margins)
   - Multivariate Calculation (complex derivations)

2. **MAPE (Mean Absolute Percentage Error):**
   - For numeric answers, report MAPE alongside Num-Exact and Num-F1.
   - Also report **tolerance-based accuracy** (±1%, ±5%, ±10%) as in FinLongDocQA.

3. **Table structure analysis:**
   - Performance on simple tables vs. hierarchical tables vs. multi-row/multi-column tables.
   - Impact of oversized tables (flagged in `oversized_table` field).

### 3.6 Task 6: Multi-hop Reasoning (FinBharat-MultiHop) — Same as original

No changes needed. The hop-count analysis and support coverage metrics are good.

### 3.7 Task 7: Hard Forensic QA (FinBharat-Hard) — Enhanced

**Additions:**

1. **Red-flag detection as binary classification:**
   - Label each hard QA as "flags an issue" vs. "confirms compliance."
   - Evaluate models on their ability to identify red flags (precision, recall for issue detection).

2. **Governance risk scoring:**
   - For hard QAs in governance/auditor sections, create a composite governance risk score.
   - Correlate model performance with sector-level risk.

### 3.8 NEW: Task 8 — BRSR/ESG-Specific Evaluation (FinBharat-ESG)

**Motivation:** Your dataset is uniquely rich in BRSR content (Business Responsibility and Sustainability Report) — this is the India-specific differentiator. No existing financial benchmark evaluates ESG/BRSR QA.

**Task definition:**

- Filter QAs from BRSR sections (sections containing "BRSR", "Business Responsibility", "Sustainability", "ESG", "Principle 1–9").
- Evaluate separately to show how models handle ESG-specific questions.

**Metrics:**

- Standard QA metrics (EM, F1, Num-Exact) on BRSR subset.
- **KPI extraction accuracy:** For BRSR Core KPIs (emissions, water, energy, gender ratios, etc.), measure exact match of numeric values.
- **Compliance completeness:** What fraction of mandatory BRSR disclosures can the model correctly extract?
- **Greenwashing detection:** For hard QAs in BRSR sections, can models identify vague/unsupported ESG claims vs. specific, quantified claims?

### 3.9 NEW: Task 9 — Bilingual Evaluation (English + Hindi)

**Motivation:** FABRIC and BhashaBench-Finance demonstrate that Indian financial benchmarks must include Indic languages. FABRIC shows Indian models don't perform better on Indian finance in English — and there's a 5–7% drop from English to Hindi. This is a compelling finding.

**Design:**

1. Translate 2,000 QA pairs (500 per difficulty tier) from English to Hindi using:
   - Professional translators (preferred, for 500 pairs).
   - Google Translate / IndicTrans2 + human post-edit (for remaining 1,500).
   
2. Translate corresponding evidence chunks to Hindi.

3. Evaluate same model panel on Hindi QA and report:
   - Performance gap: English vs. Hindi.
   - Whether Indian-origin models (e.g., Sarvam, Navarasa) perform better on Hindi.
   - Whether code-switched (Hinglish) prompts help (as FABRIC found).

4. Report linguistic quality of translations (BLEU, ChrF against professional translator subset).

---

## 4. Metrics: Additions & Refinements

### 4.1 Add Statistical Significance Testing

- For all main results, report **bootstrap 95% confidence intervals** (1,000 iterations).
- For pairwise model comparisons, report **paired bootstrap test** p-values.
- Mark statistically significant differences (p < 0.05) in tables.

### 4.2 Add Tolerance-Based Numeric Accuracy

Following FinLongDocQA:
- **Tol-1**: accuracy within ±1% of gold value.
- **Tol-5**: accuracy within ±5%.
- **Tol-10**: accuracy within ±10%.

### 4.3 Add Relaxed Exact Match

- **Relaxed EM**: match after normalizing units (₹ in Crores vs. Lakhs vs. absolute), currency symbols, percentage signs.
- This is important because Indian reports use varying unit conventions.

### 4.4 Add Attribution / Citation Quality Metrics

Following ARQA and FINLFQA:
- **Citation precision:** Fraction of cited evidence chunks that are actually relevant.
- **Citation recall:** Fraction of required evidence chunks that are cited.
- **Citation F1.**

### 4.5 LLM-Judge Calibration

- Calibrate the LLM-judge against human labels on the 500-pair validation set.
- Report: Spearman correlation between judge scores and human faithfulness ratings.
- Report: ROC-AUC of judge for binary hallucination detection.
- Compare multiple judge models: GPT-4o, Claude-3.5-Sonnet, Llama-3.1-70B.

---

## 5. Experimental Design (Revised)

### 5.1 Model Panel (Concrete, Current Models)

**Open-weight models via NVIDIA NIM (or vLLM):**

| Category | Model | Params |
|----------|-------|--------|
| Small | Qwen3-8B-Instruct | 8B |
| Small | Llama-3.2-3B-Instruct | 3B |
| Small | Gemma-3-4B-Instruct | 4B |
| Medium | Qwen3-14B-Instruct | 14B |
| Medium | Llama-3.1-8B-Instruct | 8B |
| Medium | Mistral-Small-3.2 | 24B |
| Large | Qwen3-72B-Instruct | 72B |
| Large | Llama-3.3-70B-Instruct | 70B |
| Large | DeepSeek-R1-Distill-70B | 70B |
| Reasoning | DeepSeek-R1-0528 | 671B MoE |
| Reasoning | Qwen3-235B-A22B | 235B MoE |
| Indian | Sarvam-105B | 106B MoE |

**Closed models (API):**

| Model | Purpose |
|-------|---------|
| GPT-4o | Upper bound reference |
| GPT-5.2 (if available) | Frontier reference |
| Claude-3.5-Sonnet | Alternative frontier |

**Financial-specific models:**

| Model | Purpose |
|-------|---------|
| FinMA-7B | Domain-specific baseline |
| FinGPT | Domain-specific baseline |

**Total: 15+ models.** This is comparable to FinBen (21), MULTIFINBEN (21), and BizFinBench (25).

### 5.2 Evaluation Regimes

For each model:

1. **Zero-shot, open-book (gold context)** — primary result.
2. **Few-shot (5 examples), open-book (gold context)** — in-context learning effect.
3. **Zero-shot, closed-book** — knowledge vs. reading comprehension gap.
4. **Zero-shot, RAG (retrieved context)** — practical pipeline evaluation.

### 5.3 NEW: Cross-Benchmark Comparison

To show FinBharat's unique challenges, evaluate the same models on:
- **FinQA** (test set) — numeric reasoning baseline.
- **TAT-QA** (test set) — table-text hybrid baseline.
- **PHANTOM** — hallucination detection baseline.

Report: "Model X achieves Y on FinQA but only Z on FinBharat-Hard, demonstrating that..."

### 5.4 Ablation Studies (Enhanced)

1. **Open-book vs. Closed-book** (original) ✓
2. **Zero-shot vs. Few-shot** (original) ✓
3. **Gold context vs. RAG-retrieved context** (original) ✓
4. **NEW: Table serialization format:**
   - HTML table (current) vs. Markdown vs. Linearized key-value vs. Vertical bar format.
   - This is crucial — table format significantly affects model performance (shown by TAT-QA, FinQA).
5. **NEW: Context window size:**
   - Full bundle context vs. truncated to 2k/4k/8k tokens.
   - Shows impact of document length on performance.
6. **NEW: Reasoning mode:**
   - Standard generation vs. Chain-of-Thought vs. Program-of-Thought (code generation for numeric).
   - Follows FinAgent-RAG's finding that PoT significantly improves numeric accuracy.

### 5.5 Error Analysis (Enhanced)

1. **Scale:** 200 error cases (50 per difficulty tier), up from 100.
2. **Error taxonomy** (refined):
   - **Numeric errors:** Calculation error, Unit mismatch, Wrong operand, Scale error (Crores vs. Lakhs).
   - **Retrieval errors:** Wrong section, Wrong table, Missed footnote.
   - **Reasoning errors:** Wrong comparison direction, Incomplete multi-hop chain, Speculation beyond evidence.
   - **Formatting errors:** Non-numeric answer to numeric question, Verbose vs. concise.
   - **Language errors:** (Hindi subset) Translation artifact, Entity name mismatch.

3. **Quantitative error analysis:**
   - What fraction of numeric errors are due to unit/scale mismatches (uniquely Indian)?
   - What fraction of hallucinations are in BRSR sections vs. financial statements?
   - Correlation between error type and question type / difficulty.

4. **Qualitative examples:** 3–4 detailed case studies showing error → root cause → metric capture.

---

## 6. Paper Structure (Revised for 8+1 Pages)

**Page budget:** 8 content pages + 1 extra for reviewer responses + unlimited appendix.

| Section | Pages | Content |
|---------|-------|---------|
| 1. Introduction | 1.5 | Motivation, gap, contributions (5 bullets) |
| 2. Related Work | 1 | FinQA, TAT-QA, PHANTOM, FAITH, FinLongDocQA, ARQA, FABRIC, BhashaBench-Finance, MULTIFINBEN, FinBen, FinGround |
| 3. Dataset Construction | 1.5 | Pipeline, statistics, human validation (κ scores), evidence quality audit, BRSR content analysis |
| 4. Tasks & Metrics | 1.5 | All 9 tasks, formal metric definitions |
| 5. Experimental Setup | 0.75 | Model panel, splits, regimes, cross-benchmark setup |
| 6. Results | 2 | QA accuracy tables, faithfulness results, RAG results, BRSR results, Hindi results, cross-benchmark comparison |
| 7. Analysis | 1 | Error taxonomy, sector-wise analysis, ablation studies, table format analysis |
| 8. Conclusion & Release | 0.25 | Summary, open-source commitment |
| Limitations | 0.5 (outside page limit) | Required by EMNLP — desk reject without it |
| Ethics | 0.25 (outside page limit) | Data licensing, PII, financial advice disclaimer |
| Appendix | — | Full result tables, prompt templates, annotation guidelines, additional case studies |

### 6.1 Key Tables & Figures

1. **Table 1:** Dataset statistics comparison with existing benchmarks (like the table in §1.3 above).
2. **Table 2:** Overall QA results across all models × difficulty tiers.
3. **Table 3:** Faithfulness/hallucination detection results.
4. **Table 4:** RAG pipeline comparison (5 architectures × metrics).
5. **Table 5:** BRSR/ESG-specific results.
6. **Table 6:** English vs. Hindi performance gap.
7. **Table 7:** Cross-benchmark comparison (FinQA, TAT-QA, PHANTOM vs. FinBharat).
8. **Table 8:** Ablation studies summary.
9. **Figure 1:** Dataset construction pipeline.
10. **Figure 2:** Performance degradation curve across difficulty tiers.
11. **Figure 3:** Hallucination type distribution by section type.
12. **Figure 4:** Error taxonomy sunburst chart.
13. **Figure 5:** RAG retrieval gap (gold vs. retrieved context).

---

## 7. Engineering Roadmap (Revised & Prioritized)

### Phase 1: Data Consolidation & Quality (Weeks 1–3)

1. **Merge all per-company data** into unified indexes with global IDs.
2. **Create train/dev/test splits** with stratification.
3. **Run evidence grounding audit** on 1,000 QA pairs.
4. **Extract BRSR-specific QAs** into separate subset.
5. **Human validation** of 500 QA pairs (recruit annotators, design guidelines, compute κ).

### Phase 2: Metric Implementation (Weeks 2–4)

1. Numeric parsing + Num-Exact/Num-F1/MAPE + tolerance-based accuracy.
2. Relaxed EM with Indian unit normalization.
3. Boolean/directional accuracy.
4. BERTScore + ROUGE-L.
5. NLI-based entailment (DeBERTa-v3-large-mnli).
6. LLM-judge harness (GPT-4o, Claude, Llama-70B).
7. Evidence traceability + completeness.
8. Attribution/citation metrics.
9. **Statistical significance testing** (bootstrap CI, paired tests).

### Phase 3: Model Evaluation (Weeks 3–6)

1. **Core QA evaluation** — all models, all difficulty tiers, gold context.
2. **Faithfulness evaluation** — NLI + LLM-judge on all model outputs.
3. **Closed-book evaluation** — subset.
4. **Few-shot evaluation** — subset.
5. **RAG pipeline evaluation** — 5 retrieval architectures + 3 generator models.
6. **BRSR-specific evaluation.**
7. **Hindi evaluation** — on translated subset.
8. **Cross-benchmark evaluation** — FinQA, TAT-QA, PHANTOM.

### Phase 4: Analysis & Ablation (Weeks 5–7)

1. **Ablation studies** (table format, context window, reasoning mode).
2. **Error analysis** (200 cases, taxonomy, quantitative + qualitative).
3. **Sector-wise and section-wise analysis.**
4. **LLM-judge calibration** against human labels.

### Phase 5: Paper & Release (Weeks 7–9)

1. Generate all tables and figures.
2. Write paper sections.
3. Write Limitations and Ethics sections.
4. Prepare HuggingFace dataset release.
5. Prepare GitHub repo with evaluation code, prompts, model configs.
6. Set up leaderboard (optional but recommended).

---

## 8. NEW: Limitations Section (Required by EMNLP)

The paper must include a "Limitations" section. Plan to address:

1. **Single fiscal year (FY2025):** Temporal generalization is not tested; future work should add multi-year data.
2. **LLM-generated QA:** Despite human validation of a subset, the majority of QA pairs are model-generated; some noise is inevitable.
3. **One company per sector:** Limits within-sector variance analysis.
4. **Hindi translation quality:** Translated subset may have artifacts; professional translation covers only 500 pairs.
5. **No vision/multimodal evaluation:** Annual reports contain charts and images; our text-only extraction omits these.
6. **Indian regulatory context:** Findings may not generalize to US 10-K or other regulatory frameworks.
7. **Retriever-specific results:** RAG results depend on specific retriever choices; other retrievers may yield different outcomes.

---

## 9. NEW: Ethics Statement (Required by EMNLP)

1. **Data licensing:** All annual reports are publicly available from BSE/NSE; no proprietary data.
2. **PII:** Company names and financial data are public; no individual PII. Verify and strip any accidentally included personal information.
3. **Financial advice disclaimer:** The benchmark evaluates information extraction and QA, NOT financial advice. Include prominent disclaimer.
4. **Dual use:** The forensic QA track could be used for both governance improvement and adversarial purposes; we frame it positively.
5. **Carbon footprint:** Report estimated GPU hours and carbon emissions for all model evaluations.
6. **Hindi translation consent:** Ensure translators are fairly compensated.

---

## 10. NEW: Release Plan & Leaderboard

### 10.1 HuggingFace Dataset

- Release the **test set** (QA pairs + evidence + metadata) publicly.
- Release the **dev set** publicly.
- Keep train set optional (for fine-tuning experiments).
- Include both English and Hindi subsets.
- Include the human-validated subset with annotator agreement scores.
- Data card with: dataset summary, supported tasks, languages, stats, annotation details, licensing.

### 10.2 GitHub Repository

- Evaluation scripts for all metrics.
- Model inference wrappers (NIM API + vLLM).
- Prompt templates used in all experiments.
- Raw model outputs for reproducibility.
- RAG pipeline implementations.
- Analysis and visualization notebooks.

### 10.3 Leaderboard (Recommended)

- Set up a simple leaderboard (can be HuggingFace Spaces + Gradio).
- Categories: FinBharat-QA, FinBharat-Hall, FinBharat-RAG, FinBharat-ESG.
- Allow submission of model outputs for evaluation on the hidden test set.
- This is increasingly expected — FinBen, MULTIFINBEN, FABRIC all have leaderboards.

---

## 11. Timeline (Adjusted for ARR May 2026 Cycle)

| Week | Dates | Milestone |
|------|-------|-----------|
| 1 | May 25–31 | Data consolidation, split creation, human validation setup |
| 2 | Jun 1–7 | Metric implementation begins, start human annotation |
| 3 | Jun 8–14 | Complete human validation, evidence audit, BRSR subset extraction |
| 4 | Jun 15–21 | Core QA evaluation (all models), metric implementation complete |
| 5 | Jun 22–28 | Faithfulness evaluation, closed-book + few-shot experiments |
| 6 | Jun 29–Jul 5 | RAG pipeline evaluation, Hindi evaluation |
| 7 | Jul 6–12 | Cross-benchmark evaluation, ablation studies |
| 8 | Jul 13–19 | Error analysis, sector analysis, judge calibration |
| 9 | Jul 20–26 | Paper writing, figure/table generation |
| 10 | Jul 27–Aug 2 | Paper revision, release preparation, submission to ARR |

---

## 12. Summary: What Was Changed and Why

| Change | Why |
|--------|-----|
| Renamed to FinBharat | Signals India-focus; aligns with naming conventions |
| Added human validation (500 QA pairs, κ scores) | Biggest gap — reviewers will reject LLM-only QA without validation |
| Added BRSR/ESG track | Unique differentiator — no other benchmark covers Indian BRSR |
| Added RAG evaluation track | Every 2025–26 benchmark has this; omission would look outdated |
| Added bilingual (Hindi) evaluation | FABRIC showed this is compelling; EMNLP values multilinguality |
| Added cross-benchmark comparison | Reviewers will ask "how does this compare?" |
| Added statistical significance testing | Standard expectation at top venues |
| Added tolerance-based numeric accuracy | FinLongDocQA set this standard |
| Added table format ablation | Indian reports have unique table formats; this is a novel finding |
| Added concrete model names | Original had placeholders; needed for feasibility |
| Added error taxonomy with Indian-specific categories | Unit/scale mismatches, BRSR-specific errors |
| Added Limitations section | EMNLP desk-rejects without it |
| Added Ethics statement | Required by ARR checklist |
| Added HuggingFace + leaderboard release | Expected for benchmark papers in 2026 |
| Added attribution/citation metrics | ARQA and FINLFQA demonstrated importance |
| Added PoT reasoning ablation | FinAgent-RAG showed significant gains |
| Revised paper structure | Better page allocation, cross-benchmark table |

---

*This updated plan is designed to make FinBharat a strong, competitive submission to EMNLP 2026 that addresses every concern that reviewers raised about recent financial NLP benchmarks, while leveraging the unique strengths of the Indian annual report dataset.*
