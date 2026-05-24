# FinAR-Hall Benchmark Plan

This document lays out the detailed plan for turning our existing 189-company QA corpus into a publishable benchmark and EMNLP-level paper focused on **financial QA, hallucination, and faithfulness evaluation over annual reports**.

We assume we already have, per company and FY:
- `chunks*.jsonl`: ground-truth text chunks (tables + prose) with section and page metadata.
- `chunk_scoring_details*.jsonl`: per-chunk scores and detected financial keywords.
- `semantic_bundles_details*.jsonl`: bundle IDs, scores, and section summaries.
- `test_easy_qa.jsonl`, `test_medium_qa*.jsonl`, `hard_qa_pairs*.jsonl`, `multihop_qa_pairs*.jsonl`: generated QAs for four difficulty tiers.
- `question_count*.log`, `hallucination_count*.log`: per-report stats.

All of the plan below assumes these artifacts exist for **189 companies** across multiple sectors.

---

## 1. High-Level Goal

Create **FinAR-Hall** (name placeholder): a benchmark and paper that:

1. Evaluates **financial QA accuracy** over real annual reports, across four difficulty tiers:
   - Easy, Medium, Hard (forensic), Multihop.

2. Evaluates **hallucination and faithfulness**:
   - Are model answers *actually* supported by annual-report text and tables?
   - Do models fabricate numbers, misattribute values, or over-interpret ESG/governance content?

3. Provides **task-specific, deterministic metrics** for numeric correctness plus semantic and NLI-based faithfulness metrics.

4. Offers **stratified analyses** by:
   - Difficulty tier.
   - Question type: Text Only, Table Only, Table with Text, Numerical Calculation.
   - Document section type (financial statements, notes, ESG/BRSR, governance, risk).

5. Evaluates a **panel of open models via NVIDIA NIM**, plus ideally one strong closed model, under multiple conditions (open/closed-book, zero/few-shot).

6. Is **fully reproducible and open-sourced** (dataset subset + all evaluation scripts + prompts + raw model outputs).

---

## 2. Dataset Structure and Splits

### 2.1 Existing Data per Company

For each of the 189 companies (per FY):

- `chunks-*.jsonl`:
  - Fields: `chunkid`, `docid`, `sector`, `company`, `year`, `section`, `pagenumbers`, `hastable`, `tokencount`, `oversizedtable`, `score`, `text`.
 
- `chunk_scoring_details-*.jsonl`:
  - Fields: `chunk_id`, `section`, `total_score`, optional `table_bonus`, `found_keywords`, optional `found_high_priority_sections`.

- `semantic_bundles_details-*.jsonl`:
  - Fields: `bundle_id` (range of chunk IDs), `score`, `chunk_count`, `section` summary.

- QA files:
  - `test_easy_qa.jsonl`
  - `test_medium_qa*.jsonl`
  - `hard_qa_pairs*.jsonl`
  - `multihop_qa_pairs*.jsonl`
  - Each QA record: `question`, `answer`, `evidence`, `question_type`, `requires_calculation`, `sector`, `company`, `year`, `section`, `page_numbers`, `bundle_id`, `difficulty`, plus `verification_anchors` for hard/multihop.

- Logs:
  - `question_count*.log`: per-report total questions and difficulty/type breakdown.
  - `hallucination_count*.log`: number of generation-time JSON hallucinations.

### 2.2 Global Dataset Split

We will create **report-level splits** to avoid leakage:

- **Train** (for potential fine-tuning, optional): ~60% reports.
- **Dev**: ~20% reports.
- **Test**: ~20% reports.

For the benchmark paper, we primarily focus on **Dev+Test** as evaluation sets. Train can be used for experiments involving fine-tuning or in-context selection.

Within each split, we will ensure:
- Coverage across sectors.
- Coverage across difficulty tiers.
- Coverage across question types.

---

## 3. Tasks and Benchmarks

We define multiple benchmark tracks built from the same corpus.

### 3.1 Task 1: Core QA Accuracy (FinAR-QA)

**Task definition:**

- Input: `(question, context)` where `context` is either:
  - Gold bundle text (concatenation of chunks for `bundle_id`), or
  - Retrieved top-k chunks from `chunks*.jsonl` (for retrieval experiments).
- Output: free-form answer.

**Variants:**

- Difficulty tiers: Easy, Medium, Hard, Multihop.
- Question types: Text Only, Table Only, Table with Text, Numerical Calculation.

**Core metrics:**

1. **Exact Match (EM)** and **token-level F1**:
   - For short, factoid-style answers (numbers, names, dates).

2. **Num-Exact & Num-F1** (numeric specific):
   - Extract numeric values from gold and predicted answers.
   - `Num-Exact`: 1 if final numeric value (after normalisation) matches exactly.
   - `Num-F1`: F1 over numeric tokens.

3. **Boolean / Directional accuracy**:
   - For questions whose gold answers can be reduced to boolean/directional labels (increase/decrease, yes/no, above/below threshold).

4. **BERTScore (P/R/F1) and ROUGE-L**:
   - For textual answers (Text Only, some Table with Text).
   - ROUGE-L included for compatibility; BERTScore is main semantic overlap metric.

**Stratifications:**

- By difficulty tier.
- By question type.
- By sector.
- By document section type.


### 3.2 Task 2: Faithfulness & Hallucination (FinAR-Hall)

**Task definition:**

Given `(document context, question, answer)`, assess whether the answer is fully supported by the context (no hallucinated facts).

We use two complementary evaluation styles:

1. **Score answers given by QA models** (primary use).
2. **Build a labeled hallucination detection dataset** by including both correct and hallucinated answers.

**Metrics:**

1. **LLM-judge faithfulness (Tier-3 metric):**
   - A separate judge model sees:
     - Ground truth text (bundle or extended window extracted from `chunks*.jsonl`).
     - Question.
     - Answer.
   - Outputs:
     - Faithfulness score in [0, 1].
     - Flags: `CHERRYPICK`, `ARITHMETIC_ERROR`, `OVERINFERENCE`.
   - Report mean and distribution per difficulty × question_type.

2. **NLI-based entailment from evidence:**
   - Premise: evidence context.
   - Hypothesis: each sentence from the answer.
   - Use open-source cross-encoder NLI (e.g., DeBERTa-MNLI) to compute:
     - Fraction of sentences classified as ENTAILMENT.
     - Fraction as CONTRADICTION or NEUTRAL.

3. **Binary/multi-class hallucination detection:**
   - On a labeled subset (with human labels):
     - Accuracy, precision, recall, F1 for hallucinated vs faithful answers.
     - Multi-class confusion between error types (numeric, over-interpretation, etc.).


### 3.3 Task 3: Evidence Quality & Retrieval

**A. Evidence traceability & completeness (intrinsic dataset quality)**

- Evidence traceability:
  - Score 1.0 if evidence string is an exact/near substring of the concatenated ground truth chunks for `bundle_id`.
  - Use 0.8+ threshold for minor whitespace/format differences.

- Evidence completeness (numeric):
  - For each QA, extract all numbers from the answer.
  - Measure what fraction appear in the evidence string.

**B. Evidence retrieval (model task)**

- Given `(question, full document chunks)`, rank chunks.
- Gold chunks: minimal supporting set (derived by matching answer/evidence numbers/text in `chunks*.jsonl`).

**Metrics:**

- Retrieval metrics:
  - Recall@k, MRR@k, R-Precision at chunk level.


### 3.4 Task 4: Numeric Reasoning & Table QA (FinAR-Numeric)

**Task definition:**

Focus on questions where numerical reasoning over tables is central:

- `question_type ∈ {Table Only, Numerical Calculation, Table with Text}`.
- Many multihop questions will also fall here.

**Metrics:**

- Num-Exact, Num-F1, directional accuracy, MAPE.
- Operation-type breakdown (difference, ratio, YoY growth, sum/aggregation, multi-table join).


### 3.5 Task 5: Multi-hop Reasoning (FinAR-MultiHop)

**Task definition:**

- Use `multihop_qa_pairs*.jsonl` (and multihop-like questions from other tiers).
- These require chaining across multiple rows, multiple tables, or multiple sections.

**Metrics:**

- EM, Num-Exact/Num-F1 as in core QA.
- Support coverage:
  - For each multihop QA, define a set of necessary supporting facts (manually for a subset, heuristic for the rest).
  - For models that output evidence, measure recall of these supports.

- Performance vs hop count (2-hop, 3-hop, 4+), estimated per question.


### 3.6 Task 6: Hard Forensic QA (FinAR-Hard)

**Task definition:**

- Use `hard_qa_pairs*.jsonl` where questions come from a forensic auditor/analyst persona.
- Focus on red flags, earnings quality, governance, ESG risk.

**Metrics:**

- Textual correctness: BERTScore, EM/F1 when answers have clear labels.
- Issue detection:
  - On a labeled subset, mark whether the gold answer calls out specific issues.
  - Measure whether the model identifies the same issue category.
- Argumentation quality:
  - Judge/human rating (e.g., 1–5) for reasoning clarity and evidence use.


---

## 4. Metrics: Detailed Definitions

### 4.1 Deterministic QA Metrics

1. **EM / F1 (text tokens)**
   - Normalize case, whitespace, punctuation.
   - Tokenize by whitespace; compute EM and standard F1.

2. **Num-Exact**
   - Extract all numeric tokens from gold and prediction (regex on [0-9.,%₹$]+).
   - Normalize units where needed (e.g., Crores vs absolute, Lakh vs absolute) using heuristics based on the question and evidence.
   - Identify the main target number (heuristics: last numeric token, or use gold answer metadata) and check exact equality after normalization.

3. **Num-F1**
   - Treat each distinct numeric literal as a token.
   - Compute F1 over the sets of numbers in gold vs prediction.

4. **Boolean / directional accuracy**
   - Map gold answers to a label space: {Yes, No} or {Up, Down, No-change}, or threshold categories.
   - Build a simple classifier over model answers (regex / heuristic) to extract predicted label.
   - Accuracy = proportion of matches.

### 4.2 Overlap & Semantic Metrics

1. **BERTScore (P/R/F1)**
   - Use a strong sentence embedding model (e.g., DeBERTa or RoBERTa-based) for contextual embedding.
   - Compute precision, recall, F1 between gold and prediction.

2. **ROUGE-L**
   - Use for compatibility only, mainly in appendix.

### 4.3 Faithfulness & Hallucination Metrics

1. **LLM-judge faithfulness**
   - Prompt: as defined in the benchmarking report.
   - Output: score in [0,1], flags.

2. **NLI-based entailment**
   - For each answer sentence:
     - Inputs: `(premise = evidence context, hypothesis = sentence)`.
     - Use cross-encoder NLI to get probabilities.
   - Entailment ratio: fraction of sentences where ENTAILMENT is highest.
   - Contradiction ratio: fraction of CONTRADICTION.

3. **Binary hallucination classification (labeled subset)**
   - Label answers as {faithful, hallucinated}.
   - Evaluate detectors (NLI decisions, LLM-judge, simple overlap rules).

### 4.4 Evidence Metrics

1. **Traceability score**
   - Compute normalized Levenshtein/substring match between evidence string and concatenated ground truth for `bundle_id`.
   - Score 1.0 for exact substring; lower for modifications.

2. **Evidence completeness (numeric)**
   - Ratio = (# answer numbers found in evidence) / (# numbers in answer).

### 4.5 Retrieval Metrics

- Recall@k, MRR@k, R-Precision at chunk level.


---

## 5. Experimental Design

### 5.1 Model Panel (via NVIDIA NIM)

Select a representative set of open-weight LLMs served through NIM:

- Small / medium models (7–14B):
  - General chat model (e.g., family A-7B-chat).
  - Reasoning/math/coder variant (e.g., family A-7B-math).

- Large models (30–70B):
  - General chat (e.g., family B-34B-chat).
  - Strong open model (e.g., family C-70B-chat).

- Optional baseline: smallest general model in catalog.

Optionally include **1 strong closed model** (via external API) on a subset for upper bound.

For each model, define evaluation regimes:

- Zero-shot open-book.
- Few-shot open-book (e.g., 5 in-context examples).
- Closed-book for sampling ablations.


### 5.2 Core Evaluation Runs

For each selected model:

1. **Core QA (Dev+Test sets)**
   - Condition: open-book with gold bundle context.
   - Generate answers for all QAs (or a large, stratified subset if needed for compute).
   - Compute: EM, F1, Num-Exact, Num-F1, Boolean accuracy, BERTScore.
   - Compute faithfulness: LLM-judge scores, NLI entailment.

2. **Modality analysis**
   - Aggregate metrics by question_type.

3. **Difficulty analysis**
   - Aggregate metrics by difficulty tier.

4. **Section analysis**
   - Aggregate metrics by high-level section categories (financial statements, notes, BRSR/ESG, governance, risk).


### 5.3 Ablation Studies

Run ablations on a stratified subset (e.g., 3–5k QAs) for 2–3 models:

1. **Open-book vs Closed-book**
   - Closed-book: model sees question only.
   - Compare Num-Exact, Num-F1, BERTScore, faithfulness.

2. **Zero-shot vs Few-shot**
   - Zero-shot: single instruction.
   - Few-shot: 3–5 examples per difficulty/question_type.

3. **Retrieval variants**
   - Gold bundle vs top-k retrieved chunks.
   - Show impact of retrieval quality.


### 5.4 Error Analysis

Select one strong open-weight model (e.g., 70B) for deep error analysis:

1. Collect ~100 error cases:
   - 25 easy, 25 medium, 25 hard, 25 multihop.
   - Sample across question types.

2. Manually categorize errors into taxonomy:
   - Calculation Error.
   - Misattribution (wrong row/year/segment).
   - Evidence Ignored / Contradiction.
   - Over-interpretation / Speculation.
   - Cherry-picking evidence.
   - Formatting/output issues.

3. Link error categories back to metrics:
   - Show which metrics flag which error types.


---

## 6. Paper Structure (Outline)

**1. Introduction**
- Motivation: financial QA on annual reports; hallucinations are dangerous.
- Gap: existing datasets (FinQA, FinTextQA, PHANTOM) do not provide an end-to-end, table-heavy, India-focused annual report benchmark with numeric + faithfulness metrics.
- Contributions (three bullets):
  1. FinAR-Hall benchmark over 189 annual reports, with 4 difficulty tiers and structured evidence links.
  2. Multi-layer metric suite for numeric correctness, evidence quality, and faithfulness.
  3. Extensive evaluation of open-weight models (via NIM) showing failure modes and error taxonomy.

**2. Dataset Construction**
- Recap OCR → chunking → scoring → bundling → QA generation pipeline.
- Statistics across 189 reports (questions per difficulty, per type, per sector).
- Evidence traceability and question diversity stats.

**3. Tasks and Metrics**
- Describe FinAR-QA, FinAR-Hall, FinAR-Numeric, FinAR-MultiHop, FinAR-Hard.
- Formal definitions of metrics.

**4. Experimental Setup**
- Model panel (NIM models + any closed model).
- Splits (Dev/Test).
- Prompting setups (zero vs few-shot, open vs closed-book).

**5. Results: QA Accuracy**
- Overall and per difficulty/type tables.
- Modality analysis.

**6. Results: Faithfulness & Hallucination**
- Judge and NLI metrics.
- Hallucination detection on labeled subset.

**7. Numeric & Multi-hop Reasoning**
- Operation-wise performance.
- Multi-hop vs single-hop performance.

**8. Error Analysis**
- Taxonomy and examples.
- How metrics correlate with error types.

**9. Related Work**
- FinQA, FinTextQA, FinMTEB, FinBen, PHANTOM, automatic QA generation.

**10. Conclusion & Release Plan**
- Summary of insights.
- Commitment to open-sourcing datasets, code, metrics.


---

## 7. Engineering Roadmap (Implementation Steps)

1. **Data Consolidation**
   - Merge all per-company QA and chunk artifacts into a unified index.
   - Assign global IDs to QAs and chunks.

2. **Split Creation**
   - Implement script to create train/dev/test splits at report level.

3. **Metric Implementations**
   - Numeric parsing and Num-Exact/Num-F1.
   - Boolean/directional accuracy.
   - BERTScore and ROUGE.
   - Evidence traceability & completeness.
   - NLI-based entailment.
   - LLM-judge harness (prompt templates + batch caller).
   - Retrieval metrics.

4. **Model Wrappers (NIM)**
   - Generic evaluation harness for NIM models: given questions + contexts, get answers.
   - Caching to disk per (model, question_id, setting).

5. **Core QA Evaluation**
   - Run all selected models on Dev/Test (open-book gold bundle).
   - Compute all core metrics and faithfulness.

6. **Ablation Experiments**
   - Implement closed-book, retrieval-based contexts, zero/few-shot prompts.
   - Run on stratified subset.

7. **Error Analysis Tooling**
   - Script to select error cases and export them for manual annotation.

8. **Tables & Plots Generation**
   - Scripts to generate all result tables and key plots (difficulty curves, modality bars).

9. **Paper Drafting**
   - Fill sections according to outline, using generated tables/figures.

10. **Release Preparation**
   - Clean subset of data for public release.
   - GitHub repo with:
     - Data schema and examples.
     - Metric and evaluation code.
     - Model config files for NIM.
     - Prompt templates.

---

This plan is meant to be exhaustive enough that we can treat it as a project spec and track progress against each section and step.