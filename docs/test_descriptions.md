# FinBharat — Test Suite Reference

> **File:** `tests/test_metrics.py`  
> **Total tests:** 31 (all passing)  
> **Run:** `uv run pytest tests/ -v`  
> **Coverage:** `uv run pytest tests/ --cov=src/finbharat --cov-report=term-missing`

---

## Why these tests exist

Every metric in this paper is computed over 84,000+ model-generated answers.
A single bug in one metric function silently corrupts every result table in
the paper. These tests form a **contract**: if any test fails after a code
change, the evaluation pipeline is broken and results are untrustworthy.

The tests are structured in three layers:

| Layer | Question answered |
|-------|-----------------|
| **Layer 1 — Data Integrity** | Can we load the dataset from disk correctly? |
| **Layer 2 — Metric Correctness** | Does each metric return the right number for known inputs? |
| **Layer 3 — System Invariants** | Are architectural properties always maintained? (e.g. Relaxed EM ≥ EM, CI contains mean, no data leakage) |

---

## Layer 1 — Data Loading & Integrity

### `test_load_easy_qa`
**Purpose:** Verify the easy-tier QA loader reads JSONL files correctly and returns well-formed records.  
**Procedure:** Load all easy QA pairs for HDFC_Bank. Assert at least one record exists, difficulty = "easy", and question_type is one of the four valid types.  
**Why this input:** HDFC_Bank is a large bank with ~133 easy QA pairs — guaranteed non-empty, representative of the most common sector (banking). The question_type check validates the JSONL schema matches what downstream metric routing expects.  
**If this fails:** Every downstream test is meaningless. The data loader is broken.

---

### `test_load_sample`
**Purpose:** Verify the convenience sampler loads the exact expected count of records across multiple companies.  
**Procedure:** Call `load_sample()` with default SAMPLE_COMPANIES (HDFC_Bank + Bosch + Interglobe_Aviat). Assert total = 383 (133 + 132 + 118) and exactly those three companies appear.  
**Why this input:** These three companies are the default for all CLI `evaluate` runs without `--all-companies`. The count 383 is a known regression fixture — any change to the underlying JSONL files will break it immediately.  
**If this fails:** The default evaluation pipeline is loading wrong data. All quick-iteration results are based on incorrect samples.

---

### `test_load_chunks`
**Purpose:** Verify chunk records load correctly with proper IDs for bundle text construction.  
**Procedure:** Load all chunks for HDFC_Bank. Assert count = 830 and first chunk ID starts with `Private_Sector_Bank/HDFC_Bank`.  
**Why this matters:** Bundle text (the context passed to models) is built by iterating chunks using their IDs as keys. Malformed IDs cause `build_context_for_qa()` to silently return empty strings — models would answer without any context, making all results meaningless.  
**If this fails:** All model outputs in the evaluation are generated with empty context strings.

---

### `test_chunk_cache`
**Purpose:** Verify the in-memory chunk cache returns the same Python object on repeated calls (not a re-read copy).  
**Procedure:** Call `load_chunks()` twice for HDFC_Bank. Use Python's `is` operator (identity, not equality) to assert it's the same object.  
**Why identity not equality:** We need the cache to actually prevent disk I/O, not just return an equivalent list. `is` confirms only one read happened.  
**Scale impact:** Without caching: 187 companies × ~400 QA pairs = 74,800 JSONL reads per full run. With cache: 187 reads. The test catches cache regressions before they make full runs 400x slower.

---

## Layer 2A — Numeric Metrics

### `test_numeric_metrics_basic`
**Purpose:** Verify Num-Exact = 1 and Num-F1 = 1.0 when gold and prediction are identical.  
**Procedure:** `"Rs 41,173 million"` vs `"Rs 41,173 million"`.  
**Why this input:** "Rs 41,173 million" is a real number from HDFC_Bank's balance sheet — it tests the extraction regex handles: `Rs` prefix, comma-formatted number (41,173), and scale unit (million) all in one string. Testing with plain "100" would not catch Indian format failures.  
**If this fails:** The number extraction regex is broken for the most common answer format in the dataset. Num-EM and Num-F1 will return 0 for most numeric answers.

---

### `test_numeric_metrics_mismatch`
**Purpose:** Verify Num-Exact = 0 when numbers differ.  
**Procedure:** Gold = `"Rs 41,173 million"`, Pred = `"Rs 42,000 million"`.  
**Why this matters:** Confirms Num-Exact has no hidden tolerance. Models often produce numbers close but not equal (rounding, scale errors). Num-Exact must be strict 0/1 — the tolerance is handled separately by Tol-1/5/10 metrics.  
**If this fails:** Num-Exact is applying fuzzy matching and inflating scores across the board.

---

### `test_numeric_metrics_partial`
**Purpose:** Verify Num-F1 returns 0 < score < 1 for a two-number answer where only one matches.  
**Procedure:** Gold = `"549,498 and 103,386"`, Pred = `"549,498 and 100,000"`.  
**Why two numbers:** Hard and multihop questions frequently contain multi-part answers (e.g., "revenue grew from ₹X to ₹Y"). Num-F1 treats each number as a token and computes set-based F1 — partial credit for getting some numbers right.  
**If this fails:** Num-F1 collapses to 0/1 binary and loses discriminative power for multi-value answers.

---

### `test_tolerance_accuracy`
**Purpose:** Verify Tol-5 accepts predictions within ±5% and rejects those outside.  
**Procedure:** Gold = 1000. Test 1050 (5% → pass), 1060 (6% → fail), 1020 (2% → pass).  
**Why these values:** The 1060 test specifically verifies the boundary is exclusive — `> 5%` fails, not `≥ 5%`. This distinction matters for edge cases at the exact boundary (e.g., a prediction that is exactly 5.00% off).  
**Paper relevance:** Indian annual reports use inconsistent units (same figure as "₹1,000 Crores" vs "₹10,000 Million"). A model that reasons correctly but converts units slightly wrong gets Tol-5 credit rather than zero.

---

### `test_normalize_number`
**Purpose:** Verify number normalization strips commas and returns Python floats; returns None for non-numerics.  
**Procedure:** `"41,173"` → 41173.0, `"10.83"` → 10.83, `"abc"` → None.  
**Why None (not 0.0):** Downstream code filters `None` values before comparison. If `normalize_number` returned `0.0` for `"abc"`, it would be indistinguishable from a genuine zero answer — corrupting Num-Exact for questions about zero-valued metrics.  
**If this fails:** All Indian comma-formatted numbers (which appear in almost every answer) will fail to normalize, breaking all numeric metrics.

---

### `test_extract_numbers`
**Purpose:** Verify the extraction regex handles multiple numbers with Indian currency symbols, commas, and mixed prefixes.  
**Procedure:** Input: `"Rs 41,173 million and ₹ 549,498"`. Assert both `"41,173"` and `"549,498"` are extracted.  
**Why both symbols:** Annual reports mix `₹` (Unicode Rupee sign) and `Rs` (ASCII prefix) sometimes within the same paragraph. If the regex only handles one, it will miss ~50% of numeric answers.  
**If this fails:** Num-F1 will under-count gold numbers in multi-value answers, causing artificially low scores.

---

## Layer 2B — Text Overlap Metrics

### `test_exact_match`
**Purpose:** Verify EM = 1 for identical strings after normalization; EM = 0 for different answers.  
**Procedure:** `"Thirteen (13) Directors"` (identical) → 1. `"Thirteen (13) Directors"` vs `"14 Directors"` → 0.  
**Why this string:** Tests normalization of: parentheses (stripped), mixed case (lowercased), and a digit in text. "14 Directors" vs "Thirteen (13) Directors" both have 13 directors semantically — but EM is strict and should return 0. Token F1 and METEOR handle the semantic credit.

---

### `test_relaxed_em`
**Purpose:** Verify Relaxed EM strips Indian financial units and currency symbols before comparing.  
**Procedure:** `"₹ 139.27 crores"` vs `"139.27"` → 1. `"Rs 41,173 million"` vs `"41173"` → 1.  
**Why this matters:** Models often produce the correct number but without the unit (or with a different unit). Plain EM would fail both pairs. Relaxed EM is designed specifically for Indian financial reports where unit notation is inconsistent.  
**If this fails:** Scores are artificially lower than they should be — models that get the right number are penalized just for formatting.

---

### `test_relaxed_em_gte_exact_match`
**Purpose:** Enforce the fundamental invariant that Relaxed EM ≥ EM always.  
**Procedure:** Test 5 (gold, pred) pairs. Assert `relaxed_em >= exact_match` for each.  
**Origin of this test:** Added after discovering a real bug — in an early pilot run, Relaxed EM = 0.0667 while EM = 0.60. This is logically impossible. The cause was that Relaxed EM used a different normalization path (stripped units but left periods like `"Yes."`), causing `float("Yes.")` to fail and return 0.  
**This is the most important invariant test in the suite.** Any future normalization divergence between EM and Relaxed EM will be caught immediately.

---

### `test_token_f1`
**Purpose:** Verify Token F1 assigns partial credit for a prediction that is a correct but incomplete subset of the gold answer.  
**Procedure:** Gold = `"Thirteen Directors on the Board"` (4 tokens). Pred = `"Thirteen Directors"` (2 tokens — complete subset).  
**Expected:** precision = 2/2 = 1.0, recall = 2/4 = 0.5, F1 ≈ 0.667. Assert F1 > 0.5 and ≤ 1.0.  
**Why this matters:** Token F1 is the primary metric for Text Only and Table+Text questions. Models often give shorter but correct answers. A subset-answer should score well, not zero.

---

### `test_directional_accuracy`
**Purpose:** Verify the directional label extractor maps direction words correctly and that mismatched directions score 0.  
**Procedure:** `"Revenue increased by 18%"` vs `"Revenue increased"` → 1. Same gold vs `"Revenue declined"` → 0.  
**Why this metric exists:** ~15-20% of medium and hard QA require directional reasoning (YoY growth comparisons). Many models get the direction right but the magnitude wrong. Directional accuracy rewards qualitative correctness even when the exact number is wrong.

---

## Layer 2C — Semantic Overlap Metrics

### `test_rouge_l_perfect`
**Purpose:** Verify ROUGE-L = 1.0 for identical strings.  
**Procedure:** Same string on both sides → assert == 1.0.  
**If this fails:** The ROUGE scorer is misconfigured (wrong stemmer, wrong tokenizer, or wrong metric key). All ROUGE scores in results would be wrong.

---

### `test_rouge_l_partial`
**Purpose:** Verify ROUGE-L correctly assigns partial credit for a partial-overlap paraphrase.  
**Procedure:** Gold = `"The revenue increased by 18 percent"`, Pred = `"Revenue increased"`. Assert 0 < score < 1.  
**Why this input:** "Revenue increased" is directionally correct but omits magnitude. ROUGE-L should assign partial credit (words "revenue" and "increased" appear in the LCS), not zero or perfect.

---

### `test_rouge_l_empty`
**Purpose:** Verify ROUGE-L returns 0.0 gracefully for empty gold/pred without crashing.  
**Procedure:** Gold = `""`, pred = `"anything"` → assert == 0.0.  
**Why this matters:** Empty predictions occur when API calls time out or models produce no output. The evaluation loop must not crash on these edge cases.

---

### `test_meteor_perfect`
**Purpose:** Verify METEOR returns ≥ 0.99 for identical strings.  
**Procedure:** Same string → assert >= 0.99.  
**Why not == 1.0:** NLTK's METEOR implementation applies a small fragmentation penalty (a quirk of the algorithm) even for identical strings, returning ~0.996. Using `>= 0.99` avoids a false failure while still confirming the scorer works.  
**Why METEOR:** Added per Abhay Shakya's recommendation — METEOR handles stemming and synonym matching better than ROUGE for short financial answers, and improves backward compatibility with older financial NLP literature.

---

### `test_meteor_partial`
**Purpose:** Verify METEOR assigns partial credit for a correct-but-incomplete prediction.  
**Procedure:** Gold = `"Revenue increased by eighteen percent"`, Pred = `"Revenue increased"`. Assert 0 < score < 1.  
**Why "eighteen percent" spelled out:** Tests METEOR's stemming — "eighteen" and "increased" are matched by word form. This also tests that numeric words are not handled differently from regular words.

---

### `test_meteor_empty`
**Purpose:** Verify METEOR returns 0.0 for empty strings without exceptions.  
**Procedure:** Gold = `""`, pred = `"anything"` → assert == 0.0.  
**Same rationale as `test_rouge_l_empty`.** NLTK tokenization can crash on empty strings if not guarded.

---

## Layer 2D — BERTScore

### `test_bertscore_prf`
**Purpose:** Verify the batched BERTScore function returns P, R, and F1 as separate lists of the correct length, with values in [0, 1], and near-1.0 F1 for identical strings.  
**Procedure:** Two pairs: (identical, paraphrase). Assert len(P) = len(R) = len(F) = 2, all values in [0,1], F1[0] ≥ 0.95.  
**Why P and R separately:** Reviewers expect BERTScore precision (how much of the prediction is gold-like) and recall (how much of gold is covered) to be reported separately. High P + low R means the model's answer is semantically similar to gold but misses content. High R + low P means it covers gold content but adds noise.  
**Why batched:** The evaluation runs all predictions in one batch call — 1 BERTScore call per model×difficulty×regime run, not per question. A per-call interface would be 80x slower on a full 187-company run.  
**BERTScore limitation:** Due to RoBERTa-large's 512-token limit, texts are truncated to 1024 chars. This is acceptable for short-to-medium answers but may underestimate similarity for very long hard/multihop answers.

---

## Layer 2E — NLI Faithfulness Metrics

### `test_nli_entailment_heuristic`
**Purpose:** Verify the NLI pipeline (DeBERTa or word-overlap fallback) correctly classifies a faithful paraphrase as ENTAILMENT.  
**Procedure:** Evidence = `"The Board consists of thirteen (13) Directors."`, Answer = `"The Board has thirteen Directors."`. Assert entailment_ratio > 0.  
**Why this pair:** "consists of" → "has" is a synonym substitution. The DeBERTa cross-encoder should recognize this as entailment. If it returns CONTRADICTION or NEUTRAL, the faithfulness metric will falsely penalize correct answers.  
**Note on model loading:** The first call downloads `cross-encoder/nli-deberta-v3-small` (~180MB). Subsequent calls use the cached model. If transformers is unavailable, the word-overlap heuristic runs instead.

---

### `test_evidence_traceability`
**Purpose:** Verify evidence traceability returns 1.0 for identical evidence and gold evidence strings.  
**Procedure:** Evidence = gold evidence = exact verbatim sentence from HDFC_Bank's balance sheet. Assert == 1.0.  
**What it measures:** Whether a model's cited/retrieved evidence matches the gold supporting text. Score degrades as word overlap decreases. Testing 1.0 confirms the normalization (whitespace collapse, lowercasing) doesn't corrupt perfectly matching strings.

---

## Layer 2F — Abstain Rate

### `test_abstain_detection`
**Purpose:** Verify the abstain detector correctly identifies model refusals and does not misclassify genuine answers.  
**Procedure:**  
  - True cases: 3 real phrases observed in pilot model outputs  
  - False cases: 3 genuine financial answers and an empty string  
**Why this was added:** Pilot runs revealed 6–20% abstain rate on hard/multihop questions. This is healthy model behaviour (acknowledging uncertainty > hallucinating). But the detector must not accidentally classify a short answer like "No" as an abstain.  
**The empty string case:** An empty string is not an abstain — it's a missing answer (API error). It should not be counted as "model correctly declined to answer."

---

## Layer 3A — Dataset Quality & Structure

### `test_verification_anchors_parsed`
**Purpose:** Verify hard-tier QA records correctly parse the `verification_anchors` field into a structured object with all required fields.  
**Procedure:** Load hard QA for HDFC_Bank. Find records with non-None anchors. Assert the first has `alignment_status`, `hop_count`, `is_red_flag`, `calculation_inputs`.  
**What verification_anchors contains:**  
  - `alignment_status`: "consistent" or "contradiction" (used for red-flag detection)  
  - `calculation_inputs`: gold numeric operands for verifying arithmetic  
  - `cross_section_sources`: which sections were needed (hop chain)  
  - `hop_count`: number of hops (derived from sources)  
**If this fails:** Red-flag detection and hop-count breakdown in aggregate results will silently produce all-zero values for hard and multihop tiers.

---

### `test_brsr_flag`
**Purpose:** Verify the BRSR/ESG section detector correctly classifies India's Business Responsibility framework sections.  
**Procedure:**  
  - True: "PRINCIPLE 6: Businesses should respect the environment" → True  
  - True: "Essential Indicators | Leadership Indicators" → True  
  - False: "Financial Statements" → False  
  - False: "Report on Corporate Governance" → False  
**Why this matters:** BRSR/ESG evaluation is the paper's unique differentiator — no other financial benchmark covers India's mandatory sustainability reporting. The keyword set was derived from the actual 748 unique BRSR section strings in the dataset. False positives pollute the ESG subset; false negatives undercount it. Both hurt the paper's ESG analysis section.

---

### `test_table_serialization`
**Purpose:** Verify HTML-to-Markdown and HTML-to-linearized table serializers produce well-formed output containing all original data.  
**Procedure:** Input: HTML table with 2 data rows (Revenue, PAT) and 3 columns (Item, FY2025, FY2024).  
  - Markdown: assert `"| Item |"` (pipe header) and `"45,320"` present  
  - Linearized: assert `"Revenue"` and `"FY2025: 45,320"` (key-value row) present  
**Why this test exists:** 70% of easy QA and 44% of hard QA are Table Only or Table+Text. The table format ablation (`--table-format html|markdown|linearized`) is a required section in the paper. If serializers produce malformed output, the ablation runs make the model look worse than it is — it's receiving unreadable input, not failing at reasoning.

---

## Layer 3B — Statistical Significance

### `test_bootstrap_ci`
**Purpose:** Verify bootstrap CI computation returns a valid interval with lower ≤ mean ≤ upper and correct sample count.  
**Procedure:** 10 scores between 0.6 and 1.0. Assert 0.6 ≤ lower ≤ mean ≤ upper ≤ 1.0, n = 10.  
**Why this invariant:** EMNLP 2026 reviewers expect 95% bootstrap CIs on all main result tables. If lower > mean or upper < mean, the CI is mathematically impossible and the implementation is broken. Every CI in the paper would be wrong.  
**n_bootstrap = 200 in tests:** The production code uses 1000 bootstrap iterations. Tests use 200 to run faster while still exercising the resampling logic.

---

### `test_paired_bootstrap`
**Purpose:** Verify the paired bootstrap test correctly identifies significant differences and correctly accepts the null hypothesis for equal systems.  
**Procedure:**  
  - A = [1.0]*20, B = [0.0]*20 → significant = True, delta = 1.0  
  - A = B = [0.5]*20 → significant = False  
**Why both cases:** The "clearly better" case verifies the test can detect real differences. The "identical" case is more important — it verifies we don't publish spurious "X significantly outperforms Y" claims when both systems score equally. False positives in significance testing are more damaging than false negatives for a benchmark paper.

---

## Layer 3C — Data Split Integrity

### `test_split_creation`
**Purpose:** Verify the 60/20/20 stratified split covers all 187 companies exactly once with zero leakage between splits.  
**Procedure:**  
  1. Total = 187 (all companies accounted for)  
  2. No `sector:company` key appears in more than one split (no leakage)  
  3. All three splits are non-empty  
**Why the composite key (`sector:company`):** Company names like "HDFC_Bank" are unique in our dataset, but using the composite key is defensive programming — if a future dataset version has two HDFC_Bank entries in different sectors, the leakage check would still catch it.  
**Critical importance:** A contaminated test set invalidates the entire paper. If a company appears in both train and test, a fine-tuned model could memorize specific annual report facts and score artificially high on "unseen" test data. This test is the last line of defense against this failure mode.  
**seed=42:** Makes the split deterministic and reproducible. Any team member running `create_splits(seed=42)` gets the exact same 38 test companies.

---

## Running the Tests

```bash
# Full suite with verbose output
uv run pytest tests/ -v

# Quick pass/fail check
uv run pytest tests/ -q

# Stop on first failure (useful during development)
uv run pytest tests/ -x

# Run a specific test by name
uv run pytest tests/ -k "test_relaxed_em"

# Run a specific layer
uv run pytest tests/ -k "load or chunk or sample"

# With coverage
uv run pytest tests/ --cov=src/finbharat --cov-report=term-missing
```

## Adding New Tests

When adding a new metric or data processing function:
1. Add a **basic** test (known input → expected output)
2. Add an **edge case** test (empty string, None, zero denominator, etc.)
3. If the new function must satisfy an invariant relative to an existing one (like Relaxed EM ≥ EM), add an **invariant** test
4. Document the **thought process** — why this specific input was chosen, what real failure mode it guards against
