"""
FinBharat — Unit Test Suite
============================
31 tests covering data loading, all evaluation metrics, statistical significance,
dataset quality checks, and infrastructure correctness.

Run with:  uv run pytest tests/ -v
Coverage:  uv run pytest tests/ --cov=src/finbharat --cov-report=term-missing

Thought process
---------------
Every metric in the paper must be verified to behave correctly on known inputs
before it can be trusted on 84K+ model outputs. These tests form a contract:
if any test fails after a code change, the evaluation pipeline is broken and
results are untrustworthy. The tests are written in three layers:

  Layer 1 — Data integrity:   Can we load the dataset correctly?
  Layer 2 — Metric correctness: Does each metric produce the right number?
  Layer 3 — System properties: Are architectural invariants maintained?
             (e.g. Relaxed EM >= EM always, CI contains mean, no leakage)
"""

import json
from pathlib import Path
from finbharat.data.loader import FinBharatDataset, SAMPLE_COMPANIES, parse_qa_record
from finbharat.metrics.numeric import (
    compute_numeric_metrics, compute_tolerance_accuracy,
    normalize_number, extract_numbers,
)
from finbharat.metrics.text import (
    compute_exact_match, compute_token_f1, compute_relaxed_em,
    compute_directional_accuracy, compute_rouge_l,
)
from finbharat.metrics.faithfulness import compute_nli_entailment, compute_evidence_traceability


# tests/ is one level under project root → dataset is at project_root/dataset
DATA_ROOT = Path(__file__).parent.parent / "dataset"


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — Data Loading & Integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_load_easy_qa():
    """
    Purpose:
        Verify that the easy-tier QA loader correctly reads JSONL files from disk
        and returns well-formed QARecord objects.

    Procedure:
        Load all easy QA pairs for HDFC_Bank (Private Sector Bank sector).
        Check that at least one record exists, the difficulty label is 'easy',
        and the question_type is one of the four valid types defined in the dataset.

    Thought process:
        Before running any evaluation, we must confirm the data loader works.
        HDFC_Bank is used as the canary — it is a large bank with ~133 easy
        QA pairs, guaranteed to be non-empty. The question_type check ensures
        the JSONL schema matches what the metric routing code expects.
        If this test fails, every downstream test is meaningless.
    """
    dataset = FinBharatDataset(DATA_ROOT)
    records = dataset.load_easy_qa("Private_Sector_Bank", "HDFC_Bank")
    assert len(records) > 0
    assert records[0].difficulty == "easy"
    assert records[0].question_type in (
        "Text Only", "Table Only", "Table with Text", "Numerical Calculation"
    )


def test_load_sample():
    """
    Purpose:
        Verify that the convenience sampler loads the correct number of records
        across multiple companies and assigns correct global IDs.

    Procedure:
        Call load_sample() with the default SAMPLE_COMPANIES list (HDFC_Bank,
        Bosch, Interglobe_Aviat). Assert the total count matches the known
        per-company counts (133 + 132 + 118 = 383) and that exactly the three
        expected companies appear.

    Thought process:
        The SAMPLE_COMPANIES list is the default used in all CLI evaluate runs.
        This test encodes the expected size as a regression guard — if the
        dataset is modified or a company's QA file changes, this test will
        immediately catch it. The count (383) is a known fixture, not an estimate.
    """
    dataset = FinBharatDataset(DATA_ROOT)
    records = dataset.load_sample(difficulty="easy")
    assert len(records) == 383
    companies = {r.company for r in records}
    assert companies == {"HDFC_Bank", "Bosch", "Interglobe_Aviat"}


def test_load_chunks():
    """
    Purpose:
        Verify that raw chunk records (the evidence backing QA pairs) load
        correctly and that chunk IDs follow the expected hierarchical format.

    Procedure:
        Load all chunks for HDFC_Bank. Assert the count is 830 (known fixture),
        and that the first chunk's ID starts with 'Private_Sector_Bank/HDFC_Bank',
        which is the sector/company prefix used in bundle ID resolution.

    Thought process:
        Bundle text is built by iterating chunks in order using chunk IDs
        as keys. If chunk IDs are malformed or the count is wrong, all context
        construction will silently fail (returning empty strings). This test
        catches disk/encoding issues and schema drift early.
    """
    dataset = FinBharatDataset(DATA_ROOT)
    chunks = dataset.load_chunks("Private_Sector_Bank", "HDFC_Bank")
    assert len(chunks) == 830
    assert chunks[0].chunk_id.startswith("Private_Sector_Bank/HDFC_Bank")


def test_chunk_cache():
    """
    Purpose:
        Verify that the in-memory chunk cache returns the same Python object
        on repeated calls, not a freshly re-read copy.

    Procedure:
        Call load_chunks() twice for the same company. Use 'is' (identity check,
        not equality) to confirm both calls return the identical list object.

    Thought process:
        Without caching, a full evaluation run (187 companies × 400 QA pairs)
        would re-read the chunks JSONL file for every QA record in that company —
        potentially 400 disk reads per company, 74,800 total. The cache reduces
        this to one read per company. This test ensures the cache is actually
        working (not just returning equal lists, but the exact same list).
    """
    dataset = FinBharatDataset(DATA_ROOT)
    c1 = dataset.load_chunks("Private_Sector_Bank", "HDFC_Bank")
    c2 = dataset.load_chunks("Private_Sector_Bank", "HDFC_Bank")
    assert c1 is c2  # identity check — same object, not a copy


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2A — Numeric Metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_numeric_metrics_basic():
    """
    Purpose:
        Verify Num-Exact and Num-F1 both return perfect scores when gold
        and prediction are identical numeric strings.

    Procedure:
        Pass "Rs 41,173 million" as both gold and prediction (a real answer
        format from HDFC_Bank's annual report). Check Num-Exact == 1 and
        Num-F1 == 1.0.

    Thought process:
        "Rs 41,173 million" is a realistic Indian financial report number —
        it contains a currency prefix, comma separators, and a scale unit.
        Using this instead of a plain "100" tests that the number extraction
        regex correctly handles the full Indian format. If this returns 0,
        the regex is broken for the most common answer type in the dataset.
    """
    result = compute_numeric_metrics("Rs 41,173 million", "Rs 41,173 million")
    assert result.num_exact == 1
    assert result.num_f1 == 1.0


def test_numeric_metrics_mismatch():
    """
    Purpose:
        Verify Num-Exact returns 0 when the predicted number differs from gold.

    Procedure:
        Gold = "Rs 41,173 million", Pred = "Rs 42,000 million". Both contain
        a single number; they differ. Assert Num-Exact == 0.

    Thought process:
        Models often produce numbers close to but not equal to the gold value
        (rounding, unit confusion). Num-Exact must be strictly 0/1 — not a
        fuzzy match. The tolerance-based metrics (Tol-1/5/10) handle near-misses.
        This test ensures Num-Exact has no hidden tolerance baked in.
    """
    result = compute_numeric_metrics("Rs 41,173 million", "Rs 42,000 million")
    assert result.num_exact == 0


def test_numeric_metrics_partial():
    """
    Purpose:
        Verify Num-F1 returns a value strictly between 0 and 1 when the
        prediction shares some but not all numbers with the gold answer.

    Procedure:
        Gold = "549,498 and 103,386" (two numbers). Pred = "549,498 and 100,000"
        (first number matches, second differs). Assert 0 < Num-F1 < 1.

    Thought process:
        Many hard and multihop questions have multi-part answers containing
        several numbers (e.g. YoY comparison: "revenue grew from ₹X to ₹Y").
        Num-F1 treats each distinct number as a token and computes set-based F1.
        This test verifies the partial-credit behaviour that distinguishes
        a near-correct from a completely wrong answer.
    """
    result = compute_numeric_metrics("549,498 and 103,386", "549,498 and 100,000")
    assert result.num_f1 > 0
    assert result.num_f1 < 1.0


def test_tolerance_accuracy():
    """
    Purpose:
        Verify Tol-5 accepts predictions within ±5% of gold and rejects those
        outside, with edge cases at exactly the boundary.

    Procedure:
        Gold = 1000.
        - Pred = 1050: 5% above → should pass (Tol-5 = 1)
        - Pred = 1060: 6% above → should fail (Tol-5 = 0)
        - Pred = 1020: 2% above → should pass (Tol-5 = 1)

    Thought process:
        Indian annual reports use inconsistent units — the same value may appear
        as "₹ 1,000 Crores" in one place and "₹ 10,000 Million" elsewhere.
        A model that correctly reasons but converts to a slightly different
        scale should still get partial credit. Tol-1/5/10 quantify this.
        The 1060 test verifies the boundary is exclusive (> 5% fails), not
        that we accidentally used ≤ instead of <.
    """
    assert compute_tolerance_accuracy("1000", "1050", tolerance_pct=5.0) == 1
    assert compute_tolerance_accuracy("1000", "1060", tolerance_pct=5.0) == 0
    assert compute_tolerance_accuracy("1000", "1020", tolerance_pct=5.0) == 1


def test_normalize_number():
    """
    Purpose:
        Verify the number normalizer correctly strips commas and returns
        Python floats, and returns None for non-numeric strings.

    Procedure:
        "41,173" → 41173.0 (comma stripped)
        "10.83"  → 10.83   (decimal preserved)
        "abc"    → None    (non-numeric)

    Thought process:
        Indian numbers always use commas (e.g. "1,23,456"). If normalize_number
        fails on commas, every numeric comparison will silently return 0.
        The None case is critical — downstream code filters None values before
        comparison, so returning 0.0 instead of None would corrupt Num-Exact.
    """
    assert normalize_number("41,173") == 41173.0
    assert normalize_number("10.83") == 10.83
    assert normalize_number("abc") is None


def test_extract_numbers():
    """
    Purpose:
        Verify the regex extracts all numeric tokens from a sentence containing
        Indian currency symbols, commas, and multiple numbers.

    Procedure:
        Input: "Rs 41,173 million and ₹ 549,498"
        Assert both "41,173" and "549,498" appear in the extracted list.

    Thought process:
        The extraction regex must handle: ₹ prefix, Rs prefix, comma-formatted
        numbers, and numbers adjacent to scale units. This string is a minimal
        real example from HDFC_Bank's report. If the regex misses ₹-prefixed
        numbers, the Num-F1 metric will under-count gold numbers and inflate scores.
    """
    nums = extract_numbers("Rs 41,173 million and ₹ 549,498")
    assert "41,173" in nums
    assert "549,498" in nums


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2B — Text Overlap Metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_match():
    """
    Purpose:
        Verify EM returns 1 for identical strings after normalization and 0
        for clearly different answers.

    Procedure:
        Gold = pred = "Thirteen (13) Directors" → EM = 1
        Gold = "Thirteen (13) Directors", pred = "14 Directors" → EM = 0

    Thought process:
        EM normalizes by lowercasing, stripping punctuation, and collapsing
        whitespace. "Thirteen (13) Directors" exercises parentheses (stripped),
        mixed case (lowered), and a number in text. The 0 case uses a different
        number format to verify we're not accidentally doing fuzzy matching.
    """
    assert compute_exact_match("Thirteen (13) Directors", "Thirteen (13) Directors") == 1
    assert compute_exact_match("Thirteen (13) Directors", "14 Directors") == 0


def test_relaxed_em():
    """
    Purpose:
        Verify Relaxed EM correctly strips Indian financial units and currency
        symbols before comparing, matching answers that differ only in unit notation.

    Procedure:
        "₹ 139.27 crores" vs "139.27"  → should match (₹ and 'crores' stripped)
        "Rs 41,173 million" vs "41173" → should match (Rs, 'million', commas stripped)

    Thought process:
        Models often omit or add units to numeric answers. "139.27" and
        "₹ 139.27 crores" refer to the same value. EM would fail both; Relaxed EM
        is designed to pass them. This captures genuine understanding that EM
        penalizes due to formatting differences specific to Indian financial reports.
    """
    assert compute_relaxed_em("₹ 139.27 crores", "139.27") == 1
    assert compute_relaxed_em("Rs 41,173 million", "41173") == 1


def test_relaxed_em_gte_exact_match():
    """
    Purpose:
        Enforce the fundamental invariant that Relaxed EM is always ≥ EM.
        Relaxed EM is a strictly looser metric by definition.

    Procedure:
        For five (gold, pred) pairs spanning text, boolean, numeric, and
        unit-stripped cases, compute both EM and Relaxed EM. Assert
        relaxed_em >= exact_match for every pair.

    Thought process:
        This test was added after discovering a bug where Relaxed EM returned
        0.0667 while EM was 0.60 on a real evaluation run — logically impossible.
        The cause: different normalization paths (Relaxed EM stripped units but
        left periods, causing float() to fail on "Yes."). The invariant test
        catches any future normalization divergence before it silently corrupts
        published results. It is one of the most important tests in the suite.
    """
    pairs = [
        ("Thirteen (13) Directors", "Thirteen (13) Directors"),
        ("Yes.", "Yes."),
        ("Revenue increased by 18%", "Revenue increased by 18%"),
        ("₹ 139.27 crores", "139.27"),
        ("Not available in context", "Not available in context"),
    ]
    for gold, pred in pairs:
        em = compute_exact_match(gold, pred)
        rem = compute_relaxed_em(gold, pred)
        assert rem >= em, (
            f"INVARIANT BROKEN: relaxed_em={rem} < exact_match={em} "
            f"for gold={gold!r} vs pred={pred!r}"
        )


def test_token_f1():
    """
    Purpose:
        Verify Token F1 correctly handles partial token overlap, returning a
        value between 0.5 and 1.0 when the prediction is a strict subset of gold.

    Procedure:
        Gold = "Thirteen Directors on the Board" (4 tokens)
        Pred = "Thirteen Directors" (2 tokens — a subset of gold)
        Both tokens in pred appear in gold, so precision = 1.0, recall = 0.5,
        F1 = 0.667. Assert F1 > 0.5 and F1 <= 1.0.

    Thought process:
        Token F1 is the primary metric for text-heavy answers (Text Only and
        Table with Text question types). Short predictions that are wholly
        correct but incomplete should score well, not zero. The subset case
        exercises precision vs recall tradeoff, the most common failure mode
        where models give truncated answers.
    """
    result = compute_token_f1("Thirteen Directors on the Board", "Thirteen Directors")
    assert result.f1 > 0.5
    assert result.f1 <= 1.0


def test_directional_accuracy():
    """
    Purpose:
        Verify the directional label extractor correctly maps 'increased' →
        'up' and 'declined' → 'down', and that mismatched directions score 0.

    Procedure:
        Gold = "Revenue increased by 18%", Pred = "Revenue increased" → 1
        Gold = "Revenue increased by 18%", Pred = "Revenue declined" → 0

    Thought process:
        ~15-20% of medium and hard QA pairs require directional reasoning
        (YoY growth, comparative performance). Many models give correct direction
        but wrong magnitude. Directional accuracy gives credit for the qualitative
        insight even when the exact number is wrong. This test verifies that
        word-level direction extraction ('increased' → 'up') handles realistic
        sentence structures from annual report language.
    """
    assert compute_directional_accuracy("Revenue increased by 18%", "Revenue increased") == 1
    assert compute_directional_accuracy("Revenue increased by 18%", "Revenue declined") == 0


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2C — Semantic Overlap Metrics (ROUGE-L, METEOR)
# ─────────────────────────────────────────────────────────────────────────────

def test_rouge_l_perfect():
    """
    Purpose:
        Verify ROUGE-L returns 1.0 for identical strings.

    Procedure:
        Gold = pred = "Thirteen Directors" → assert score == 1.0

    Thought process:
        ROUGE-L measures longest common subsequence overlap. Identical strings
        must score 1.0 — if they don't, the scorer is misconfigured (e.g., wrong
        stemming or tokenization). This is the minimal sanity check before
        trusting any ROUGE scores in the results.
    """
    assert compute_rouge_l("Thirteen Directors", "Thirteen Directors") == 1.0


def test_rouge_l_partial():
    """
    Purpose:
        Verify ROUGE-L returns a value strictly between 0 and 1 for a
        paraphrase that shares some but not all words.

    Procedure:
        Gold = "The revenue increased by 18 percent"
        Pred = "Revenue increased"
        The words 'revenue' and 'increased' are shared; the rest are not.
        Assert 0.0 < score < 1.0.

    Thought process:
        ROUGE-L must be sensitive to both correct and incorrect content.
        "Revenue increased" correctly captures the direction but omits the
        magnitude. The test ensures partial credit is awarded (not zero) and
        that perfect credit is not given (not 1.0), verifying the metric
        discriminates between complete and partial answers.
    """
    score = compute_rouge_l("The revenue increased by 18 percent", "Revenue increased")
    assert 0.0 < score < 1.0


def test_rouge_l_empty():
    """
    Purpose:
        Verify ROUGE-L returns 0.0 when the gold or prediction is empty,
        without raising an exception.

    Procedure:
        Gold = "" (empty string), pred = "anything" → assert score == 0.0

    Thought process:
        Empty predictions occur when a model returns no output (e.g., API
        timeout, context overflow). ROUGE-L must handle this gracefully without
        crashing the evaluation loop. A 0.0 score for an empty prediction is
        correct — it contributes nothing useful and should not inflate averages.
    """
    assert compute_rouge_l("", "anything") == 0.0


def test_meteor_perfect():
    """
    Purpose:
        Verify METEOR returns a near-perfect score (≥ 0.99) for identical strings.

    Procedure:
        Gold = pred = "Thirteen Directors on the Board"
        Assert score >= 0.99.

    Thought process:
        Unlike ROUGE-L, NLTK's METEOR applies a small fragmentation penalty
        even for perfect matches — a known quirk of the implementation. Using
        >= 0.99 instead of == 1.0 avoids a false failure while still verifying
        the scorer is working. METEOR is included alongside ROUGE-L per
        Abhay's recommendation for backward compatibility with older literature
        and because it handles stemming and synonyms better for short answers.
    """
    from finbharat.metrics.text import compute_meteor
    score = compute_meteor("Thirteen Directors on the Board", "Thirteen Directors on the Board")
    assert score >= 0.99  # NLTK METEOR applies a small fragmentation penalty even on perfect match


def test_meteor_partial():
    """
    Purpose:
        Verify METEOR correctly scores a partial match between a long gold
        answer and a short correct-but-incomplete prediction.

    Procedure:
        Gold = "Revenue increased by eighteen percent"
        Pred = "Revenue increased"
        Assert 0.0 < score < 1.0.

    Thought process:
        METEOR uses stemming and synonym matching. "increased" and "increased"
        match directly. "eighteen percent" is missing from pred. The test
        verifies METEOR assigns partial credit proportional to the overlap,
        making it more informative than EM for open-ended financial answers.
    """
    from finbharat.metrics.text import compute_meteor
    score = compute_meteor("Revenue increased by eighteen percent", "Revenue increased")
    assert 0.0 < score < 1.0


def test_meteor_empty():
    """
    Purpose:
        Verify METEOR returns 0.0 for empty gold/pred without raising exceptions.

    Procedure:
        Gold = "", pred = "anything" → assert score == 0.0

    Thought process:
        Same robustness requirement as test_rouge_l_empty. METEOR uses NLTK
        tokenization which can fail on empty strings if not guarded. The
        function should absorb these edge cases and return 0.0 cleanly.
    """
    from finbharat.metrics.text import compute_meteor
    assert compute_meteor("", "anything") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2D — BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def test_bertscore_prf():
    """
    Purpose:
        Verify the batched BERTScore function returns Precision, Recall, and F1
        as separate lists of correct length, all in [0, 1], with near-1.0 F1
        for an identical pair.

    Procedure:
        Two (gold, pred) pairs:
          1. "Thirteen Directors" vs "Thirteen Directors" (identical)
          2. "Revenue increased by 18 percent" vs "Revenue grew by 18%" (paraphrase)
        Assert len(P) == len(R) == len(F) == 2.
        Assert all values in [0, 1].
        Assert F1 for the identical pair >= 0.95 (BERTScore for identical text is
        not always exactly 1.0 due to subword tokenization).

    Thought process:
        BERTScore uses contextual embeddings from RoBERTa-large. Unlike ROUGE-L,
        it can match "grew" to "increased" semantically. We added P and R
        separately (not just F1) because reviewers expect to see precision vs
        recall tradeoff in semantic faithfulness analysis — high BERTScore-P
        means the predicted answer is semantically similar to gold; high
        BERTScore-R means all gold content is covered. The batch API is tested
        (not per-call) because the evaluation runs all predictions in one batch
        call for efficiency; a per-call interface would be 80x slower.
    """
    from finbharat.eval.evaluate import compute_bertscore_batch
    golds = ["Thirteen Directors", "Revenue increased by 18 percent"]
    preds = ["Thirteen Directors", "Revenue grew by 18%"]
    P, R, F = compute_bertscore_batch(golds, preds)
    assert len(P) == len(R) == len(F) == 2
    assert all(0.0 <= v <= 1.0 for v in P + R + F)
    assert F[0] >= 0.95  # identical strings should be near-perfect


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2E — NLI Faithfulness Metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_nli_entailment_heuristic():
    """
    Purpose:
        Verify the NLI pipeline (or its word-overlap heuristic fallback) correctly
        classifies a faithful paraphrase as ENTAILMENT, producing entailment_ratio > 0.

    Procedure:
        Evidence = "The Board consists of thirteen (13) Directors."
        Answer   = "The Board has thirteen Directors."
        Assert entailment_ratio > 0.

    Thought process:
        The real DeBERTa NLI model is loaded lazily; the first call downloads
        ~180MB. This test verifies the full NLI path end-to-end (or the heuristic
        fallback if transformers is unavailable). The evidence/answer pair is a
        faithful paraphrase — "consists of" → "has" is a synonym-level change.
        If entailment_ratio == 0.0, the NLI pipeline is broken and all faithfulness
        scores in the paper would be unreliable.
    """
    evidence = "The Board consists of thirteen (13) Directors."
    answer = "The Board has thirteen Directors."
    result = compute_nli_entailment(evidence, answer)
    assert result.entailment_ratio > 0


def test_evidence_traceability():
    """
    Purpose:
        Verify evidence traceability returns 1.0 when the evidence exactly
        matches the gold evidence (perfect source attribution).

    Procedure:
        evidence = gold_evidence = "As on March 31, 2025, the Balance Sheet size
        was US $ 10.83 billion." (verbatim string)
        Assert traceability == 1.0.

    Thought process:
        Evidence traceability measures whether a model's retrieved/cited evidence
        matches the gold supporting text. When they are identical, the model
        perfectly attributed its answer to the source. The score degrades as
        the overlap decreases. Testing the 1.0 case confirms the word-overlap
        logic is not off-by-one or doing wrong string normalization.
    """
    evidence = "As on March 31, 2025, the Balance Sheet size was US $ 10.83 billion."
    gold = "As on March 31, 2025, the Balance Sheet size was US $ 10.83 billion."
    assert compute_evidence_traceability(evidence, gold) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2F — Abstain Rate
# ─────────────────────────────────────────────────────────────────────────────

def test_abstain_detection():
    """
    Purpose:
        Verify the abstain detector correctly identifies predictions where the
        model declined to answer, and does not misclassify genuine answers.

    Procedure:
        Positive cases (model abstained):
          - "Not available in context"          (exact phrase from system prompt)
          - "The answer is not provided in the document"
          - "Cannot be determined from the given context"
        Negative cases (model answered):
          - "Thirteen (13) Directors"   (factual answer)
          - "₹ 41,173 crores"           (numeric answer)
          - ""                           (empty string — not an abstain, just missing)

    Thought process:
        Abstain rate is a new metric introduced because our pilot revealed that
        models abstain on 6–20% of hard/multihop questions. This is actually
        healthy behaviour — better than hallucinating. But if the detector
        incorrectly flags genuine answers as abstains, the metric becomes
        misleading. The test uses real phrases from the models' actual outputs
        (observed in pilot runs) to prevent false positives.
    """
    from finbharat.eval.evaluate import _is_abstain
    assert _is_abstain("Not available in context") is True
    assert _is_abstain("The answer is not provided in the document") is True
    assert _is_abstain("Cannot be determined from the given context") is True
    assert _is_abstain("Thirteen (13) Directors") is False
    assert _is_abstain("₹ 41,173 crores") is False
    assert _is_abstain("") is False


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3A — Dataset Quality & Structural Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_verification_anchors_parsed():
    """
    Purpose:
        Verify that hard-tier QA records correctly parse the verification_anchors
        field from JSONL, producing a structured VerificationAnchors object with
        all required fields.

    Procedure:
        Load hard QA records for HDFC_Bank. Find records with non-None
        verification_anchors. Assert the first such record has the four
        expected fields: alignment_status, hop_count, is_red_flag, calculation_inputs.

    Thought process:
        verification_anchors is the gold evidence chain available on hard and
        multihop QA pairs. It contains:
          - alignment_status: "consistent" or "contradiction" (red-flag signal)
          - calculation_inputs: gold numeric operands for arithmetic verification
          - cross_section_sources: the hop chain used to generate the question
          - hop_count: derived from len(cross_section_sources)
        If this test fails, the red-flag detection metric and hop-count breakdown
        in aggregate_results will silently produce zeros for hard/multihop tiers.
    """
    dataset = FinBharatDataset(DATA_ROOT)
    records = dataset.load_hard_qa("Private_Sector_Bank", "HDFC_Bank")
    assert len(records) > 0
    has_va = [r for r in records if r.verification_anchors is not None]
    assert len(has_va) > 0, "Expected at least one hard QA record with verification_anchors"
    va = has_va[0].verification_anchors
    assert hasattr(va, "alignment_status")
    assert hasattr(va, "hop_count")
    assert hasattr(va, "is_red_flag")
    assert hasattr(va, "calculation_inputs")


def test_brsr_flag():
    """
    Purpose:
        Verify the BRSR/ESG section detector correctly identifies sections
        from India's Business Responsibility and Sustainability Reporting
        framework and does not false-positive on unrelated sections.

    Procedure:
        True cases:
          - "PRINCIPLE 6: Businesses should respect and make efforts to protect
            and restore the environment"  (BRSR Principle language)
          - "Essential Indicators | Leadership Indicators"  (BRSR Core structure)
        False cases:
          - "Financial Statements"         (pure finance section)
          - "Report on Corporate Governance" (governance, not BRSR)

    Thought process:
        BRSR/ESG evaluation is the paper's key differentiator — no other
        financial benchmark covers India's mandatory ESG reporting framework.
        The detector uses a keyword set (BRSR, Principle 1–9, Essential
        Indicators, Leadership Indicators, etc.) derived from the actual
        section names in the dataset (748 unique BRSR sections found).
        False positives would pollute the BRSR subset with non-ESG questions;
        false negatives would undercount it. Both hurt the paper's ESG analysis.
    """
    from finbharat.data.loader import is_brsr_section
    assert is_brsr_section("PRINCIPLE 6: Businesses should respect the environment") is True
    assert is_brsr_section("Essential Indicators | Leadership Indicators") is True
    assert is_brsr_section("Financial Statements") is False
    assert is_brsr_section("Report on Corporate Governance") is False


def test_table_serialization():
    """
    Purpose:
        Verify the HTML-to-Markdown and HTML-to-linearized table serializers
        produce well-formed output containing all the original data.

    Procedure:
        Input: An HTML table with header row (Item, FY2025, FY2024) and two
        data rows (Revenue, PAT with values).
        Markdown check: assert "| Item |" (pipe-format header) and "45,320" appear.
        Linearized check: assert "Revenue" and "FY2025: 45,320" (key-value format) appear.

    Thought process:
        70% of easy-tier QA and 44% of hard-tier QA are Table Only or Table+Text.
        Annual report tables are embedded as raw HTML in the evidence text.
        The table format ablation (--table-format html|markdown|linearized) is
        a required ablation study in the paper — it tests whether LLMs perform
        better with structured Markdown vs flat key-value representations.
        If the serializers produce empty or malformed output, the ablation runs
        would look like the model is failing when it's actually getting
        unreadable input.
    """
    from finbharat.data.loader import _serialize_tables
    html_table = """<table>
      <tr><th>Item</th><th>FY2025</th><th>FY2024</th></tr>
      <tr><td>Revenue</td><td>45,320</td><td>38,900</td></tr>
      <tr><td>PAT</td><td>8,240</td><td>7,100</td></tr>
    </table>"""
    md = _serialize_tables(html_table, fmt="markdown")
    assert "| Item |" in md
    assert "45,320" in md
    lin = _serialize_tables(html_table, fmt="linearized")
    assert "Revenue" in lin
    assert "FY2025: 45,320" in lin


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3B — Statistical Significance
# ─────────────────────────────────────────────────────────────────────────────

def test_bootstrap_ci():
    """
    Purpose:
        Verify bootstrap confidence interval computation returns a valid CI
        where lower ≤ mean ≤ upper and the sample count n is correct.

    Procedure:
        Input: 10 scores between 0.6 and 1.0.
        Assert: 0.6 ≤ lower ≤ mean ≤ upper ≤ 1.0, and n == 10.

    Thought process:
        EMNLP 2026 reviewers expect bootstrap 95% CIs on all main result tables.
        The CI must contain the empirical mean — if lower > mean or upper < mean,
        the implementation is wrong and results would misrepresent uncertainty.
        We use n_bootstrap=200 (not 1000) to keep the test fast while still
        exercising the resampling logic. The seed=0 ensures reproducibility
        across test runs.
    """
    from finbharat.metrics.stats import bootstrap_ci
    scores = [1.0, 0.8, 0.9, 0.7, 1.0, 0.6, 0.9, 0.8, 0.7, 0.8]
    ci = bootstrap_ci(scores, n_bootstrap=200, seed=0)
    assert 0.6 <= ci.lower <= ci.mean <= ci.upper <= 1.0
    assert ci.n == len(scores)


def test_paired_bootstrap():
    """
    Purpose:
        Verify the paired bootstrap test correctly identifies a clearly superior
        system as significant (p < 0.05) and correctly identifies two equal
        systems as not significant.

    Procedure:
        Case 1 — System A clearly better: A = [1.0]*20, B = [0.0]*20.
          Assert: significant == True, delta == 1.0.
        Case 2 — Systems equal: A = B = [0.5]*20.
          Assert: significant == False.

    Thought process:
        The paired bootstrap test is used in the paper to claim that 70B models
        significantly outperform 8B models on specific tiers. Without significance
        testing, any claim of "X is better than Y" on a 9–30 QA sample is
        statistically unverifiable. If the test falsely marks identical systems
        as significant, we would publish spurious claims. The two extreme cases
        (perfect separation vs. identical) bound the expected behavior.
    """
    from finbharat.metrics.stats import paired_bootstrap_test
    # Clearly better system
    a = [1.0] * 20
    b = [0.0] * 20
    result = paired_bootstrap_test(a, b, n_bootstrap=200, seed=0)
    assert result.significant
    assert result.delta == 1.0
    # Identical systems
    c = [0.5] * 20
    result2 = paired_bootstrap_test(c, c, n_bootstrap=200, seed=0)
    assert not result2.significant


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3C — Data Split Integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_split_creation():
    """
    Purpose:
        Verify the stratified 60/20/20 train/dev/test split covers all 187
        companies exactly once with no overlap between splits.

    Procedure:
        Create splits with seed=42. Assert:
          1. Total across all splits == 187 (all companies accounted for)
          2. No company appears in more than one split (no data leakage)
          3. All three splits are non-empty

    Thought process:
        A contaminated test set would invalidate all reported results.
        If a company appears in both train and test, the model could memorize
        specific annual report facts from training data and score artificially
        high on "unseen" test questions. The no-overlap check (using
        sector:company composite keys) is the critical guard against this.
        The count == 187 check ensures no company was accidentally dropped
        or duplicated during split assignment. seed=42 makes the split
        deterministic and reproducible across machines.
    """
    from finbharat.data.split import create_splits
    splits = create_splits(DATA_ROOT, seed=42)
    total = sum(len(v) for v in splits.values())
    assert total == 187, f"Expected 187 companies total, got {total}"
    # No leakage: each sector:company key appears exactly once
    all_keys = [
        f"{c['sector']}:{c['company']}"
        for split in splits.values()
        for c in split
    ]
    assert len(all_keys) == len(set(all_keys)), "Data leakage: a company appears in multiple splits"
    # All splits non-empty
    for name, companies in splits.items():
        assert len(companies) > 0, f"{name} split is empty"
