import json
from pathlib import Path
from finbharat.data.loader import FinBharatDataset, SAMPLE_COMPANIES, parse_qa_record
from finbharat.metrics.numeric import compute_numeric_metrics, compute_tolerance_accuracy, normalize_number, extract_numbers
from finbharat.metrics.text import compute_exact_match, compute_token_f1, compute_relaxed_em, compute_directional_accuracy
from finbharat.metrics.faithfulness import compute_nli_entailment, compute_evidence_traceability


# tests/ is one level under project root, so dataset is at project_root/dataset
DATA_ROOT = Path(__file__).parent.parent / "dataset"


def test_load_easy_qa():
    dataset = FinBharatDataset(DATA_ROOT)
    records = dataset.load_easy_qa("Private_Sector_Bank", "HDFC_Bank")
    assert len(records) > 0
    assert records[0].difficulty == "easy"
    assert records[0].question_type in ("Text Only", "Table Only", "Table with Text", "Numerical Calculation")


def test_load_sample():
    dataset = FinBharatDataset(DATA_ROOT)
    records = dataset.load_sample(difficulty="easy")
    assert len(records) == 383
    companies = {r.company for r in records}
    assert companies == {"HDFC_Bank", "Bosch", "Interglobe_Aviat"}


def test_load_chunks():
    dataset = FinBharatDataset(DATA_ROOT)
    chunks = dataset.load_chunks("Private_Sector_Bank", "HDFC_Bank")
    assert len(chunks) == 830
    assert chunks[0].chunk_id.startswith("Private_Sector_Bank/HDFC_Bank")


def test_numeric_metrics_basic():
    result = compute_numeric_metrics("Rs 41,173 million", "Rs 41,173 million")
    assert result.num_exact == 1
    assert result.num_f1 == 1.0


def test_numeric_metrics_mismatch():
    result = compute_numeric_metrics("Rs 41,173 million", "Rs 42,000 million")
    assert result.num_exact == 0


def test_numeric_metrics_partial():
    result = compute_numeric_metrics("549,498 and 103,386", "549,498 and 100,000")
    assert result.num_f1 > 0
    assert result.num_f1 < 1.0


def test_tolerance_accuracy():
    assert compute_tolerance_accuracy("1000", "1050", tolerance_pct=5.0) == 1
    assert compute_tolerance_accuracy("1000", "1060", tolerance_pct=5.0) == 0
    assert compute_tolerance_accuracy("1000", "1020", tolerance_pct=5.0) == 1


def test_exact_match():
    assert compute_exact_match("Thirteen (13) Directors", "Thirteen (13) Directors") == 1
    assert compute_exact_match("Thirteen (13) Directors", "14 Directors") == 0


def test_relaxed_em():
    assert compute_relaxed_em("₹ 139.27 crores", "139.27") == 1
    assert compute_relaxed_em("Rs 41,173 million", "41173") == 1


def test_token_f1():
    result = compute_token_f1("Thirteen Directors on the Board", "Thirteen Directors")
    assert result.f1 > 0.5
    assert result.f1 <= 1.0


def test_directional_accuracy():
    assert compute_directional_accuracy("Revenue increased by 18%", "Revenue increased") == 1
    assert compute_directional_accuracy("Revenue increased by 18%", "Revenue declined") == 0


def test_nli_entailment_heuristic():
    evidence = "The Board consists of thirteen (13) Directors."
    answer = "The Board has thirteen Directors."
    result = compute_nli_entailment(evidence, answer)
    assert result.entailment_ratio > 0


def test_evidence_traceability():
    evidence = "As on March 31, 2025, the Balance Sheet size was US $ 10.83 billion."
    gold = "As on March 31, 2025, the Balance Sheet size was US $ 10.83 billion."
    assert compute_evidence_traceability(evidence, gold) == 1.0


def test_normalize_number():
    assert normalize_number("41,173") == 41173.0
    assert normalize_number("10.83") == 10.83
    assert normalize_number("abc") is None


def test_extract_numbers():
    nums = extract_numbers("Rs 41,173 million and ₹ 549,498")
    assert "41,173" in nums
    assert "549,498" in nums
