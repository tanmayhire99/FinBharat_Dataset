import re
from dataclasses import dataclass


@dataclass
class NumericResult:
    num_exact: int
    num_f1: float
    num_precision: float
    num_recall: float
    gold_numbers: list[str]
    pred_numbers: list[str]


# Ordered longest-first so "crore" matches before "cr"
_INDIAN_UNIT_MAP: list[tuple[str, int]] = [
    ("crores",   10_000_000),
    ("crore",    10_000_000),
    ("cr",       10_000_000),
    ("lakhs",    100_000),
    ("lakh",     100_000),
    ("lac",      100_000),
    ("lk",       100_000),
    ("billions", 1_000_000_000),
    ("billion",  1_000_000_000),
    ("bn",       1_000_000_000),
    ("millions", 1_000_000),
    ("million",  1_000_000),
    ("mn",       1_000_000),
    ("thousands",1_000),
    ("thousand", 1_000),
    ("k",        1_000),
    # percentage — kept as-is (multiplier 1, treated separately)
]

_CURRENCY_SYMBOLS = {"₹", "$", "€", "£", "Rs", "INR", "USD", "EUR"}

# Matches: optional currency, number with optional commas/decimals, optional unit
_NUMBER_WITH_UNIT_REGEX = re.compile(
    r"(?:[₹$€£]|Rs\.?\s*|INR\s*|USD\s*)?"
    r"([\d,]+(?:\.\d+)?)"
    r"\s*(crores?|cr|lakhs?|lac|lk|billions?|bn|millions?|mn|thousands?|k"
    r"|%|per\s*cent|percent)?",
    re.IGNORECASE,
)

_PURE_NUMBER_REGEX = re.compile(r"[\d,]+(?:\.\d+)?")

# Legacy simple regex for backward-compatible extract_numbers()
_NUMBER_REGEX = re.compile(
    r"[₹$€£]?\s*(?:Rs\.?\s*|INR\s*|USD\s*)?"
    r"[\d,]+(?:\.\d+)?"
    r"\s*(?:%|per\s*cent|percent|crore|cr|lakh|lac|lk|million|mn|billion|bn|thousand|k)?",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> list[str]:
    matches = _NUMBER_REGEX.findall(text)
    numbers = []
    for m in matches:
        pure = _PURE_NUMBER_REGEX.search(m)
        if pure:
            numbers.append(pure.group())
    return numbers


def normalize_number(num_str: str) -> float | None:
    cleaned = num_str.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_canonical_value(text: str) -> float | None:
    """
    Extract a canonical absolute numeric value from a financial text fragment.

    Converts units to their absolute form so cross-unit comparisons work:
      "₹ 500 crore"   → 5_000_000_000
      "₹ 5 billion"   → 5_000_000_000   ← same canonical value
      "₹ 45,000 cr"   → 450_000_000_000
      "₹ 450 billion" → 450_000_000_000  ← same

    Returns None if no numeric value can be extracted.
    """
    text_lower = text.lower().strip()
    # Remove currency symbols/prefixes
    for sym in ("₹", "$", "€", "£", "rs.", "rs ", "inr", "usd", "eur"):
        text_lower = text_lower.replace(sym, " ")
    text_lower = text_lower.strip()

    # Try to find a number followed by a unit
    for unit_str, multiplier in _INDIAN_UNIT_MAP:
        pattern = rf"([\d,]+(?:\.\d+)?)\s*{re.escape(unit_str)}\b"
        m = re.search(pattern, text_lower)
        if m:
            val = normalize_number(m.group(1))
            if val is not None:
                return val * multiplier

    # No unit found — just extract the number
    m = re.search(r"([\d,]+(?:\.\d+)?)", text_lower)
    if m:
        return normalize_number(m.group(1))
    return None


def normalize_indian_number(num_str: str, unit_hint: str = "") -> float | None:
    """Normalize a number with an optional external unit hint."""
    base = normalize_number(num_str)
    if base is None:
        return None
    unit_hint_lower = unit_hint.lower().strip()
    for unit_name, multiplier in _INDIAN_UNIT_MAP:
        if unit_name in unit_hint_lower:
            return base * multiplier
    return base


def are_numerically_equivalent(text_a: str, text_b: str, tolerance: float = 0.015) -> bool:
    """
    Compare two financial text fragments for numeric equivalence after
    converting both to a canonical absolute value.

    Handles:
      - Indian unit conversions: crore ↔ lakh ↔ million ↔ billion
      - Currency symbol differences
      - Comma formatting differences

    Examples:
      "₹ 500 crore" ≡ "₹ 5 billion"     → True
      "₹ 45,000 crores" ≡ "₹ 450 billion" → True
      "1,23,456" ≡ "123456"               → True
    """
    va = extract_canonical_value(text_a)
    vb = extract_canonical_value(text_b)
    if va is None or vb is None:
        return False
    if abs(va) < 1e-10 and abs(vb) < 1e-10:
        return True
    if abs(va) < 1e-10 or abs(vb) < 1e-10:
        return False
    return abs(va - vb) / max(abs(va), abs(vb)) <= tolerance


def compute_numeric_metrics(gold: str, pred: str) -> NumericResult:
    gold_nums_raw = extract_numbers(gold)
    pred_nums_raw = extract_numbers(pred)
    gold_nums = [normalize_number(n) for n in gold_nums_raw]
    pred_nums = [normalize_number(n) for n in pred_nums_raw]
    gold_nums = [n for n in gold_nums if n is not None]
    pred_nums = [n for n in pred_nums if n is not None]

    if not gold_nums:
        has_target = False
        num_exact = 0
    else:
        has_target = True
        target = gold_nums[-1]
        num_exact = 1 if pred_nums and abs(pred_nums[-1] - target) < 0.01 else 0

    if not gold_nums and not pred_nums:
        return NumericResult(
            num_exact=1, num_f1=1.0, num_precision=1.0, num_recall=1.0,
            gold_numbers=gold_nums_raw, pred_numbers=pred_nums_raw,
        )

    gold_set = _build_number_set(gold_nums)
    pred_set = _build_number_set(pred_nums)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return NumericResult(
        num_exact=num_exact,
        num_f1=round(f1, 4),
        num_precision=round(precision, 4),
        num_recall=round(recall, 4),
        gold_numbers=gold_nums_raw,
        pred_numbers=pred_nums_raw,
    )


def _build_number_set(nums: list[float], tol: float = 0.01) -> set[int]:
    rounded = set()
    for n in nums:
        bucket = round(n / tol)
        rounded.add(bucket)
    return rounded


def compute_tolerance_accuracy(gold: str, pred: str, tolerance_pct: float = 5.0) -> int:
    gold_nums = [n for n in (normalize_number(x) for x in extract_numbers(gold)) if n is not None]
    pred_nums = [n for n in (normalize_number(x) for x in extract_numbers(pred)) if n is not None]
    if not gold_nums or not pred_nums:
        return 0
    target = gold_nums[-1]
    predicted = pred_nums[-1]
    if abs(target) < 1e-10:
        return 1 if abs(predicted) < 1e-10 else 0
    pct_error = abs(predicted - target) / abs(target) * 100
    return 1 if pct_error <= tolerance_pct else 0


def compute_mape(golds: list[str], preds: list[str]) -> float | None:
    """
    Mean Absolute Percentage Error.

    Guards:
    - Skip pairs where gold value is very small (< 1.0) — percentages like 0.74%
      cause MAPE to explode when predictions are in absolute terms.
    - Cap individual errors at 1000% before averaging to prevent single outliers
      dominating the mean (sMAPE alternative).
    """
    errors = []
    for g, p in zip(golds, preds):
        gold_nums = [n for n in (normalize_number(x) for x in extract_numbers(g)) if n is not None]
        pred_nums = [n for n in (normalize_number(x) for x in extract_numbers(p)) if n is not None]
        if gold_nums and pred_nums:
            target = gold_nums[-1]
            predicted = pred_nums[-1]
            # Skip very small denominators (percentages expressed as decimals, near-zero values)
            if abs(target) < 1.0:
                continue
            pct_error = abs(predicted - target) / abs(target)
            errors.append(min(pct_error, 10.0))  # cap at 1000% (10.0 × 100)
    if not errors:
        return None
    return round(sum(errors) / len(errors) * 100, 4)
