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


_INDIAN_UNIT_MAP = {
    "crore": 10_000_000,
    "cr": 10_000_000,
    "lakh": 100_000,
    "lac": 100_000,
    "lk": 100_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "m": None,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "thousand": 1_000,
    "k": 1_000,
}

_CURRENCY_SYMBOLS = {"₹", "$", "€", "£", "Rs", "INR", "USD", "EUR"}

_NUMBER_REGEX = re.compile(
    r"[₹$€£]?\s*(?:Rs\.?\s*|INR\s*|USD\s*)?"
    r"[\d,]+(?:\.\d+)?"
    r"\s*(?:%|per\s*cent|percent|crore|cr|lakh|lac|lk|million|mn|billion|bn|thousand|k)?",
    re.IGNORECASE,
)

_PURE_NUMBER_REGEX = re.compile(r"[\d,]+(?:\.\d+)?")


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


def normalize_indian_number(num_str: str, unit_hint: str = "") -> float | None:
    base = normalize_number(num_str)
    if base is None:
        return None
    unit_hint_lower = unit_hint.lower().strip()
    for unit_name, multiplier in _INDIAN_UNIT_MAP.items():
        if unit_name in unit_hint_lower and multiplier is not None:
            return base * multiplier
    return base


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
    errors = []
    for g, p in zip(golds, preds):
        gold_nums = [n for n in (normalize_number(x) for x in extract_numbers(g)) if n is not None]
        pred_nums = [n for n in (normalize_number(x) for x in extract_numbers(p)) if n is not None]
        if gold_nums and pred_nums:
            target = gold_nums[-1]
            predicted = pred_nums[-1]
            if abs(target) > 1e-10:
                errors.append(abs(predicted - target) / abs(target))
    if not errors:
        return None
    return sum(errors) / len(errors) * 100
