import re
import string
from dataclasses import dataclass
from rouge_score import rouge_scorer as _rouge_scorer

_SCORER = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


@dataclass
class TextResult:
    exact_match: int
    f1: float
    precision: float
    recall: float


_DIRECTION_MAP = {
    "increase": "up", "increased": "up", "growth": "up", "grew": "up",
    "rose": "up", "higher": "up", "up": "up", "gain": "up", "gained": "up",
    "decrease": "down", "decreased": "down", "decline": "down", "declined": "down",
    "fell": "down", "drop": "down", "dropped": "down", "lower": "down", "down": "down",
    "reduction": "down", "reduced": "down", "shrink": "down", "shrunk": "down",
    "no change": "flat", "unchanged": "flat", "stable": "flat", "flat": "flat",
    "yes": "yes", "no": "no",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for sym in ("₹", "$", "€", "£", "rs", "inr"):
        text = text.replace(sym, "")
    return text.strip()


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def compute_exact_match(gold: str, pred: str) -> int:
    return 1 if normalize_text(gold) == normalize_text(pred) else 0


def compute_token_f1(gold: str, pred: str) -> TextResult:
    gold_tokens = set(tokenize(gold))
    pred_tokens = set(tokenize(pred))
    if not gold_tokens and not pred_tokens:
        return TextResult(exact_match=1, f1=1.0, precision=1.0, recall=1.0)
    if not gold_tokens or not pred_tokens:
        return TextResult(exact_match=compute_exact_match(gold, pred), f1=0.0, precision=0.0, recall=0.0)
    common = gold_tokens & pred_tokens
    tp = len(common)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return TextResult(
        exact_match=compute_exact_match(gold, pred),
        f1=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
    )


def compute_relaxed_em(gold: str, pred: str) -> int:
    # Relaxed EM is a superset of EM — pass EM first as a fast-path
    if compute_exact_match(gold, pred):
        return 1

    def _strip_units(t: str) -> str:
        t = t.lower().strip()
        for unit in ("crores", "crore", "cr", "lakhs", "lakh", "lac", "lk",
                      "millions", "million", "mn", "billions", "billion", "bn",
                      "thousands", "thousand", "rs", "inr", "₹", "$"):
            t = t.replace(unit, "")
        # Strip all punctuation and whitespace for numeric comparison
        t = re.sub(r"[,()%\s]", "", t)
        return t

    g = _strip_units(gold)
    p = _strip_units(pred)
    if g == p:
        return 1
    try:
        gf = float(g)
        pf = float(p)
        return 1 if abs(gf - pf) < 0.015 * max(abs(gf), 1) else 0
    except ValueError:
        return 0


def extract_directional_label(text: str) -> str | None:
    lower = normalize_text(text)
    words = lower.split()
    for word in reversed(words):
        if word in _DIRECTION_MAP:
            return _DIRECTION_MAP[word]
    if "yes" in lower:
        return "yes"
    if "no" in lower:
        return "no"
    return None


def compute_rouge_l(gold: str, pred: str) -> float:
    if not gold.strip() or not pred.strip():
        return 0.0
    scores = _SCORER.score(gold, pred)
    return round(scores["rougeL"].fmeasure, 4)


def compute_directional_accuracy(gold: str, pred: str) -> int | None:
    gold_label = extract_directional_label(gold)
    pred_label = extract_directional_label(pred)
    if gold_label is None:
        return None
    if pred_label is None:
        return 0
    return 1 if gold_label == pred_label else 0
