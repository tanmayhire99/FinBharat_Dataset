import re
import string
from dataclasses import dataclass
from rouge_score import rouge_scorer as _rouge_scorer

_SCORER = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

# METEOR uses NLTK — download wordnet once lazily
_meteor_ready = False

def _ensure_meteor():
    global _meteor_ready
    if not _meteor_ready:
        import nltk
        for pkg in ("wordnet", "punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        _meteor_ready = True


@dataclass
class TextResult:
    exact_match: int
    f1: float
    precision: float
    recall: float


_DIRECTION_MAP = {
    # Positive / upward
    "increase": "up", "increased": "up", "increases": "up",
    "growth": "up", "grew": "up", "grow": "up",
    "rose": "up", "rise": "up", "risen": "up",
    "higher": "up", "up": "up",
    "gain": "up", "gained": "up", "gains": "up",
    "improved": "up", "improvement": "up",
    "profit": "up", "profits": "up",
    "strengthened": "up", "expansion": "up", "expanded": "up",
    "surge": "up", "surged": "up", "jumped": "up",
    # Negative / downward
    "decrease": "down", "decreased": "down", "decreases": "down",
    "decline": "down", "declined": "down", "declining": "down",
    "fell": "down", "fall": "down", "fallen": "down",
    "drop": "down", "dropped": "down", "drops": "down",
    "lower": "down", "down": "down",
    "reduction": "down", "reduced": "down", "reduce": "down",
    "shrink": "down", "shrunk": "down", "shrank": "down",
    "loss": "down", "losses": "down",
    "deteriorated": "down", "weakened": "down", "contraction": "down", "contracted": "down",
    "negative": "down", "deficit": "down",
    # Flat
    "no change": "flat", "unchanged": "flat", "stable": "flat", "flat": "flat",
    "marginal": "flat",
    # Boolean
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
    """
    Relaxed Exact Match — a strictly looser metric than EM.

    Handles:
    1. Indian unit stripping: "₹ 139.27 crores" == "139.27"
    2. Sign-direction equivalence: "decrease by 30%" == "-30%"
       i.e. if gold has a direction word and pred has a signed number
       (or vice versa), they match when the number and sign agree.
    3. Numeric near-match: ±1.5% tolerance after unit stripping.
    """
    # Fast-path: EM already passes
    if compute_exact_match(gold, pred):
        return 1

    # Sign-direction equivalence check
    # e.g. "decrease by 30%" ↔ "-30%",  "increase by 5%" ↔ "+5%"
    gold_dir = extract_directional_label(gold)
    pred_dir = extract_directional_label(pred)
    if gold_dir is not None and pred_dir is not None:
        if gold_dir == pred_dir:
            # Directions match — check if the magnitude matches too
            from finbharat.metrics.numeric import extract_numbers, normalize_number
            gn = [normalize_number(x) for x in extract_numbers(gold)]
            pn = [normalize_number(x) for x in extract_numbers(pred)]
            gn = [v for v in gn if v is not None]
            pn = [v for v in pn if v is not None]
            if gn and pn:
                gv, pv = abs(gn[-1]), abs(pn[-1])
                if abs(gv) < 1e-10 and abs(pv) < 1e-10:
                    return 1
                if abs(gv) > 1e-10 and abs(gv - pv) / abs(gv) < 0.015:
                    return 1
            else:
                # No numbers — direction match alone is enough
                return 1

    # Cross-unit canonical comparison: "₹500 crore" ≡ "₹5 billion"
    # Guard: skip if directions are explicitly contradictory (decrease vs +30%)
    from finbharat.metrics.numeric import are_numerically_equivalent
    _g_dir = extract_directional_label(gold)
    _p_dir = extract_directional_label(pred)
    _dirs_ok = (
        _g_dir is None or _p_dir is None          # at least one has no direction
        or _g_dir == _p_dir                        # both have same direction
    )
    if _dirs_ok and are_numerically_equivalent(gold, pred):
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


_FLAT_PHRASES = frozenset([
    "no change", "no significant change", "no major change", "no material change",
    "no meaningful change", "remained unchanged", "remained stable",
    "no notable change", "no considerable change",
])


def extract_directional_label(text: str) -> str | None:
    """
    Extract a directional label from text.

    Handles both word-based directions ("increased", "declined") and
    signed numeric formats common in financial reports:
      "-30%"   → "down"   (negative sign = decrease)
      "+5.2%"  → "up"     (explicit positive sign = increase)
      "(15%)"  → "down"   (parentheses = negative in accounting notation)

    Multi-word flat phrases like "no change" are checked BEFORE single words
    to prevent "no change" from returning "no" (boolean) instead of "flat".
    """
    stripped = text.strip()

    # Signed numeric formats: -30%, -0.5, +12.3%, +2 crore
    if re.match(r'^-[\d]', stripped):
        return "down"
    if re.match(r'^\+[\d]', stripped):
        return "up"
    # Accounting parentheses notation: (30%) or (₹ 1,200)
    if re.match(r'^\([\d₹$]', stripped):
        return "down"

    lower = normalize_text(text)

    # Check multi-word flat phrases FIRST — prevents "no change" → "no"
    for phrase in _FLAT_PHRASES:
        if phrase in lower:
            return "flat"

    words = lower.split()
    for word in reversed(words):
        if word in _DIRECTION_MAP:
            return _DIRECTION_MAP[word]
    if "yes" in lower:
        return "yes"
    if "no" in lower:
        return "no"
    return None


def compute_meteor(gold: str, pred: str) -> float:
    """METEOR score — handles synonyms and stemming better than ROUGE for short answers."""
    if not gold.strip() or not pred.strip():
        return 0.0
    try:
        _ensure_meteor()
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        score = meteor_score([word_tokenize(gold.lower())], word_tokenize(pred.lower()))
        return round(float(score), 4)
    except Exception:
        # Graceful fallback: return token F1 if NLTK unavailable
        return compute_token_f1(gold, pred).f1


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
