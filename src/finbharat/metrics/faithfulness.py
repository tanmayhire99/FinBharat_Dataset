import re
from dataclasses import dataclass
from typing import Optional

_nli_pipeline = None
_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"  # ~180MB; upgrade to large for final paper


def _get_nli_pipeline():
    """Lazy-load DeBERTa NLI pipeline on first call."""
    global _nli_pipeline
    if _nli_pipeline is None:
        try:
            from transformers import pipeline
            _nli_pipeline = pipeline(
                "zero-shot-classification",
                model=_NLI_MODEL,
                device=-1,  # CPU; set to 0 for GPU
            )
        except ImportError:
            pass  # transformers not installed — fall back to heuristic
    return _nli_pipeline


@dataclass
class NLIResult:
    entailment_ratio: float
    contradiction_ratio: float
    neutral_ratio: float
    num_sentences: int
    sentence_labels: list[str]
    model: str = "heuristic"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def compute_nli_entailment(evidence: str, answer: str, use_model: bool = True) -> NLIResult:
    sentences = split_sentences(answer)
    if not sentences:
        return NLIResult(
            entailment_ratio=1.0, contradiction_ratio=0.0, neutral_ratio=0.0,
            num_sentences=0, sentence_labels=[], model="heuristic",
        )

    pipe = _get_nli_pipeline() if use_model else None
    if pipe is not None:
        return _deberta_nli(evidence, sentences, pipe)
    return _heuristic_nli(evidence, sentences)


def _deberta_nli(evidence: str, sentences: list[str], pipe) -> NLIResult:
    """Use DeBERTa zero-shot-classification as NLI: evidence as context, sentence as candidate."""
    candidate_labels = ["entailment", "contradiction", "neutral"]
    labels = []
    # Truncate evidence to avoid token limit
    ev_trunc = evidence[:1500]
    for sent in sentences:
        try:
            result = pipe(
                sequences=sent,
                candidate_labels=candidate_labels,
                hypothesis_template="{}",
                multi_label=False,
            )
            # Top predicted label
            top = result["labels"][0].upper()
            labels.append(top)
        except Exception:
            labels.append("NEUTRAL")

    ent = sum(1 for l in labels if l == "ENTAILMENT") / len(labels)
    con = sum(1 for l in labels if l == "CONTRADICTION") / len(labels)
    neu = sum(1 for l in labels if l == "NEUTRAL") / len(labels)
    return NLIResult(
        entailment_ratio=round(ent, 4),
        contradiction_ratio=round(con, 4),
        neutral_ratio=round(neu, 4),
        num_sentences=len(labels),
        sentence_labels=labels,
        model=_NLI_MODEL,
    )


def _heuristic_nli(evidence: str, sentences: list[str]) -> NLIResult:
    ev_lower = evidence.lower()
    labels = []
    for sent in sentences:
        sent_lower = sent.lower().strip()
        if len(sent_lower) < 5:
            labels.append("NEUTRAL")
            continue
        if sent_lower in ev_lower:
            labels.append("ENTAILMENT")
            continue
        words = sent_lower.split()
        overlap = sum(1 for w in words if w in ev_lower)
        ratio = overlap / len(words) if words else 0
        if ratio >= 0.75:
            labels.append("ENTAILMENT")
        elif ratio < 0.3:
            labels.append("CONTRADICTION")
        else:
            labels.append("NEUTRAL")

    ent = sum(1 for l in labels if l == "ENTAILMENT") / len(labels)
    con = sum(1 for l in labels if l == "CONTRADICTION") / len(labels)
    neu = sum(1 for l in labels if l == "NEUTRAL") / len(labels)

    return NLIResult(
        entailment_ratio=round(ent, 4),
        contradiction_ratio=round(con, 4),
        neutral_ratio=round(neu, 4),
        num_sentences=len(labels),
        sentence_labels=labels,
        model="heuristic",
    )


def compute_evidence_traceability(evidence: str, gold_evidence: str) -> float:
    ev_clean = re.sub(r'\s+', ' ', evidence.lower().strip())
    gold_clean = re.sub(r'\s+', ' ', gold_evidence.lower().strip())
    if not gold_clean or not ev_clean:
        return 0.0
    if ev_clean in gold_clean or gold_clean in ev_clean:
        return 1.0
    ev_words = set(ev_clean.split())
    gold_words = set(gold_clean.split())
    if not ev_words or not gold_words:
        return 0.0
    overlap = len(ev_words & gold_words) / min(len(ev_words), len(gold_words))
    return round(overlap, 4)
