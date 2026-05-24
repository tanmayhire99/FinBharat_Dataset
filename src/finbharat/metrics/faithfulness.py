import re
from dataclasses import dataclass
from typing import Optional

_nli_pipeline = None  # reset on import — forces correct pipeline type to load
_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"  # ~180MB; upgrade to large for final paper

# Label order emitted by cross-encoder/nli-deberta-v3-small: contradiction, entailment, neutral
_LABEL_MAP = {"contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL"}


def _get_nli_pipeline():
    """Lazy-load DeBERTa NLI text-classification pipeline on first call."""
    global _nli_pipeline
    if _nli_pipeline is None:
        try:
            from transformers import pipeline
            # Use text-classification, NOT zero-shot-classification.
            # cross-encoder NLI models expect (premise, hypothesis) text pairs.
            _nli_pipeline = pipeline(
                "text-classification",
                model=_NLI_MODEL,
                device=-1,  # CPU; use 0 for GPU
                function_to_apply="softmax",
                top_k=None,  # return all label scores
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
        # Empty or whitespace-only answer — not entailed, not contradicted.
        # Returning 1.0 would falsely mark blank predictions as perfectly faithful.
        return NLIResult(
            entailment_ratio=0.0, contradiction_ratio=0.0, neutral_ratio=1.0,
            num_sentences=0, sentence_labels=[], model="heuristic",
        )

    pipe = _get_nli_pipeline() if use_model else None
    if pipe is not None:
        return _deberta_nli(evidence, sentences, pipe)
    return _heuristic_nli(evidence, sentences)


def _deberta_nli(evidence: str, sentences: list[str], pipe) -> NLIResult:
    """
    Use cross-encoder NLI: premise = evidence, hypothesis = answer sentence.
    text-classification pipeline returns all label scores; pick argmax.
    """
    labels = []
    # Truncate evidence to stay within token budget (~512 tokens ≈ 1800 chars)
    ev_trunc = evidence[:1800]
    for sent in sentences:
        try:
            # Input is (premise, hypothesis) text pair
            outputs = pipe({"text": ev_trunc, "text_pair": sent})
            # outputs is a list of {"label": ..., "score": ...}
            best = max(outputs, key=lambda x: x["score"])
            label = _LABEL_MAP.get(best["label"].lower(), "NEUTRAL")
            labels.append(label)
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
    # Build word-level vocabulary of evidence using word boundaries.
    # Using `w in ev_lower` (substring) was a bug: "NOT ₹100" would match
    # "₹100" because the substring "100" appears in both.
    ev_lower = evidence.lower()
    ev_words = set(re.findall(r'\b\w+\b', ev_lower))
    labels = []
    for sent in sentences:
        sent_lower = sent.lower().strip()
        if len(sent_lower) < 5:
            labels.append("NEUTRAL")
            continue
        if sent_lower in ev_lower:
            labels.append("ENTAILMENT")
            continue
        # Word-boundary match (not substring) prevents false positives
        words = re.findall(r'\b\w+\b', sent_lower)
        if not words:
            labels.append("NEUTRAL")
            continue
        overlap = sum(1 for w in words if w in ev_words)
        ratio = overlap / len(words)
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
