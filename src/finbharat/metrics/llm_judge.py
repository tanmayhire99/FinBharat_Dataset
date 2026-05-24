"""
LLM-based Numerical Equivalence Judge
======================================
Uses an LLM to determine if two financial answers are numerically equivalent.
This handles all cases that deterministic metrics cannot:
  - "decrease by 30%" ≡ "-30%"           (direction + signed number)
  - "₹500 crore" ≡ "₹5 billion"          (unit conversion)
  - "revenue grew 3x" ≡ "200% increase"  (multiplicative form)
  - "EPS of Rs 12.50" ≡ "12.5"           (abbreviation stripping)
  - "fell thirty percent" ≡ "-30%"       (word-form numbers)

Design:
  - Cheap: ~40 tokens per call (yes/no answer)
  - Cached: results stored per (question_id, model_name) to avoid re-calls
  - Graceful: falls back to 0 on any API error
  - Uses any model in PREDEFINED_MODELS, defaults to llama3.3-70b (free on NIM)

Usage:
    judge = NumericalJudge(model_key="llama3.3-70b")
    result = judge.evaluate("decrease by 30%", "-30%", question="Did revenue change?")
    print(result.equivalent)   # True
    print(result.confidence)   # "HIGH"
    print(result.reason)       # "Both express a 30% decrease"
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class JudgeResult:
    equivalent: bool        # True if answers are numerically equivalent
    confidence: str         # "HIGH" | "MEDIUM" | "LOW"
    reason: str             # brief explanation
    model: str              # which model judged this
    latency_ms: float
    error: Optional[str] = None


_JUDGE_SYSTEM_PROMPT = """You are a precise financial answer evaluator.
Your task: determine if a predicted answer is numerically equivalent to the gold answer.

Rules:
1. Narrative ↔ signed number: "decreased by 30%" = "-30%" = "fell 30%" = "decline of 30%"
2. Unit conversion: "₹500 crore" = "₹5 billion" = "₹5,000 million" = "5 × 10^9 rupees"
3. Indian units: 1 crore = 10 million = 0.01 billion = 100 lakh
4. Decimal equivalence: "12.50" = "12.5" = "12½"
5. Multiplicative: "3x" = "200% increase" (3x means 300% of original = 200% increase)
6. Word numbers: "thirty percent" = "30%"
7. Abbreviations: "EPS" = "earnings per share", "PAT" = "profit after tax", "YoY" = year-on-year
8. Rounding: allow ±2% tolerance for rounding differences
9. Only compare the core numerical claim — ignore verbal filler

IMPORTANT: Answer in exactly this JSON format, nothing else:
{"equivalent": true/false, "confidence": "HIGH"/"MEDIUM"/"LOW", "reason": "one sentence"}"""

_JUDGE_USER_TEMPLATE = """Gold answer: {gold}
Predicted answer: {pred}
Question (for context): {question}

Are these numerically equivalent?"""


class NumericalJudge:
    """
    LLM-based numerical equivalence judge.

    Designed to be used only for questions where requires_calculation=True
    or question_type is 'Numerical Calculation' — the cases where deterministic
    metrics are most likely to fail.
    """

    def __init__(
        self,
        model_key: str = "llama3.3-70b",
        cache_path: Optional[Path] = None,
        api_key: Optional[str] = None,
    ):
        self.model_key = model_key
        self.cache_path = cache_path
        self._cache: dict[str, JudgeResult] = {}
        self._api_key = api_key
        self._runner = None

        if cache_path and Path(cache_path).exists():
            self._load_cache()

    def _load_cache(self):
        try:
            with open(self.cache_path) as f:
                for line in f:
                    d = json.loads(line.strip())
                    key = d.pop("_cache_key")
                    self._cache[key] = JudgeResult(**d)
        except Exception:
            pass

    def _save_to_cache(self, key: str, result: JudgeResult):
        if not self.cache_path:
            return
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            d = {"_cache_key": key, **result.__dict__}
            f.write(json.dumps(d) + "\n")

    def _get_runner(self):
        if self._runner is None:
            from finbharat.models.runner import ModelRunner, PREDEFINED_MODELS
            config = PREDEFINED_MODELS.get(self.model_key)
            if config is None:
                raise ValueError(f"Unknown model key: {self.model_key}")
            self._runner = ModelRunner(config, api_key=self._api_key)
        return self._runner

    def evaluate(
        self,
        gold: str,
        pred: str,
        question: str = "",
        cache_key: Optional[str] = None,
    ) -> JudgeResult:
        """
        Evaluate if gold and pred are numerically equivalent.
        Results are cached by cache_key to avoid redundant API calls.
        """
        key = cache_key or f"{hash(gold)}:{hash(pred)}"
        if key in self._cache:
            return self._cache[key]

        start = time.time()
        prompt = _JUDGE_USER_TEMPLATE.format(gold=gold, pred=pred, question=question)

        try:
            runner = self._get_runner()
            # Temporarily override system prompt for judge role
            import finbharat.models.runner as _r
            orig_system = _r.QA_SYSTEM_PROMPT
            _r.QA_SYSTEM_PROMPT = _JUDGE_SYSTEM_PROMPT

            gen = runner.generate(question=prompt, context="")
            _r.QA_SYSTEM_PROMPT = orig_system

            latency = (time.time() - start) * 1000

            if gen.error:
                result = JudgeResult(
                    equivalent=False, confidence="LOW",
                    reason=f"API error: {gen.error}", model=self.model_key,
                    latency_ms=latency, error=gen.error,
                )
            else:
                result = self._parse_response(gen.predicted_answer, latency)

        except Exception as e:
            latency = (time.time() - start) * 1000
            result = JudgeResult(
                equivalent=False, confidence="LOW",
                reason=f"Exception: {e}", model=self.model_key,
                latency_ms=latency, error=str(e),
            )

        self._cache[key] = result
        self._save_to_cache(key, result)
        return result

    def _parse_response(self, response: str, latency: float) -> JudgeResult:
        """Parse LLM JSON response into JudgeResult."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                d = json.loads(json_match.group())
                return JudgeResult(
                    equivalent=bool(d.get("equivalent", False)),
                    confidence=str(d.get("confidence", "LOW")).upper(),
                    reason=str(d.get("reason", "")),
                    model=self.model_key,
                    latency_ms=latency,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: look for yes/no in plain text
        lower = response.lower()
        equivalent = "yes" in lower[:50] or '"equivalent": true' in lower
        return JudgeResult(
            equivalent=equivalent,
            confidence="LOW",
            reason=response[:100],
            model=self.model_key,
            latency_ms=latency,
        )

    def evaluate_batch(
        self,
        golds: list[str],
        preds: list[str],
        questions: list[str],
        cache_keys: Optional[list[str]] = None,
    ) -> list[JudgeResult]:
        """Evaluate a batch of (gold, pred) pairs."""
        results = []
        keys = cache_keys or [None] * len(golds)
        for gold, pred, question, key in zip(golds, preds, questions, keys):
            results.append(self.evaluate(gold, pred, question, key))
        return results

    def close(self):
        if self._runner:
            self._runner.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def judge_numerical_answers(
    gold: str,
    pred: str,
    question: str = "",
    model_key: str = "llama3.3-70b",
    api_key: Optional[str] = None,
) -> JudgeResult:
    """
    Convenience one-shot function for evaluating a single (gold, pred) pair.
    Creates a fresh judge, evaluates, and closes. Use NumericalJudge directly
    for batch evaluation or caching.
    """
    with NumericalJudge(model_key=model_key, api_key=api_key) as judge:
        return judge.evaluate(gold, pred, question)
