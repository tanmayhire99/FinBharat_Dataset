"""
LLM Numerical Equivalence Judge
=================================
Determines whether a predicted answer is numerically equivalent to the gold
answer using an LLM. Handles every case that deterministic metrics cannot:

  - "decrease by 30%" ≡ "-30%"             (direction + signed number)
  - "₹500 crore" ≡ "₹5 billion"            (unit conversion — now also in Relaxed EM)
  - "revenue grew 3x" ≡ "200% increase"    (multiplicative form)
  - "EPS of Rs 12.50" ≡ "earnings per share 12.5"  (abbreviation)
  - "fell thirty percent" ≡ "-30%"          (word-form numbers)
  - "PAT was negative" ≡ "net loss reported" (semantic equivalence)

Design decisions:
  - Direct httpx calls with judge-specific system prompt — no monkey-patching
  - Only triggered on numerical questions: requires_calculation=True
    or question_type in ("Numerical Calculation", "Table Only")
  - Batch evaluation with tqdm progress
  - Disk cache keyed by (gold, pred, question) hash — re-runs are free
  - Falls back to None (not 0) on API error — never penalises correct answers
  - Any NIM model works; llama3.1-8b is fine for this task (cost-efficient)

Usage (standalone):
    judge = NumericalJudge(model_key="llama3.1-8b")
    r = judge.evaluate("decrease by 30%", "-30%", "Did revenue fall?")
    print(r.equivalent, r.confidence, r.reason)

Usage (batch, wired into evaluation pipeline):
    judge = NumericalJudge(model_key="llama3.1-8b", cache_path=Path("results/judge_cache.jsonl"))
    results = judge.evaluate_batch(golds, preds, questions, cache_keys)
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# ── Numerical questions: only judge these types ───────────────────────────────
NUMERICAL_QUESTION_TYPES = frozenset([
    "Numerical Calculation",
    "Table Only",
    "Table with Text",
])


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a strict financial answer verifier. Your ONLY job is to decide if a
predicted answer conveys the same numerical information as the gold answer.

Equivalence rules (apply ALL of them):
1. Direction forms: "decreased by 30%" = "-30%" = "fell 30%" = "decline of 30%"
2. Indian units:    1 crore = 10 million = 0.01 billion = 100 lakh
   "₹500 crore" = "₹5 billion" = "₹5,000 million"
3. Rounding:        Allow ±2% relative tolerance. 98 ≈ 100 ✓
4. Multiplicative:  "3x" = "200% increase"  (3× original = +200%)
5. Word numbers:    "thirty percent" = "30%"
6. Abbreviations:   EPS = earnings per share, PAT = profit after tax,
                    YoY = year-on-year, EBITDA = earnings before interest…
7. Omitted units:   If gold has a unit and pred is the bare number, accept if
                    the number matches (e.g., gold="₹12.5 crore", pred="12.5")
8. Negative:        "(300)" = "-300" (accounting parenthesis = negative)

Non-equivalence (return false):
- Different magnitudes after all conversions (e.g., 30% vs 3%)
- Wrong direction (increase vs decrease)
- Completely different metrics (revenue vs profit)

Output ONLY this JSON on a single line, no other text:
{"equivalent": true, "confidence": "HIGH", "reason": "Both express a 30% decrease"}
{"equivalent": false, "confidence": "HIGH", "reason": "Gold is 30% decrease, pred is 30% increase"}
Confidence: HIGH if obvious, MEDIUM if requires interpretation, LOW if ambiguous."""

_USER_TEMPLATE = """\
Question: {question}
Gold answer: {gold}
Predicted answer: {pred}"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    equivalent: bool
    confidence: str       # "HIGH" | "MEDIUM" | "LOW"
    reason: str
    model: str
    latency_ms: float
    error: Optional[str] = None

    @property
    def score(self) -> Optional[int]:
        """1 if equivalent, 0 if not, None if error."""
        if self.error:
            return None
        return 1 if self.equivalent else 0


# ── Core judge ────────────────────────────────────────────────────────────────

class NumericalJudge:
    """
    LLM-based judge for numerical answer equivalence.
    Use with context manager or call close() when done.
    """

    def __init__(
        self,
        model_key: str = "llama3.1-8b",
        cache_path: Optional[Path] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ):
        from finbharat.models.runner import PREDEFINED_MODELS
        config = PREDEFINED_MODELS.get(model_key)
        if config is None:
            available = list(PREDEFINED_MODELS.keys())
            raise ValueError(f"Unknown model key '{model_key}'. Available: {available}")

        self._config = config
        self._api_key = api_key or os.environ.get(config.api_key_env, "")
        self._max_retries = max_retries
        self._client: Optional[httpx.Client] = None
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, JudgeResult] = {}

        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    # ── HTTP client ───────────────────────────────────────────────────────────

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._config.api_base,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    def _call_api(self, user_content: str) -> tuple[str, float]:
        """Make API call. Returns (response_text, latency_ms)."""
        start = time.time()
        backoff = 2.0
        last_err = ""

        for attempt in range(self._max_retries):
            try:
                resp = self._get_client().post(
                    "/chat/completions",
                    json={
                        "model": self._config.model_id,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user",   "content": user_content},
                        ],
                        "max_tokens": 120,   # JSON answer is short
                        "temperature": 0.0,
                    },
                )
                if resp.status_code == 429:
                    time.sleep(backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"].strip()
                return text, (time.time() - start) * 1000
            except Exception as e:
                last_err = str(e)
                time.sleep(backoff * (2 ** attempt))

        return "", (time.time() - start) * 1000

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse(self, response: str, latency: float) -> JudgeResult:
        """Parse LLM response into JudgeResult. Never crashes."""
        if not response:
            return JudgeResult(
                equivalent=False, confidence="LOW",
                reason="empty response", model=self._config.name,
                latency_ms=latency, error="empty response",
            )

        # Try JSON (preferred path)
        m = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                return JudgeResult(
                    equivalent=bool(d.get("equivalent", False)),
                    confidence=str(d.get("confidence", "LOW")).upper(),
                    reason=str(d.get("reason", "")).strip(),
                    model=self._config.name,
                    latency_ms=latency,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: extract yes/no from plain text
        lower = response.lower()
        equiv = (
            lower.startswith("yes")
            or '"equivalent": true' in lower
            or "are equivalent" in lower
            or "numerically equivalent" in lower
        )
        return JudgeResult(
            equivalent=equiv,
            confidence="LOW",
            reason=response[:120].replace("\n", " "),
            model=self._config.name,
            latency_ms=latency,
        )

    # ── Cache ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(gold: str, pred: str, question: str) -> str:
        raw = f"{gold}||{pred}||{question}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self):
        try:
            with open(self.cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    key = d.pop("_key")
                    self._cache[key] = JudgeResult(**d)
        except Exception:
            pass

    def _persist(self, key: str, result: JudgeResult):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            d = {"_key": key, **result.__dict__}
            f.write(json.dumps(d, default=str) + "\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        gold: str,
        pred: str,
        question: str = "",
    ) -> JudgeResult:
        """Evaluate a single (gold, pred) pair. Cached."""
        key = self._make_key(gold, pred, question)
        if key in self._cache:
            return self._cache[key]

        user_content = _USER_TEMPLATE.format(
            question=question or "(not provided)",
            gold=gold,
            pred=pred,
        )
        response_text, latency = self._call_api(user_content)

        if not response_text:
            result = JudgeResult(
                equivalent=False, confidence="LOW",
                reason="API call failed — defaulting to not-equivalent",
                model=self._config.name,
                latency_ms=latency,
                error="no response",
            )
        else:
            result = self._parse(response_text, latency)

        self._cache[key] = result
        self._persist(key, result)
        return result

    def evaluate_batch(
        self,
        golds: list[str],
        preds: list[str],
        questions: list[str],
    ) -> list[JudgeResult]:
        """
        Evaluate a batch of pairs, skipping cached results.
        Shows tqdm progress bar.
        """
        assert len(golds) == len(preds) == len(questions)

        # Check cache first
        keys = [self._make_key(g, p, q) for g, p, q in zip(golds, preds, questions)]
        cached_count = sum(1 for k in keys if k in self._cache)

        results: list[Optional[JudgeResult]] = [None] * len(golds)
        pending_idx = [i for i, k in enumerate(keys) if k not in self._cache]

        # Emit cached results
        for i, key in enumerate(keys):
            if key in self._cache:
                results[i] = self._cache[key]

        desc = f"LLM Judge ({self._config.name})"
        with tqdm(total=len(golds), initial=cached_count, desc=desc, unit="q") as pbar:
            for i in pending_idx:
                results[i] = self.evaluate(golds[i], preds[i], questions[i])
                pbar.update(1)

        return results  # type: ignore[return-value]

    def close(self):
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Helper: is this question worth judging? ───────────────────────────────────

def should_judge(question_type: str, requires_calculation: bool) -> bool:
    """
    Returns True for questions where the LLM judge adds value over
    deterministic metrics. Avoids burning API calls on text-only questions
    where Token F1 / ROUGE-L / BERTScore are sufficient.
    """
    return requires_calculation or question_type in NUMERICAL_QUESTION_TYPES
