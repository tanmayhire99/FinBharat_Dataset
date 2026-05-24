import os
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from finbharat.data.loader import QARecord

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


@dataclass
class ModelConfig:
    name: str
    model_id: str
    provider: str = "nim"
    api_base: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0


PREDEFINED_MODELS: dict[str, ModelConfig] = {
    # --- Small (≤8B) ---
    "llama3.1-8b": ModelConfig(
        name="Llama-3.1-8B-Instruct",
        model_id="meta/llama-3.1-8b-instruct",
    ),
    "llama3.2-3b": ModelConfig(
        name="Llama-3.2-3B-Instruct",
        model_id="meta/llama-3.2-3b-instruct",
    ),
    "qwen3-8b": ModelConfig(
        name="Qwen3-8B-Instruct",
        model_id="qwen/qwen3-8b",
    ),
    "gemma3-4b": ModelConfig(
        name="Gemma-3-4B-Instruct",
        model_id="google/gemma-3-4b-it",
    ),
    # --- Medium (14–24B) ---
    "qwen3-14b": ModelConfig(
        name="Qwen3-14B-Instruct",
        model_id="qwen/qwen3-14b",
    ),
    "mistral-small": ModelConfig(
        name="Mistral-Small-3.2-24B",
        model_id="mistralai/mistral-small-3.2-24b-instruct",
    ),
    # --- Large (70B+) ---
    "llama3.3-70b": ModelConfig(
        name="Llama-3.3-70B-Instruct",
        model_id="meta/llama-3.3-70b-instruct",
    ),
    "qwen3-72b": ModelConfig(
        name="Qwen3-72B-Instruct",
        model_id="qwen/qwen3-72b",
    ),
    "deepseek-r1-70b": ModelConfig(
        name="DeepSeek-R1-Distill-70B",
        model_id="deepseek-ai/deepseek-r1",
        max_tokens=1024,
    ),
    # --- Reasoning / MoE ---
    "qwen3-235b": ModelConfig(
        name="Qwen3-235B-A22B",
        model_id="qwen/qwen3-235b-a22b",
    ),
    # --- Closed (OpenAI) ---
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        model_id="gpt-4o",
        provider="openai",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
}


@dataclass
class GenerationResult:
    question_id: str
    model_name: str
    question: str
    gold_answer: str
    predicted_answer: str
    context: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


QA_SYSTEM_PROMPT = (
    "You are a financial analyst answering questions about Indian company annual reports. "
    "Answer the question based ONLY on the provided context. Do not use any external knowledge. "
    "Be precise and concise. For numeric questions, provide the exact number with its unit. "
    "For yes/no questions, answer yes or no first. "
    "If the context does not contain enough information, say \"Not available in context\"."
)

QA_SYSTEM_PROMPT_CLOSED = (
    "You are a financial analyst with expertise in Indian companies and annual reports. "
    "Answer the following question about an Indian company's annual report using your knowledge. "
    "Be precise and concise. For numeric questions, provide the exact number with its unit. "
    "For yes/no questions, answer yes or no first."
)

QA_USER_TEMPLATE = """Context from the annual report:
---
{context}
---

Question: {question}

Answer:"""

QA_USER_TEMPLATE_CLOSED = """Question about an Indian company's annual report:

{question}

Answer:"""

QA_FEW_SHOT_PREFIX = """Here are some examples of questions and answers about Indian company annual reports:

{examples}

Now answer the following:
"""

FEW_SHOT_EXAMPLES = [
    {
        "question": "How many members does the Board of Directors have?",
        "answer": "12 directors",
    },
    {
        "question": "What was the total revenue for FY2025?",
        "answer": "₹ 45,320 crores",
    },
    {
        "question": "Did the company's net profit increase or decrease compared to the previous year?",
        "answer": "Increased by 14.3% to ₹ 8,240 crores",
    },
]


def _build_few_shot_prefix(examples: list[dict] | None = None) -> str:
    exs = examples or FEW_SHOT_EXAMPLES
    lines = []
    for ex in exs:
        lines.append(f"Q: {ex['question']}\nA: {ex['answer']}")
    return QA_FEW_SHOT_PREFIX.format(examples="\n\n".join(lines))


def _get_api_keys(env_name: str) -> list[str]:
    """Return all available keys for an env var, including numbered variants."""
    keys = []
    base = os.environ.get(env_name, "").strip()
    if base:
        keys.append(base)
    for i in range(1, 10):
        k = os.environ.get(f"{env_name}_{i}", "").strip()
        if k:
            keys.append(k)
    return keys


VALID_REGIMES = ("zero_shot", "few_shot", "closed_book", "few_shot_closed")


class ModelRunner:
    def __init__(
        self,
        config: ModelConfig,
        api_key: str | None = None,
        max_retries: int = 5,
        regime: str = "zero_shot",
        few_shot_examples: list[dict] | None = None,
    ):
        assert regime in VALID_REGIMES, f"regime must be one of {VALID_REGIMES}"
        self.config = config
        self.regime = regime
        self.few_shot_examples = few_shot_examples
        self._explicit_key = api_key
        self.max_retries = max_retries
        self._keys: list[str] = []
        self._key_idx: int = 0
        self._client: httpx.Client | None = None

    def _resolve_keys(self) -> list[str]:
        if self._explicit_key:
            return [self._explicit_key]
        return _get_api_keys(self.config.api_key_env)

    def _current_key(self) -> str | None:
        if not self._keys:
            self._keys = self._resolve_keys()
        if not self._keys:
            return None
        return self._keys[self._key_idx % len(self._keys)]

    def _rotate_key(self):
        self._key_idx += 1

    def _get_client(self) -> httpx.Client:
        key = self._current_key()
        # Re-create client if key changed or closed
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.config.api_base,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    def _rebuild_client(self):
        """Force rebuild with current (possibly rotated) key."""
        if self._client and not self._client.is_closed:
            self._client.close()
        self._client = None

    def _build_messages(self, question: str, context: str) -> list[dict]:
        closed = "closed" in self.regime
        few_shot = "few_shot" in self.regime

        system = QA_SYSTEM_PROMPT_CLOSED if closed else QA_SYSTEM_PROMPT

        if closed:
            user = QA_USER_TEMPLATE_CLOSED.format(question=question)
        else:
            user = QA_USER_TEMPLATE.format(context=context, question=question)

        if few_shot:
            prefix = _build_few_shot_prefix(self.few_shot_examples)
            user = prefix + user

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, question: str, context: str) -> GenerationResult:
        start = time.time()
        messages = self._build_messages(question, context)
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        backoff = 2.0
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                client = self._get_client()
                resp = client.post("/chat/completions", json=payload)

                if resp.status_code == 429:
                    wait = backoff * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    self._rotate_key()
                    self._rebuild_client()
                    continue

                resp.raise_for_status()
                data = resp.json()
                predicted = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                latency = (time.time() - start) * 1000
                return GenerationResult(
                    question_id="",
                    model_name=self.config.name,
                    question=question,
                    gold_answer="",
                    predicted_answer=predicted,
                    context=context,
                    latency_ms=round(latency, 2),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                if e.response.status_code in (500, 502, 503, 504):
                    time.sleep(backoff * (2 ** attempt))
                    continue
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(backoff * (2 ** attempt))

        latency = (time.time() - start) * 1000
        return GenerationResult(
            question_id="",
            model_name=self.config.name,
            question=question,
            gold_answer="",
            predicted_answer="",
            context=context,
            latency_ms=round(latency, 2),
            error=last_error,
        )

    def generate_batch(
        self,
        qa_records: list[QARecord],
        contexts: list[str],
        cache_path: Path | None = None,
    ) -> list[GenerationResult]:
        import json

        cache: dict[str, GenerationResult] = {}
        if cache_path and cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        cache[obj["question_id"]] = GenerationResult(**{
                            k: v for k, v in obj.items()
                            if k in GenerationResult.__dataclass_fields__
                        })

        results: list[GenerationResult] = []
        pending = [(qa, ctx) for qa, ctx in zip(qa_records, contexts)
                   if (qa.global_id or "") not in cache]
        cached_count = len(qa_records) - len(pending)

        desc = f"{self.config.name}"
        with tqdm(total=len(qa_records), initial=cached_count, desc=desc, unit="q") as pbar:
            # Emit cached results first (in order)
            result_map: dict[str, GenerationResult] = dict(cache)

            for qa, ctx in pending:
                qid = qa.global_id or f"{qa.company}_{qa.difficulty}_{hash(qa.question)}"
                result = self.generate(qa.question, ctx)
                result.question_id = qid
                result.gold_answer = qa.answer
                result_map[qid] = result

                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "a") as f:
                        f.write(json.dumps({
                            "question_id": result.question_id,
                            "model_name": result.model_name,
                            "question": result.question,
                            "gold_answer": result.gold_answer,
                            "predicted_answer": result.predicted_answer,
                            "context": result.context[:300],
                            "latency_ms": result.latency_ms,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "error": result.error,
                        }) + "\n")
                pbar.update(1)

        # Return in original order
        for qa in qa_records:
            qid = qa.global_id or f"{qa.company}_{qa.difficulty}_{hash(qa.question)}"
            results.append(result_map[qid])
        return results

    def close(self):
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
