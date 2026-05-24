import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from finbharat.data.loader import QARecord


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


PREDEFINED_MODELS = {
    "qwen3-8b": ModelConfig(
        name="Qwen3-8B-Instruct",
        model_id="qwen/qwen3-8b",
        provider="nim",
        api_key_env="NVIDIA_API_KEY",
    ),
    "llama3.1-8b": ModelConfig(
        name="Llama-3.1-8B-Instruct",
        model_id="meta/llama-3.1-8b-instruct",
        provider="nim",
        api_key_env="NVIDIA_API_KEY",
    ),
    "qwen3-72b": ModelConfig(
        name="Qwen3-72B-Instruct",
        model_id="qwen/qwen3-72b-instruct",
        provider="nim",
        api_key_env="NVIDIA_API_KEY",
    ),
    "llama3.3-70b": ModelConfig(
        name="Llama-3.3-70B-Instruct",
        model_id="meta/llama-3.3-70b-instruct",
        provider="nim",
        api_key_env="NVIDIA_API_KEY",
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


QA_SYSTEM_PROMPT = """You are a financial analyst answering questions about Indian company annual reports.
Answer the question based ONLY on the provided context. Do not use any external knowledge.
Be precise and concise. For numeric questions, provide the exact number.
For yes/no questions, answer yes or no.
If the context does not contain enough information, say "Not available in context"."""


QA_USER_TEMPLATE = """Context from the annual report:
---
{context}
---

Question: {question}

Answer:"""


class ModelRunner:
    def __init__(self, config: ModelConfig, api_key: str | None = None):
        self.config = config
        self._api_key = api_key
        self._client: httpx.Client | None = None

    @property
    def api_key(self) -> str | None:
        if self._api_key:
            return self._api_key
        import os
        return os.environ.get(self.config.api_key_env)

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.config.api_base,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    def generate(self, question: str, context: str) -> GenerationResult:
        start = time.time()
        user_prompt = QA_USER_TEMPLATE.format(context=context, question=question)
        try:
            client = self._get_client()
            payload = {
                "model": self.config.model_id,
                "messages": [
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            resp = client.post("/chat/completions", json=payload)
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
        except Exception as e:
            latency = (time.time() - start) * 1000
            return GenerationResult(
                question_id="",
                model_name=self.config.name,
                question=question,
                gold_answer="",
                predicted_answer="",
                context=context,
                latency_ms=round(latency, 2),
                error=str(e),
            )

    def generate_batch(
        self,
        qa_records: list[QARecord],
        contexts: list[str],
        cache_path: Path | None = None,
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        cache: dict[str, GenerationResult] = {}

        if cache_path and cache_path.exists():
            import json
            with open(cache_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        cache[obj["question_id"]] = GenerationResult(**{
                            k: v for k, v in obj.items()
                            if k in GenerationResult.__dataclass_fields__
                        })

        for qa, ctx in zip(qa_records, contexts):
            qid = qa.global_id or f"{qa.company}_{qa.difficulty}_{hash(qa.question)}"
            if qid in cache:
                cached = cache[qid]
                cached.question_id = qid
                results.append(cached)
                continue
            result = self.generate(qa.question, ctx)
            result.question_id = qid
            result.gold_answer = qa.answer
            results.append(result)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "a") as f:
                    import json
                    f.write(json.dumps({
                        "question_id": result.question_id,
                        "model_name": result.model_name,
                        "question": result.question,
                        "gold_answer": result.gold_answer,
                        "predicted_answer": result.predicted_answer,
                        "context": result.context[:200],
                        "latency_ms": result.latency_ms,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "error": result.error,
                    }) + "\n")
        return results

    def close(self):
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
