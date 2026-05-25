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
    provider: str = "nim"         # "nim" | "openai" | "vllm"
    api_base: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    use_completion: bool = False         # True for models like FinMA that use /completions not /chat
    max_context_chars: int = 0          # Truncate context to N chars before sending. 0 = no limit.
    use_max_completion_tokens: bool = False  # GPT-5+ uses max_completion_tokens not max_tokens


ALPACA_COMPLETION_TEMPLATE = """### Instruction:
{system}
{user}

### Response:"""

ALPACA_COMPLETION_TEMPLATE_NOSYS = """### Instruction:
{user}

### Response:"""


def make_vllm_config(
    model_id: str,
    name: str | None = None,
    host: str = "localhost",
    port: int = 8000,
    max_tokens: int = 512,
    use_completion: bool = False,
    max_context_chars: int = 0,
) -> ModelConfig:
    """
    Create a ModelConfig for a locally hosted vLLM server.

    The vLLM OpenAI-compatible API (--enable-lora, LoRA models, base models)
    is identical in format to NIM — just different base URL and no auth key.

    Args:
        model_id:   The model name as served by vLLM.
                    - For base model:  "meta-llama/Meta-Llama-3-8B"
                    - For LoRA:        "fingpt"  (the --lora-modules name)
        name:       Human-readable label for logs. Defaults to model_id.
        host:       Host where vLLM is running (default: localhost).
                    Use "localhost" if you have an SSH tunnel:
                      ssh -L 8000:localhost:8000 user@remote
        port:       Port vLLM is listening on (default: 8000).
        max_tokens: Max tokens to generate per call.

    Example:
        cfg = make_vllm_config("fingpt")
        cfg = make_vllm_config("meta-llama/Meta-Llama-3-8B", host="10.0.0.5")
    """
    return ModelConfig(
        name=name or model_id.split("/")[-1],
        model_id=model_id,
        provider="vllm",
        api_base=f"http://{host}:{port}/v1",
        api_key_env="VLLM_API_KEY",
        max_tokens=max_tokens,
        use_completion=use_completion,
        max_context_chars=max_context_chars,
    )


PREDEFINED_MODELS: dict[str, ModelConfig] = {
    # ── Small / Efficient (≤8B) ── verified ✅ ──────────────────────────────
    "llama3.1-8b": ModelConfig(
        name="Llama-3.1-8B-Instruct",
        model_id="meta/llama-3.1-8b-instruct",
    ),
    "llama3.2-3b": ModelConfig(
        name="Llama-3.2-3B-Instruct",
        model_id="meta/llama-3.2-3b-instruct",
    ),
    "nemotron-nano-8b": ModelConfig(
        name="Nemotron-Nano-8B",
        model_id="nvidia/llama-3.1-nemotron-nano-8b-v1",
    ),
    "phi4-mini": ModelConfig(
        name="Phi-4-Mini-Instruct",
        model_id="microsoft/phi-4-mini-instruct",
    ),
    "gpt-oss-20b": ModelConfig(
        name="GPT-OSS-20B",
        model_id="openai/gpt-oss-20b",
    ),

    # ── Medium (17–49B) ── verified ✅ ────────────────────────────────────
    "llama4-maverick": ModelConfig(
        name="Llama-4-Maverick-17B-128E",
        model_id="meta/llama-4-maverick-17b-128e-instruct",
        max_tokens=1024,
    ),
    "nemotron-super-49b": ModelConfig(
        name="Nemotron-Super-49B",
        model_id="nvidia/llama-3.3-nemotron-super-49b-v1",
    ),
    "gemma4-31b": ModelConfig(
        name="Gemma-4-31B-Instruct",
        model_id="google/gemma-4-31b-it",
    ),
    "mistral-nemotron": ModelConfig(
        name="Mistral-Nemotron",
        model_id="mistralai/mistral-nemotron",
    ),

    # ── Large (70B–120B) ── verified ✅ ──────────────────────────────────
    "llama3.3-70b": ModelConfig(
        name="Llama-3.3-70B-Instruct",
        model_id="meta/llama-3.3-70b-instruct",
    ),
    "nemotron-120b": ModelConfig(
        name="Nemotron-3-Super-120B",
        model_id="nvidia/nemotron-3-super-120b-a12b",
    ),
    "gpt-oss-120b": ModelConfig(
        name="GPT-OSS-120B",
        model_id="openai/gpt-oss-120b",
    ),
    "mistral-small-119b": ModelConfig(
        name="Mistral-Small-4-119B",
        model_id="mistralai/mistral-small-4-119b-2603",
    ),

    # ── Very Large (120B+) / MoE ── verified ✅ ──────────────────────────
    "qwen3.5-122b": ModelConfig(
        name="Qwen3.5-122B-A10B",
        model_id="qwen/qwen3.5-122b-a10b",
        max_tokens=1024,
    ),
    "qwen3.5-397b": ModelConfig(
        name="Qwen3.5-397B-A17B",
        model_id="qwen/qwen3.5-397b-a17b",
        max_tokens=1024,
    ),
    "mistral-large-675b": ModelConfig(
        name="Mistral-Large-3-675B",
        model_id="mistralai/mistral-large-3-675b-instruct-2512",
        max_tokens=1024,
    ),

    # ── DeepSeek ── verified ✅ ────────────────────────────────────────────
    "deepseek-v4-flash": ModelConfig(
        name="DeepSeek-V4-Flash",
        model_id="deepseek-ai/deepseek-v4-flash",
        max_tokens=1024,
    ),
    "deepseek-v4-pro": ModelConfig(
        name="DeepSeek-V4-Pro",
        model_id="deepseek-ai/deepseek-v4-pro",
        max_tokens=1024,
    ),

    # ── Financial domain ── (account-specific, may 404) ───────────────────
    "palmyra-fin-70b": ModelConfig(
        name="Palmyra-Fin-70B-32K",
        model_id="writer/palmyra-fin-70b-32k",
        max_tokens=1024,
    ),

    # ── OpenAI (requires OPENAI_API_KEY) ─────────────────────────────────
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        model_id="gpt-4o",
        provider="openai",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),

    # ── Lightning AI (requires LIGHTNING_API_KEY) ─────────────────────────
    # Trailing slash required: httpx merges "base/v1" + "/path" → "base/path" (drops /v1)
    # With trailing slash: "base/v1/" + "chat/completions" → "base/v1/chat/completions" ✓
    "gpt-5-nano": ModelConfig(
        name="GPT-5-Nano",
        model_id="openai/gpt-5-nano",
        provider="lightning",
        api_base="https://lightning.ai/api/v1/",
        api_key_env="LIGHTNING_API_KEY",
        max_tokens=512,
        use_max_completion_tokens=True,  # GPT-5 API uses max_completion_tokens
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
    "Answer the following question using only your pre-training knowledge about this company. "
    "You do NOT have access to the actual annual report document. "
    "Be precise and concise. For numeric questions, provide the exact number with its unit. "
    "For yes/no questions, answer yes or no first. "
    "If you are not confident, say \"Not available in memory\"."
)

QA_USER_TEMPLATE = """Context from the annual report:
---
{context}
---

Question: {question}

Answer:"""

QA_USER_TEMPLATE_CLOSED = """Company information:
{metadata}

Question (answer from your pre-training knowledge — no document provided):
{question}

Answer:"""

QA_USER_TEMPLATE_CLOSED_NO_META = """Question about an Indian company's annual report:

{question}

Answer:"""

QA_FEW_SHOT_PREFIX = """Here are some examples of questions and answers about Indian company annual reports:

{examples}

Now answer the following:
"""

FEW_SHOT_EXAMPLES = [
    # Text Only — short factual answer (most common in easy tier)
    {
        "question": "How many directors are on the Board as of March 31, 2025?",
        "answer": "Thirteen (13) Directors",
    },
    # Table Only — bare number with unit (most common in easy/medium tier)
    {
        "question": "What was the Profit After Tax for FY2025?",
        "answer": "₹ 8,240 crore",
    },
    # Numerical Calculation — directional with magnitude (hard/multihop tier)
    {
        "question": "How did net revenue change year-over-year in FY2025?",
        "answer": "Net revenue increased by 14.3% to ₹ 2,479 crore from ₹ 2,168 crore in FY2024.",
    },
]


_SECTOR_LABELS = {
    "Private_Sector_Bank": "Private Sector Banking",
    "Public_Sector_Bank":  "Public Sector Banking",
    "Information_Technology": "Information Technology",
    "Pharmaceutical": "Pharmaceuticals",
    "Automobile": "Automotive",
    "Fast_Moving_Consumer_Goods": "FMCG",
}


def _format_company_metadata(company: str, sector: str, year: str) -> str:
    """Format QARecord metadata into a concise preamble for the closed-book prompt.

    Gives the model enough context to retrieve from pre-training memory
    without providing any document content.

    Example output:
        Company:     HDFC Bank Limited
        Sector:      Private Sector Banking
        Fiscal Year: FY2025 (April 2024 – March 2025)
        Country:     India (BSE/NSE listed)
    """
    # Make company name human-readable
    company_display = company.replace("_", " ")
    sector_display  = _SECTOR_LABELS.get(sector, sector.replace("_", " "))
    year_display    = year  # e.g. "FY2025"

    # Add April–March clarification for Indian fiscal year
    if year_display.startswith("FY") and len(year_display) == 6:
        cal = year_display[2:]   # "2025"
        try:
            prev = str(int(cal) - 1)  # "2024"
            year_display = f"{year_display} (April {prev} – March {cal})"
        except ValueError:
            pass

    return (
        f"Company:     {company_display}\n"
        f"Sector:      {sector_display}\n"
        f"Fiscal Year: {year_display}\n"
        f"Country:     India (BSE/NSE listed, SEBI regulated)"
    )


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
        key = self._current_key() or "EMPTY"   # vLLM accepts any non-empty key
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.config.api_base,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                # vLLM on local GPU can be slow for large models — longer timeout
                timeout=300.0 if self.config.provider == "vllm" else 120.0,
            )
        return self._client

    def _rebuild_client(self):
        """Force rebuild with current (possibly rotated) key."""
        if self._client and not self._client.is_closed:
            self._client.close()
        self._client = None

    def _build_messages(self, question: str, context: str, metadata: str = "") -> list[dict]:
        closed = "closed" in self.regime
        few_shot = "few_shot" in self.regime

        system = QA_SYSTEM_PROMPT_CLOSED if closed else QA_SYSTEM_PROMPT

        if closed:
            if metadata:
                user = QA_USER_TEMPLATE_CLOSED.format(
                    metadata=metadata, question=question
                )
            else:
                # Fallback: no metadata provided (old behaviour)
                user = QA_USER_TEMPLATE_CLOSED_NO_META.format(question=question)
        else:
            user = QA_USER_TEMPLATE.format(context=context, question=question)

        if few_shot:
            prefix = _build_few_shot_prefix(self.few_shot_examples)
            user = prefix + user

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_completion_prompt(self, question: str, context: str, metadata: str = "") -> str:
        """Build Alpaca-style completion prompt for models like FinMA.
        Truncates context to max_context_chars if set (needed for 2k-context models).
        """
        closed = "closed" in self.regime

        # Truncate context for small context window models (e.g. FinMA 2048 tokens)
        ctx = context
        if self.config.max_context_chars > 0 and len(ctx) > self.config.max_context_chars:
            ctx = ctx[:self.config.max_context_chars] + "\n[context truncated]"

        if closed and metadata:
            user_content = QA_USER_TEMPLATE_CLOSED.format(metadata=metadata, question=question)
        elif closed:
            user_content = QA_USER_TEMPLATE_CLOSED_NO_META.format(question=question)
        else:
            user_content = QA_USER_TEMPLATE.format(context=ctx, question=question)
        system = QA_SYSTEM_PROMPT_CLOSED if closed else QA_SYSTEM_PROMPT
        return ALPACA_COMPLETION_TEMPLATE.format(system=system, user=user_content)

    def generate(self, question: str, context: str, metadata: str = "") -> GenerationResult:
        start = time.time()

        if self.config.use_completion:
            # Models like FinMA: use /completions endpoint with Alpaca format
            prompt = self._build_completion_prompt(question, context, metadata)
            payload = {
                "model": self.config.model_id,
                "prompt": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stop": ["###", "\n\n\n"],
            }
            endpoint = "/completions"
        else:
            messages = self._build_messages(question, context, metadata)
            # GPT-5+ uses max_completion_tokens; older models use max_tokens
            tok_key = "max_completion_tokens" if self.config.use_max_completion_tokens else "max_tokens"
            payload = {
                "model": self.config.model_id,
                "messages": messages,
                tok_key: self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            endpoint = "/chat/completions"

        backoff = 2.0
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                client = self._get_client()
                # If base_url has trailing slash, use relative path (no leading /)
                # e.g. "https://lightning.ai/api/v1/" + "chat/completions" → correct
                # e.g. "https://nim.api/v1" + "/chat/completions" → correct (httpx keeps /v1)
                ep = endpoint.lstrip("/") if self.config.api_base.endswith("/") else endpoint
                resp = client.post(ep, json=payload)

                if resp.status_code == 429:
                    wait = backoff * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    self._rotate_key()
                    self._rebuild_client()
                    continue

                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                # /completions returns "text", /chat/completions returns "message.content"
                if self.config.use_completion:
                    predicted = (choice.get("text") or "").strip()
                else:
                    predicted = (choice.get("message", {}).get("content") or "").strip()
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
                # For closed-book regime, build metadata preamble from QA record
                meta = ""
                if "closed" in self.regime:
                    meta = _format_company_metadata(qa.company, qa.sector, qa.year)
                result = self.generate(qa.question, ctx, metadata=meta)
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
