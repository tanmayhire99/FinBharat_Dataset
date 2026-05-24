import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from qa_pipeline_v2.prompts import get_system_user_messages

logger = logging.getLogger(__name__)


@dataclass
class NIMConfig:
    api_key: str = "[ENCRYPTION_KEY]"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    model: str = "meta/llama3-70b-instruct"
    temperature: float = 0.2
    max_tokens: int = 16384
    timeout: int = 60
    requests_per_minute: int = 30


class NIMClient:
    def __init__(self, cfg: NIMConfig) -> None:
        self.cfg = cfg
        self.api_key = cfg.api_key or os.getenv(cfg.api_key_env, "")
        if not self.api_key:
            raise ValueError(
                "Missing API key. Provide it manually in NIMConfig(api_key='YOUR_KEY_HERE') "
                f"or set the environment variable {cfg.api_key_env}."
            )
        self.client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.api_key,
            timeout=self.cfg.timeout * 5,
            max_retries=3,
        )
        self.last_call = 0.0
        self.min_interval = 60.0 / max(cfg.requests_per_minute, 1)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        self._rate_limit()
        logger.info(f"Sending chat completion request to model: {self.cfg.model}")

        # Exponential backoff retry for transient failures
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    extra_body={"chat_template_kwargs": {"thinking": False}},
                    stream=False,
                )
                content = completion.choices[0].message.content
                logger.info(f"Received response successfully. Length: {len(content) if content else 0} chars.")
                return content
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_attempts} failed: {e}")
                if attempt == max_attempts:
                    logger.error(f"Error during OpenAI API call after retries: {e}")
                    raise
                time.sleep(2 ** attempt)  # exponential backoff

    def _rate_limit(self) -> None:
        now = time.time()
        wait = self.min_interval - (now - self.last_call)
        if wait > 0:
            time.sleep(wait)
        self.last_call = time.time()


def generate_qa_pairs(
    client: NIMClient,
    chunk_text: str,
    sector: str,
    company: str,
    year: str,
    section: str,
    max_questions: int,
    difficulty: str = "medium",
) -> List[Dict[str, Any]]:
    logger.info(f"Preparing to generate QA [{difficulty}]. Target: {sector}/{company}/{year}/{section}")

    system_msg, user_msg = get_system_user_messages(
        difficulty=difficulty,
        sector=sector,
        company=company,
        year=year,
        section=section,
        chunk_text=chunk_text,
        max_questions=max_questions,
    )

    logger.debug(f"System Message length: {len(system_msg)}")
    logger.debug(f"User Message length: {len(user_msg)}")

    content = client.chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    )

    logger.info("Model returned response. Parsing JSON answer.")
    logger.debug(f"Raw Model Output:\n{content}")

    pairs = _parse_json_array(content)
    logger.info(f"Successfully generated {len(pairs)} QA pairs from this chunk.")
    return pairs


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    # Try direct JSON first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _normalize_pairs(data)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array inside markdown/code blocks
    # Pattern: look for the first [ ... ] that seems balanced
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return _normalize_pairs(data)
    return []


def _normalize_pairs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not question or not answer:
            continue
        pairs.append({"question": question, "answer": answer, "evidence": evidence})
    return pairs