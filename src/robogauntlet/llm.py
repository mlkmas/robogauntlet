"""Thin client for Nebius Token Factory with per-model usage accounting."""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from typing import Any

from .config import Settings


class UsageLedger:
    """Counts calls and tokens per model, shown in the report."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.prompt_tokens: dict[str, int] = defaultdict(int)
        self.completion_tokens: dict[str, int] = defaultdict(int)
        self.purpose: dict[str, set[str]] = defaultdict(set)

    def record(self, model: str, purpose: str, usage: Any | None) -> None:
        self.calls[model] += 1
        self.purpose[model].add(purpose)
        if usage is not None:
            self.prompt_tokens[model] += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens[model] += getattr(usage, "completion_tokens", 0) or 0

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            m: {
                "calls": self.calls[m],
                "prompt_tokens": self.prompt_tokens[m],
                "completion_tokens": self.completion_tokens[m],
                "purposes": sorted(self.purpose[m]),
            }
            for m in self.calls
        }


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply that may include reasoning or code fences."""
    text = text.strip()
    # Reasoning models may emit a <think> block first.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise
        end = max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[start : end + 1])


class TokenFactory:
    """OpenAI-compatible calls to Nemotron models on Nebius Token Factory."""

    def __init__(self, settings: Settings, ledger: UsageLedger | None = None) -> None:
        self.settings = settings
        self.ledger = ledger or UsageLedger()
        self._client = None
        if settings.llm_enabled:
            from openai import OpenAI

            self._client = OpenAI(base_url=settings.token_factory_url, api_key=settings.nebius_api_key)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        purpose: str,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        retries: int = 2,
    ) -> Any | None:
        """Ask for a JSON answer. Returns None when offline or after repeated failure."""
        if not self.enabled:
            return None
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self.ledger.record(model, purpose, resp.usage)
                return extract_json(resp.choices[0].message.content or "")
            except Exception as err:  # network errors, bad JSON
                last_err = err
                time.sleep(1.5 * (attempt + 1))
        print(f"[token-factory] {purpose} failed on {model}: {last_err}; using offline fallback")
        return None
