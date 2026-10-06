"""Settings read from the environment (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

TOKEN_FACTORY_URL = "https://api.tokenfactory.nebius.com/v1/"


@dataclass
class ModelRouting:
    """Which model handles which job. Override any of them with env vars.

    Ultra: few calls that need deep reasoning (planning, next-round strategy).
    Super: medium work once per round (failure analysis write-up).
    Nano:  many cheap calls (per-failure captions).
    """

    ultra: str = field(default_factory=lambda: os.getenv("RG_MODEL_ULTRA", "nvidia/Nemotron-3-Ultra-550b-a55b"))
    super: str = field(default_factory=lambda: os.getenv("RG_MODEL_SUPER", "nvidia/nemotron-3-super-120b-a12b"))
    nano: str = field(default_factory=lambda: os.getenv("RG_MODEL_NANO", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"))


@dataclass
class Settings:
    nebius_api_key: str | None = field(default_factory=lambda: os.getenv("NEBIUS_API_KEY"))
    token_factory_url: str = field(default_factory=lambda: os.getenv("RG_TOKEN_FACTORY_URL", TOKEN_FACTORY_URL))
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    # Cosmos Reason 2 served by vLLM on a Nebius Serverless Endpoint (OpenAI-compatible)
    cosmos_reason_url: str | None = field(default_factory=lambda: os.getenv("RG_COSMOS_REASON_URL"))
    cosmos_reason_key: str | None = field(default_factory=lambda: os.getenv("RG_COSMOS_REASON_KEY"))
    cosmos_reason_model: str = field(default_factory=lambda: os.getenv("RG_COSMOS_REASON_MODEL", "nvidia/Cosmos-Reason2-8B"))
    models: ModelRouting = field(default_factory=ModelRouting)
    offline: bool = False

    @property
    def llm_enabled(self) -> bool:
        return bool(self.nebius_api_key) and not self.offline

    @property
    def judge_enabled(self) -> bool:
        return bool(self.cosmos_reason_url) and not self.offline

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.tavily_api_key) and not self.offline
