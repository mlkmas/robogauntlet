import os

from .base import SimBackend
from .isaac_arena import IsaacArenaBackend
from .mock import MockBackend


def get_backend(name: str) -> SimBackend:
    if name == "mock":
        return MockBackend(render=os.getenv("RG_MOCK_RENDER", "1") != "0")
    if name == "isaac_arena":
        return IsaacArenaBackend()
    raise ValueError(f"Unknown backend {name!r}; choose 'mock' or 'isaac_arena'")


__all__ = ["SimBackend", "MockBackend", "IsaacArenaBackend", "get_backend"]
