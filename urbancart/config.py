import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RUNTIME = ROOT / "data" / "runtime"


@dataclass(frozen=True)
class Settings:
    mode: str = "local"
    chat_model: str = "google/gemini-3.1-flash-lite"
    embedding_model: str = "openai/text-embedding-3-small"
    key: str = field(default="", repr=False)
    runtime: Path = RUNTIME

    @classmethod
    def from_env(cls, mode=None):
        load_dotenv(ROOT / ".env")
        settings = cls(
            mode=mode or os.getenv("URBANCART_MODE", "openrouter").lower(),
            chat_model=os.getenv("URBANCART_CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("URBANCART_EMBED_MODEL", cls.embedding_model),
            key=os.getenv("OPENROUTER_API_KEY", ""),
        )
        if settings.mode not in {"local", "openrouter"}:
            raise ValueError("URBANCART_MODE must be local or openrouter.")
        if settings.mode == "openrouter" and not settings.key:
            raise ValueError("Set OPENROUTER_API_KEY to use OpenRouter mode.")
        return settings
