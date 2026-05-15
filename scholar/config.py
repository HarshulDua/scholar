from __future__ import annotations

from pathlib import Path
from typing import Optional

# Windows Python 3.13: pyarrow C-extension must initialise before torch DLLs.
# Any module that imports config.py (server, evals, scripts) gets this ordering.
try:
    import pyarrow as _pa  # noqa: F401
except ImportError:
    pass

import torch
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = Field(
        default="postgresql+asyncpg://scholar:changeme@db:5432/scholar",
        alias="DATABASE_URL",
    )
    database_url_sync: str = Field(
        default="postgresql://scholar:changeme@db:5432/scholar",
        alias="DATABASE_URL_SYNC",
    )
    postgres_password: str = Field(default="changeme", alias="POSTGRES_PASSWORD")

    arxiv_data_path: Path = Field(
        default=Path("./data/arxiv-metadata-oai-snapshot.json"),
        alias="ARXIV_DATA_PATH",
    )
    faiss_index_path: Path = Field(
        default=Path("./data/faiss.index"),
        alias="FAISS_INDEX_PATH",
    )
    embeddings_path: Path = Field(
        default=Path("./data/embeddings.npy"),
        alias="EMBEDDINGS_PATH",
    )
    hf_home: Path = Field(default=Path("./data/hf_cache"), alias="HF_HOME")
    max_papers: int = Field(default=50000, alias="MAX_PAPERS")
    device: str = Field(default="cuda", alias="DEVICE")
    lora_adapter_path: Optional[Path] = Field(
        default=Path("./data/lora_adapter"),
        alias="LORA_ADAPTER_PATH",
    )
    allow_cpu_generation: bool = Field(default=False, alias="ALLOW_CPU_GENERATION")
    warmup_generation: bool = Field(default=False, alias="WARMUP_GENERATION")
    warmup_retrieval: bool = Field(default=True, alias="WARMUP_RETRIEVAL")

    @property
    def effective_device(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def generation_available(self) -> bool:
        """Phi-3-mini 4-bit NF4 needs ~2GB VRAM on CUDA, ~7GB RAM on CPU.
        CPU generation is very slow (~30s/token). Enable with ALLOW_CPU_GENERATION=true."""
        if torch.cuda.is_available():
            return True
        return self.allow_cpu_generation

    model_config = {"env_file": ".env", "populate_by_name": True, "extra": "ignore"}


settings = Settings()
