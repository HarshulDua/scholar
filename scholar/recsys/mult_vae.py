"""Mult-VAE for collaborative filtering — built from scratch with PyTorch."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 200,
        hidden_dim: int = 600,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # Input dropout (p=0.5) is a core technique from the Mult-VAE paper —
        # forces the model to learn robust representations from partial observations.
        self.input_dropout = nn.Dropout(dropout_p)

        self.enc_fc1 = nn.Linear(vocab_size, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, latent_dim * 2)

        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_dropout(x)
        h = torch.tanh(self.enc_fc1(x))
        out = self.enc_fc2(h)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.dec_fc1(z))
        return F.log_softmax(self.dec_fc2(h), dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class MultVAELoss(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = -torch.mean(torch.sum(recon_x * x, dim=-1))
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return recon_loss + self.beta * kl_loss


def _is_near_init(model: MultVAE) -> bool:
    """Heuristic: if mean abs weight in decoder is very small, model is near init."""
    with torch.no_grad():
        w = model.dec_fc2.weight
        return bool(w.abs().mean().item() < 0.05)


def _load_trained_model() -> Optional["MultVAE"]:
    """Load saved Mult-VAE checkpoint if it exists."""
    from pathlib import Path

    model_path = Path("./data/mult_vae.pt")
    if not model_path.exists():
        return None
    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        m = MultVAE(
            vocab_size=checkpoint["vocab_size"],
            latent_dim=checkpoint["latent_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout_p=checkpoint.get("dropout_p", 0.5),
        )
        m.load_state_dict(checkpoint["model_state"])
        m.eval()
        return m
    except Exception:
        return None


def get_user_embedding(
    history_paper_ids: list[str],
    all_paper_ids: list[str],
    model: Optional[MultVAE],
) -> np.ndarray:
    """Return a latent user embedding for the given history.

    Falls back to averaging sentence-BERT embeddings if model is untrained.
    Returns a zeros vector of shape (latent_dim,) or (384,) depending on fallback.
    """
    if not history_paper_ids:
        latent_dim = model.latent_dim if model is not None else 200
        return np.zeros(latent_dim, dtype=np.float32)

    if model is None:
        model = _load_trained_model()

    if model is None or _is_near_init(model):
        return _fallback_sentence_bert(history_paper_ids)

    # Use saved vocab if available, otherwise fall back to passed-in all_paper_ids
    from pathlib import Path

    vocab_path = Path("./data/vae_vocab.json")
    if vocab_path.exists():
        try:
            import json

            with open(vocab_path) as f:
                all_paper_ids = json.load(f)
        except Exception:
            pass

    id_to_idx = {pid: i for i, pid in enumerate(all_paper_ids)}
    vocab_size = len(all_paper_ids)

    if model.vocab_size != vocab_size:
        return _fallback_sentence_bert(history_paper_ids)

    x = torch.zeros(1, vocab_size, dtype=torch.float32)
    for pid in history_paper_ids:
        if pid in id_to_idx:
            x[0, id_to_idx[pid]] = 1.0

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.squeeze(0).numpy()


def _fallback_sentence_bert(paper_ids: list[str]) -> np.ndarray:
    """Average sentence-BERT embeddings of history papers as a cold-start proxy."""
    try:
        from sentence_transformers import SentenceTransformer
        from sqlmodel import Session, create_engine, select

        from scholar.config import settings
        from scholar.db.models import Paper

        engine = create_engine(settings.database_url_sync, echo=False)
        with Session(engine) as session:
            papers = session.exec(
                select(Paper.abstract).where(Paper.id.in_(paper_ids))  # type: ignore[attr-defined]
            ).all()

        if not papers:
            return np.zeros(384, dtype=np.float32)

        abstracts = list(papers)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(abstracts, convert_to_numpy=True)
        return embeddings.mean(axis=0).astype(np.float32)
    except Exception:
        return np.zeros(384, dtype=np.float32)
