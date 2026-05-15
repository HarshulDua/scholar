"""Train Mult-VAE on user-paper interaction data from user_history table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import psycopg2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scholar.recsys.mult_vae import MultVAE, MultVAELoss

DATA_DIR = Path("./data")
MODEL_PATH = DATA_DIR / "mult_vae.pt"
VOCAB_PATH = DATA_DIR / "vae_vocab.json"


def _load_interactions() -> list[tuple[str, str]]:
    from scholar.config import settings

    conn = psycopg2.connect(str(settings.database_url_sync))
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, paper_id FROM user_history")
            rows = cur.fetchall()
    finally:
        conn.close()
    return [(str(uid), str(pid)) for uid, pid in rows]


def _build_interaction_matrix(
    interactions: list[tuple[str, str]],
) -> tuple[np.ndarray, list[str], list[str]]:
    user_ids = sorted({uid for uid, _ in interactions})
    paper_ids = sorted({pid for _, pid in interactions})

    u_idx = {uid: i for i, uid in enumerate(user_ids)}
    p_idx = {pid: i for i, pid in enumerate(paper_ids)}

    matrix = np.zeros((len(user_ids), len(paper_ids)), dtype=np.float32)
    for uid, pid in interactions:
        matrix[u_idx[uid], p_idx[pid]] = 1.0

    return matrix, user_ids, paper_ids


def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    dcg = sum(float(labels[i]) / np.log2(rank + 2) for rank, i in enumerate(top_k))
    ideal = sorted(labels.tolist(), reverse=True)[:k]
    idcg = sum(v / np.log2(rank + 2) for rank, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _per_user_split(
    matrix: np.ndarray, rng: np.random.RandomState, val_frac: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Split each user's interactions into train/val by holding out val_frac items.

    Users with <2 interactions keep all items in train (nothing to hold out).
    Returns (X_train, X_val) both shape [n_users, vocab_size].
    """
    X_train = matrix.copy()
    X_val = np.zeros_like(matrix)

    for u in range(matrix.shape[0]):
        item_indices = np.where(matrix[u] > 0)[0]
        if len(item_indices) < 2:
            continue
        n_val = max(1, int(len(item_indices) * val_frac))
        val_items = rng.choice(item_indices, size=n_val, replace=False)
        X_train[u, val_items] = 0.0
        X_val[u, val_items] = 1.0

    return X_train, X_val


def train(
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    latent_dim: int = 200,
    hidden_dim: int = 600,
    dropout_p: float = 0.5,
    val_every: int = 5,
    patience: int = 20,
) -> None:
    print("Loading interactions from DB...")
    interactions = _load_interactions()
    if not interactions:
        print("No interactions found. Run citeulike_loader first.")
        return

    print(f"Loaded {len(interactions)} interactions.")
    matrix, user_ids, paper_ids = _build_interaction_matrix(interactions)
    n_users, vocab_size = matrix.shape
    print(f"Interaction matrix: {n_users} users x {vocab_size} papers")

    rng = np.random.RandomState(42)

    # Per-user interaction split: hold out 20% of each user's items for validation.
    # This is the standard CF evaluation (vs user-level split which is trivially easy
    # when the item space is tiny).
    X_train, X_val = _per_user_split(matrix, rng, val_frac=0.2)

    # Only evaluate on users who have held-out items
    eval_mask = X_val.sum(axis=1) > 0
    print(f"Users with held-out val items: {eval_mask.sum()}/{n_users}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t), batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    model = MultVAE(
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_dev = X_train_t.to(device)
    best_ndcg = -1.0
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        beta = min(0.2, epoch / 20 * 0.2)
        criterion = MultVAELoss(beta=beta)

        model.train()
        total_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch)
            loss = criterion(recon_x, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        if epoch % val_every == 0:
            model.eval()
            with torch.no_grad():
                # Encode using train interactions, score against val held-out items
                recon_all, _, _ = model(X_train_dev)
                scores_np = recon_all.cpu().numpy()

            labels_np = X_val_t.numpy()
            ndcgs = [
                _ndcg_at_k(scores_np[i], labels_np[i])
                for i in range(n_users)
                if eval_mask[i]
            ]
            avg_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
            print(
                f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                f"val NDCG@10={avg_ndcg:.4f} | beta={beta:.3f}"
            )

            if avg_ndcg > best_ndcg:
                best_ndcg = avg_ndcg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} evals).")
                    break
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | beta={beta:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "dropout_p": dropout_p,
        },
        MODEL_PATH,
    )
    with open(VOCAB_PATH, "w") as f:
        json.dump(paper_ids, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Vocab saved to {VOCAB_PATH}")
    print(f"Best val NDCG@10: {best_ndcg:.4f}")


if __name__ == "__main__":
    train()
