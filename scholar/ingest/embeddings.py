"""Build sentence-transformer embeddings and FAISS HNSW index."""
from __future__ import annotations

import argparse
import asyncio
import json

import faiss
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from scholar.config import settings

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 512
EMBED_DIM = 384


def _get_conn():
    return psycopg2.connect(str(settings.database_url_sync))


def _load_papers_batched() -> tuple[list[str], list[str]]:
    paper_ids: list[str] = []
    abstracts: list[str] = []
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, abstract FROM papers ORDER BY id")
            while True:
                rows = cur.fetchmany(BATCH_SIZE)
                if not rows:
                    break
                for pid, abstract in rows:
                    paper_ids.append(pid)
                    abstracts.append(abstract)
    finally:
        conn.close()
    return paper_ids, abstracts


def _update_embeddings_in_db(paper_ids: list[str], embeddings: np.ndarray) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            batch_size = 1000
            for start in tqdm(range(0, len(paper_ids), batch_size), desc="Updating DB embeddings"):
                batch_ids = paper_ids[start : start + batch_size]
                batch_embs = embeddings[start : start + batch_size]
                rows = [(emb.tolist(), pid) for pid, emb in zip(batch_ids, batch_embs)]
                cur.executemany(
                    "UPDATE papers SET embedding = %s::vector WHERE id = %s",
                    rows,
                )
            conn.commit()
    finally:
        conn.close()


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    index = faiss.IndexHNSWFlat(EMBED_DIM, 16)
    index.hnsw.efConstruction = 200
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    index.add(normed.astype(np.float32))
    return index


async def main(skip_db_update: bool = False) -> None:
    print("Loading paper IDs and abstracts from DB ...")
    paper_ids, abstracts = _load_papers_batched()
    print(f"Loaded {len(paper_ids)} papers.")

    if not paper_ids:
        print("No papers found. Run arxiv_loader first.")
        return

    print(f"Encoding with {EMBED_MODEL} on device: {settings.effective_device} ...")
    model = SentenceTransformer(EMBED_MODEL, device=settings.effective_device)
    embeddings = model.encode(
        abstracts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    settings.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(settings.embeddings_path), embeddings)
    print(f"Saved embeddings to {settings.embeddings_path}")

    if not skip_db_update:
        print("Updating pgvector embeddings in DB ...")
        _update_embeddings_in_db(paper_ids, embeddings)

    print("Building FAISS HNSW index ...")
    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(settings.faiss_index_path))
    print(f"Saved FAISS index to {settings.faiss_index_path}")

    ids_path = str(settings.faiss_index_path) + ".ids.json"
    with open(ids_path, "w") as f:
        json.dump(paper_ids, f)
    print(f"Saved paper IDs to {ids_path}")
    print(f"Done. Indexed {len(paper_ids)} papers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build embeddings and FAISS index")
    parser.add_argument("--skip-db-update", action="store_true", help="Skip updating pgvector")
    args = parser.parse_args()
    asyncio.run(main(skip_db_update=args.skip_db_update))
