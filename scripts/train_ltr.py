"""Train LightGBM LambdaRank model on CiteULike interactions.

Requires: ArXiv papers loaded, FAISS index built, CiteULike interactions loaded.

Run:
    python scripts/train_ltr.py
    python scripts/train_ltr.py --output ./data/ltr_model.lgb
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(output_path: Path) -> None:
    from sqlmodel import Session, create_engine, select

    from scholar.config import settings
    from scholar.db.models import Paper, UserHistory
    from scholar.recsys.ltr import LTRModel
    from scholar.retrieval.bm25 import BM25
    from scholar.retrieval.dense import DenseRetriever

    print("Loading FAISS index...")
    dense = DenseRetriever()
    dense.load()
    print(f"FAISS loaded: {len(dense._paper_ids)} papers.")

    bm25_path = Path("./data/bm25.pkl")
    bm25 = None
    if bm25_path.exists():
        bm25 = BM25.load(bm25_path)
        print("BM25 index loaded.")

    engine = create_engine(settings.database_url_sync, echo=False)

    print("Loading paper metadata...")
    paper_meta: dict[str, Paper] = {}
    offset = 0
    batch_size = 1000
    while True:
        with Session(engine) as session:
            rows = session.exec(select(Paper).offset(offset).limit(batch_size)).all()
        if not rows:
            break
        for p in rows:
            paper_meta[p.id] = p
        offset += batch_size
        if len(rows) < batch_size:
            break
    print(f"Loaded {len(paper_meta)} papers.")

    print("Loading user interactions...")
    with Session(engine) as session:
        rows = session.exec(select(UserHistory.user_id, UserHistory.paper_id)).all()
    interaction_data = [(str(uid), str(pid)) for uid, pid in rows]
    print(f"Loaded {len(interaction_data)} interactions.")

    if not interaction_data:
        print("No interactions found. Run citeulike_loader first.")
        return

    ltr = LTRModel()
    ltr.train(
        interaction_data=interaction_data,
        faiss_retriever=dense,
        bm25=bm25,
        paper_meta=paper_meta,
        save_path=output_path,
    )
    print(f"LTR model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaRank LTR model")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/ltr_model.lgb"),
        help="Path to save the LTR model",
    )
    args = parser.parse_args()
    main(output_path=args.output)
