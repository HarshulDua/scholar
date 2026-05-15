"""Stream ArXiv metadata JSON and upsert into PostgreSQL."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlmodel import create_engine
from tqdm import tqdm

from scholar.config import settings
from scholar.db.models import Paper

TARGET_CATEGORIES = {"cs.", "stat.ML", "stat.AP", "math.ST"}


def _matches_category(categories: str) -> bool:
    for cat in TARGET_CATEGORIES:
        if cat in categories:
            return True
    return False


def _parse_paper(obj: dict) -> Paper | None:
    categories = obj.get("categories", "")
    if not _matches_category(categories):
        return None

    paper_id = obj.get("id", "").strip()
    title = (obj.get("title") or "").replace("\n", " ").strip()
    abstract = (obj.get("abstract") or "").replace("\n", " ").strip()

    authors_parsed = obj.get("authors_parsed", [])
    if authors_parsed:
        author_names = []
        for a in authors_parsed:
            parts = [p.strip() for p in a if p.strip()]
            author_names.append(" ".join(reversed(parts[:2])))
        authors = ", ".join(author_names[:10])
    else:
        authors = (obj.get("authors") or "").replace("\n", " ").strip()

    update_date_str = obj.get("update_date")
    update_date = None
    if update_date_str:
        from datetime import date
        try:
            update_date = date.fromisoformat(update_date_str)
        except ValueError:
            pass

    if not paper_id or not title or not abstract:
        return None

    return Paper(
        id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        update_date=update_date,
        citation_count=0,
    )


def load_arxiv(data_path: Path, limit: int, engine) -> int:
    loaded = 0
    batch: list[Paper] = []
    batch_size = 500

    def flush(conn, papers: list[Paper]) -> None:
        if not papers:
            return
        rows = [
            (
                p.id, p.title, p.abstract, p.authors,
                p.categories, p.update_date, p.citation_count,
            )
            for p in papers
        ]
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO papers (id, title, abstract, authors, categories, update_date, citation_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                rows,
            )
        conn.commit()

    import psycopg2

    # Use the sync DSN directly — avoids SQLAlchemy URL re-parsing which
    # drops asyncpg dialect prefix before psycopg2 sees it, causing auth failures.
    dsn = str(settings.database_url_sync)
    conn = psycopg2.connect(dsn)

    try:
        with open(data_path, "r", encoding="utf-8") as fh:
            pbar = tqdm(desc="Loading papers", unit="papers")
            for line in fh:
                if loaded >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                paper = _parse_paper(obj)
                if paper is None:
                    continue
                batch.append(paper)
                loaded += 1
                pbar.update(1)
                if len(batch) >= batch_size:
                    flush(conn, batch)
                    batch = []
            if batch:
                flush(conn, batch)
            pbar.close()
    finally:
        conn.close()

    return loaded


async def main(limit: int | None = None) -> None:
    effective_limit = limit if limit is not None else settings.max_papers
    data_path = settings.arxiv_data_path

    if not data_path.exists():
        print(f"ArXiv data not found at {data_path}. Run scripts/download_arxiv.sh first.")
        return

    engine = create_engine(settings.database_url_sync, echo=False)

    print(f"Loading up to {effective_limit} papers from {data_path} ...")
    count = load_arxiv(data_path, effective_limit, engine)
    print(f"Done. Loaded {count} papers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load ArXiv papers into DB")
    parser.add_argument("--limit", type=int, default=None, help="Max papers to load")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
