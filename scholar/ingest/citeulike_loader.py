"""Load CiteULike-A interaction data into user_history table.

Dataset: https://github.com/js05212/citeulike-a
  - raw-data.csv : doc.id, title, citeulike.id, raw.title, raw.abstract
  - users.dat    : one line per user, space-separated doc.id values (1-based)

Run:
    python -m scholar.ingest.citeulike_loader --data-dir ./data/citeulike-a
"""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras
from tqdm import tqdm


def _download_citeulike(data_dir: Path) -> bool:
    import subprocess

    url = "https://github.com/js05212/citeulike-a"
    print(f"Cloning CiteULike-A from {url} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(data_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"git clone failed: {result.stderr.strip()}")
        return False
    print("Downloaded CiteULike-A.")
    return True


def _load_citeulike_titles(data_dir: Path) -> dict[int, str]:
    """Return {doc_id: raw_title} from raw-data.csv (1-based doc ids)."""
    csv_path = data_dir / "raw-data.csv"
    papers_path = data_dir / "papers.txt"

    if csv_path.exists():
        titles: dict[int, str] = {}
        with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    doc_id = int(row["doc.id"])
                    # prefer raw.title (original case) over lowercased title
                    title = row.get("raw.title") or row.get("title") or ""
                    titles[doc_id] = title.strip()
                except (ValueError, KeyError):
                    continue
        print(f"Loaded {len(titles)} titles from raw-data.csv")
        return titles

    if papers_path.exists():
        titles = {}
        with open(papers_path, encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                parts = line.strip().split("\t")
                titles[idx] = parts[0].strip('"').strip()
        print(f"Loaded {len(titles)} titles from papers.txt")
        return titles

    return {}


def _match_to_db(doc_titles: dict[int, str], conn) -> dict[int, str]:
    """Return {doc_id: db_paper_id} via exact normalised-title match."""
    print("Loading DB paper titles for matching...")
    with conn.cursor() as cur:
        cur.execute("SELECT id, title FROM papers")
        rows = cur.fetchall()

    exact: dict[str, str] = {title.lower().strip(): pid for pid, title in rows}

    mapping: dict[int, str] = {}
    for doc_id, title in doc_titles.items():
        norm = title.lower().strip()
        if norm and norm in exact:
            mapping[doc_id] = exact[norm]

    print(f"Matched {len(mapping)}/{len(doc_titles)} CiteULike papers to DB papers.")
    return mapping


def load_citeulike(data_dir: Path) -> None:
    from scholar.config import settings

    users_file = data_dir / "users.dat"
    raw_csv = data_dir / "raw-data.csv"
    papers_txt = data_dir / "papers.txt"

    if not users_file.exists() or (not raw_csv.exists() and not papers_txt.exists()):
        print(f"CiteULike data not found in {data_dir}.")
        print("Trying to download...")
        if not _download_citeulike(data_dir):
            print("Could not download CiteULike-A. Please download manually.")
            return

    doc_titles = _load_citeulike_titles(data_dir)
    if not doc_titles:
        print("No paper titles found in dataset.")
        return

    conn = psycopg2.connect(str(settings.database_url_sync))

    try:
        doc_to_db = _match_to_db(doc_titles, conn)
        if not doc_to_db:
            print("No papers matched. Ensure ArXiv papers are loaded first.")
            return

        # Load user mapping cache for reproducible reruns
        mapping_file = data_dir / "user_id_mapping.json"
        user_id_mapping: dict[str, str] = {}
        if mapping_file.exists():
            with open(mapping_file) as f:
                user_id_mapping = json.load(f)

        now = datetime.now()
        users_to_insert: list[tuple] = []
        history_to_insert: list[tuple] = []

        with open(users_file, encoding="utf-8") as f:
            lines = f.readlines()

        print(f"Processing {len(lines)} CiteULike users...")
        for citeulike_uid, line in enumerate(tqdm(lines, desc="Users")):
            line = line.strip()
            if not line:
                continue

            cu_key = str(citeulike_uid)
            if cu_key not in user_id_mapping:
                user_id_mapping[cu_key] = str(uuid.uuid4())
            db_user_id = user_id_mapping[cu_key]

            users_to_insert.append((db_user_id, now))

            for part in line.split():
                try:
                    doc_id = int(part)
                except ValueError:
                    continue
                db_paper_id = doc_to_db.get(doc_id)
                if db_paper_id:
                    history_to_insert.append((db_user_id, db_paper_id, now))

        # Bulk insert users
        print(f"Inserting {len(users_to_insert)} users...")
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO users (id, created_at) VALUES %s ON CONFLICT (id) DO NOTHING",
                users_to_insert,
            )
        conn.commit()

        # Bulk insert interactions (deduplicated)
        seen: set[tuple[str, str]] = set()
        unique_history = []
        for row in history_to_insert:
            key = (row[0], row[1])
            if key not in seen:
                seen.add(key)
                unique_history.append(row)

        print(f"Inserting {len(unique_history)} unique interactions...")
        batch_size = 5000
        with conn.cursor() as cur:
            for i in tqdm(range(0, len(unique_history), batch_size), desc="Inserting history"):
                batch = unique_history[i : i + batch_size]
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO user_history (user_id, paper_id, clicked_at) VALUES %s ON CONFLICT DO NOTHING",
                    batch,
                )
        conn.commit()

    finally:
        conn.close()

    with open(mapping_file, "w") as f:
        json.dump(user_id_mapping, f)

    print(f"Done. {len(users_to_insert)} users, {len(unique_history)} interactions inserted.")
    print(f"User ID mapping saved to {mapping_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CiteULike-A interactions")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/citeulike-a"),
        help="Path to CiteULike-A dataset directory",
    )
    args = parser.parse_args()
    load_citeulike(args.data_dir)
