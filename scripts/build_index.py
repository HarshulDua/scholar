"""Build the Scholar search indexes: ArXiv loader + embeddings/FAISS + BM25."""
from __future__ import annotations

import argparse
import asyncio
import time


async def main(bm25_only: bool = False, faiss_only: bool = False, skip_ingest: bool = False) -> None:
    from rich.console import Console

    console = Console()
    console.print("[bold cyan]Scholar — Build Index[/bold cyan]")

    run_faiss = not bm25_only
    run_bm25 = not faiss_only

    if run_faiss:
        if not skip_ingest:
            console.rule()
            console.print("\n[bold]Step 1: Load ArXiv papers into PostgreSQL[/bold]")
            t0 = time.perf_counter()
            try:
                from scholar.ingest.arxiv_loader import main as loader_main
                await loader_main()
            except Exception as e:
                console.print(f"[red]ArXiv loader failed: {e}[/red]")
                raise
            console.print(f"[green]Done in {time.perf_counter() - t0:.1f}s[/green]")

        console.rule()
        console.print("\n[bold]Step 2: Build embeddings + FAISS index[/bold]")
        t0 = time.perf_counter()
        try:
            from scholar.ingest.embeddings import main as embeddings_main
            await embeddings_main()
        except Exception as e:
            print(f"ERROR: Embeddings/FAISS failed: {e}")
            raise
        print(f"Done in {time.perf_counter() - t0:.1f}s")

    if run_bm25:
        console.rule()
        console.print("\n[bold]Step 3: Build BM25 index[/bold]")
        t0 = time.perf_counter()
        try:
            from pathlib import Path
            from scholar.retrieval.bm25 import _build_from_db
            await _build_from_db(Path("./data/bm25.pkl"))
        except Exception as e:
            print(f"ERROR: BM25 builder failed: {e}")
            raise
        print(f"Done in {time.perf_counter() - t0:.1f}s")

    console.rule()
    console.print("[bold green]Index build complete.[/bold green]")
    console.print("Start the API: uvicorn scholar.api.main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Scholar search indexes")
    parser.add_argument("--bm25-only", action="store_true", help="Rebuild BM25 index only")
    parser.add_argument("--faiss-only", action="store_true", help="Rebuild FAISS index only")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ArXiv loader (papers already in DB)")
    args = parser.parse_args()
    asyncio.run(main(bm25_only=args.bm25_only, faiss_only=args.faiss_only, skip_ingest=args.skip_ingest))
