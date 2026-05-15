"""Generate synthetic QLoRA fine-tuning data from the base Phi-3 model.

Pulls papers from DB, runs base model inference at all 3 levels,
saves to ./data/synth_train.jsonl in Phi-3 chat format.

Run:
    python -m scholar.generation.synth_data --limit 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

OUTPUT_PATH = Path("./data/synth_train.jsonl")


def _load_papers(limit: int) -> list[tuple[str, str, str]]:
    from sqlmodel import Session, create_engine, select

    from scholar.config import settings
    from scholar.db.models import Paper

    engine = create_engine(settings.database_url_sync, echo=False)
    with Session(engine) as session:
        rows = session.exec(
            select(Paper.id, Paper.title, Paper.abstract).limit(limit)
        ).all()
    return [(pid, title, abstract) for pid, title, abstract in rows]


def generate_synth_data(
    limit: int = 5000, why: str = "Recommended based on your interests."
) -> None:
    from scholar.config import settings

    if not settings.generation_available:
        print("GPU not available. Synth data generation requires CUDA for reasonable speed.")
        print("Run on a CUDA machine or reduce --limit significantly for CPU.")

    from scholar.generation.inference import Phi3Generator
    from scholar.generation.prompts import get_prompt

    print(f"Loading papers (limit={limit})...")
    papers = _load_papers(limit)
    if not papers:
        print("No papers found. Load ArXiv data first.")
        return

    n_examples = len(papers) * 3
    print(f"Loaded {len(papers)} papers. Generating at 3 levels each = {n_examples} examples...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generator = Phi3Generator.get_instance()
    levels = ["undergrad", "grad", "researcher"]

    written = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for pid, title, abstract in tqdm(papers, desc="Papers"):
            for level in levels:
                try:
                    result = generator.generate(
                        title=title,
                        abstract=abstract,
                        why=why,
                        level=level,
                    )

                    glossary_str = "\n".join(f"- {t}" for t in result.get("glossary", []))
                    response_text = (
                        f"SUMMARY: {result.get('summary', '')}\n"
                        f"GLOSSARY:\n{glossary_str}\n"
                        f"WHY: {result.get('why', '')}"
                    )

                    record = {
                        "paper_id": pid,
                        "level": level,
                        "prompt": get_prompt(level=level, title=title, abstract=abstract, why=why),
                        "response": response_text,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as e:
                    print(f"Skipped {pid}/{level}: {e}")
                    continue

    print(f"Done. Wrote {written} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic QLoRA training data")
    parser.add_argument("--limit", type=int, default=5000, help="Number of papers to use")
    args = parser.parse_args()
    generate_synth_data(limit=args.limit)
