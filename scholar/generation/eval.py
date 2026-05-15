"""Generation evaluation: ROUGE-L and BERTScore on SciTLDR.

Compares base Phi-3 vs fine-tuned LoRA adapter across all 3 levels.

Run:
    python -m scholar.generation.eval
"""
from __future__ import annotations

from pathlib import Path

BENCHMARKS_PATH = Path("./docs/benchmarks.md")
SCITLDR_HF = "allenai/scitldr"
# Raw JSONL from allenai/scitldr GitHub — fallback when HF datasets script loading fails
SCITLDR_RAW_URL = (
    "https://raw.githubusercontent.com/allenai/scitldr/master/"
    "SciTLDR-Data/SciTLDR-AIC/ctrl/test.jsonl"
)


def _load_scitldr(split: str = "test", limit: int = 200) -> list[dict]:
    # Try HF datasets first (works if parquet copy exists)
    try:
        from datasets import load_dataset

        ds = load_dataset(SCITLDR_HF, "Abstract", split=split)
        records = []
        for item in list(ds)[:limit]:
            source = item.get("source", [])
            abstract = " ".join(source) if isinstance(source, list) else str(source)
            targets = item.get("target", [])
            reference = targets[0] if isinstance(targets, list) and targets else str(targets)
            records.append({
                "title": item.get("paper_id", ""),
                "abstract": abstract,
                "reference": reference,
            })
        if records:
            print(f"Loaded {len(records)} SciTLDR records from HuggingFace.")
            return records
    except Exception as e:
        print(f"HF datasets load failed ({e}). Trying GitHub raw JSONL...")

    # Fallback: download raw JSONL from GitHub
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(SCITLDR_RAW_URL, timeout=30) as resp:
            lines = resp.read().decode("utf-8").splitlines()
        records = []
        for line in lines[:limit]:
            if not line.strip():
                continue
            item = json.loads(line)
            source = item.get("source", [])
            abstract = " ".join(source) if isinstance(source, list) else str(source)
            targets = item.get("target", [])
            reference = targets[0] if isinstance(targets, list) and targets else str(targets)
            records.append({
                "title": item.get("paper_id", ""),
                "abstract": abstract,
                "reference": reference,
            })
        if records:
            print(f"Loaded {len(records)} SciTLDR records from GitHub raw JSONL.")
            return records
    except Exception as e:
        print(f"GitHub raw JSONL load failed ({e}).")

    # Final fallback: local ArXiv snapshot (title used as reference TLDR)
    try:
        import json

        snapshot = Path("./data/arxiv-metadata-oai-snapshot.json")
        if snapshot.exists():
            records = []
            with snapshot.open(encoding="utf-8") as f:
                for line in f:
                    if len(records) >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    title = (item.get("title") or "").strip().replace("\n", " ")
                    abstract = (item.get("abstract") or "").strip().replace("\n", " ")
                    if title and abstract and len(abstract) > 100:
                        records.append({
                            "title": title,
                            "abstract": abstract,
                            "reference": title,  # title as proxy TLDR
                        })
            if records:
                print(
                    f"Loaded {len(records)} ArXiv records from local snapshot "
                    f"(title used as reference TLDR)."
                )
                return records
    except Exception as e:
        print(f"Local ArXiv snapshot fallback failed ({e}).")

    print("Could not load eval data from any source.")
    return []


def _rouge_l(hypothesis: str, reference: str) -> float:
    """Simple token-based ROUGE-L (LCS)."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / n if n else 0.0
    recall = lcs / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _bertscore(hypotheses: list[str], references: list[str]) -> list[float]:
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)  # noqa: N806
        return F1.tolist()
    except Exception as e:
        print(f"BERTScore not available ({e}). Falling back to ROUGE-L only.")
        return [_rouge_l(h, r) for h, r in zip(hypotheses, references)]


def evaluate_level(
    level: str,
    records: list[dict],
    use_adapter: bool = False,
) -> dict:
    """Run inference for one level. BERTScore is deferred to main() to avoid VRAM conflicts."""
    from scholar.generation.inference import Phi3Generator

    label = f"Phi-3 + LoRA ({level})" if use_adapter else f"Phi-3 base ({level})"
    print(f"Evaluating {label} on {len(records)} examples...")

    if use_adapter:
        adapter_path = Path("./data/lora_adapter")
        if not adapter_path.exists():
            print(f"LoRA adapter not found at {adapter_path}. Skipping.")
            return {}

    gen = Phi3Generator.get_instance(load_adapter=use_adapter)
    hypotheses = []
    references = []

    for rec in records:
        try:
            result = gen.generate(
                title=rec["title"],
                abstract=rec["abstract"],
                why="Evaluation sample.",
                level=level,
            )
            hypothesis = result.get("summary", "")
        except Exception:
            hypothesis = ""
        hypotheses.append(hypothesis)
        references.append(rec["reference"])

    import numpy as np

    rouge_scores = [_rouge_l(h, r) for h, r in zip(hypotheses, references)]
    return {
        "label": label,
        "level": level,
        "adapter": use_adapter,
        "rouge_l": float(np.mean(rouge_scores)),
        "bertscore_f1": 0.0,       # filled in by main() after all inference
        "_hypotheses": hypotheses,  # kept for bulk BERTScore pass
        "_references": references,
        "n_samples": len(hypotheses),
    }


def _update_benchmarks(all_results: list[dict]) -> None:
    import re

    content = BENCHMARKS_PATH.read_text(encoding="utf-8")

    table = (
        "## Layer 3: Generation (Phi-3-mini 4-bit)\n\n"
        "Evaluation dataset: local ArXiv snapshot, title as reference TLDR proxy\n"
        "(SciTLDR deprecated in datasets 3.x; ArXiv fallback used).\n\n"
        "| Model              | Level      | ROUGE-L | BERTScore F1 | Samples |\n"
        "|--------------------|------------|---------|--------------|--------|\n"
    )
    for r in all_results:
        model_name = "Phi-3 + LoRA" if r["adapter"] else "Phi-3 base"
        table += (
            f"| {model_name:<18} | {r['level']:<10} | {r['rouge_l']:.4f}  "
            f"| {r['bertscore_f1']:.4f}       | {r['n_samples']:>7} |\n"
        )

    if "## Layer 3: Generation" in content:
        content = re.sub(
            r"## Layer 3: Generation.*?(?=\n---|\Z)", table, content, flags=re.DOTALL
        )
    else:
        content += "\n---\n\n" + table

    BENCHMARKS_PATH.write_text(content, encoding="utf-8")
    print(f"Benchmarks updated at {BENCHMARKS_PATH}")


def _free_gpu_memory() -> None:
    """Free Phi-3 singleton and flush GPU memory before the next model load."""
    import gc

    from scholar.generation.inference import Phi3Generator

    Phi3Generator._instance = None
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def main(limit: int = 200) -> None:
    import gc

    import numpy as np

    from scholar.config import settings

    if not settings.generation_available:
        print("GPU required for generation eval. CUDA not detected.")
        return

    print(f"Loading SciTLDR (limit={limit})...")
    records = _load_scitldr(limit=limit)
    if not records:
        return

    levels = ["undergrad", "grad", "researcher"]
    all_results = []

    # --- Phase 1: Base model (no LoRA adapter) ---
    # Ensure the singleton starts fresh and will NOT load the adapter.
    _free_gpu_memory()
    for level in levels:
        result = evaluate_level(level, records, use_adapter=False)
        if result:
            all_results.append(result)

    # Free Phi-3 + any VRAM fragments before loading the LoRA model.
    _free_gpu_memory()
    gc.collect()

    # --- Phase 2: LoRA model ---
    adapter_path = Path("./data/lora_adapter")
    if adapter_path.exists():
        for level in levels:
            result = evaluate_level(level, records, use_adapter=True)
            if result:
                all_results.append(result)
        _free_gpu_memory()
    else:
        print("LoRA adapter not found — skipping LoRA eval.")

    # --- Phase 3: BERTScore in one bulk pass (GPU now free from Phi-3) ---
    print("Computing BERTScore for all results...")
    for r in all_results:
        hyps = r.pop("_hypotheses", [])
        refs = r.pop("_references", [])
        bert_scores = _bertscore(hyps, refs)
        r["bertscore_f1"] = float(np.mean(bert_scores)) if bert_scores else 0.0

    if all_results:
        _update_benchmarks(all_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    main(limit=args.limit)
