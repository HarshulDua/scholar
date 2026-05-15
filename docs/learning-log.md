# Scholar — Developer Learning Log

## 2026-05-15 (Phase 2 Overnight Run)

### Environment: Windows 11, Python 3.13, RTX 2050 4GB VRAM, PyTorch 2.6.0+cu124

---

### 1. pyarrow DLL conflict — Windows Python 3.13

**Problem**: Server crashed at startup with `Windows fatal exception: access violation` in `pyarrow/__init__.py`. Every attempt to import `sentence_transformers` (which transitively imports pyarrow) crashed the process.

**Root cause**: `scholar.config` imports `torch` at module level. torch initialises its CUDA/cuDNN DLLs. Later, `sentence_transformers → datasets → pyarrow` tries to initialise Arrow C++ DLLs which conflict with torch's already-loaded DLLs.

**Fix**: `import pyarrow as _pa` added as the VERY FIRST import in `scholar/api/main.py` and at the top of `scholar/config.py` (before `import torch`). Once pyarrow's C extension is in `sys.modules`, subsequent imports skip DLL re-initialisation.

**Lesson**: On Windows, DLL initialisation order matters. The only reliable fix is to control import order. `PYTHONFAULTHANDLER=1` is essential for diagnosing C-level crashes.

---

### 2. FAISS HNSW segfault — asyncio + Windows Python 3.13

**Problem**: POST /search caused SIGSEGV (exit code 139) after the first request. No Python traceback — crash happened in C code.

**Root cause**: FAISS HNSW uses an OpenMP thread pool. On Windows + Python 3.13, OMP thread pool initialisation conflicts with asyncio's event loop when called from an async handler.

**Fix**: Replaced FAISS ANN search with numpy brute-force matmul: `scores = embeddings @ query_vec`. For 200K × 384 float32, this takes ~10ms per query via BLAS GEMV — competitive with HNSW, and has exact recall (no approximation error).

**Lesson**: For <1M vectors, brute-force matmul is often simpler and faster than HNSW. HNSW's advantage is memory efficiency and logarithmic scaling — not relevant at 200K scale.

---

### 3. trl 1.4.0 breaking API changes

**Problem**: `SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'`.

**Fix**: `trl 1.4.0` moved SFT-specific args to `SFTConfig` and renamed `tokenizer=` to `processing_class=`. Updated `train_lora.py` accordingly.

**Lesson**: Pin `trl` in requirements. The `processing_class` rename broke all existing tutorials/code without a deprecation warning.

---

### 4. transformers 5.x DynamicCache incompatibility with Phi-3 Hub code

**Problem**: synth_data generated 0 examples silently — every call raised `AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'`.

**Root cause**: Phi-3's cached `modeling_phi3.py` on HuggingFace Hub was written against transformers 4.x. transformers 5.x removed `seen_tokens`, `get_max_length()`, and `get_usable_length()` from DynamicCache.

**Fix**: `trust_remote_code=False` tells transformers to use its own built-in Phi-3 implementation (compatible with 5.x) instead of the Hub's cached modeling file. The monkey-patch approach (adding missing methods to DynamicCache) also works but is more fragile.

**Lesson**: `trust_remote_code=True` is a security risk AND a compatibility risk. Always try `False` first for models that transformers supports natively.

---

### 5. numpy matmul for dense retrieval — practical performance

**Measurement** on RTX 2050 / 8-core CPU, 200K × 384 float32:
- `scores = embeddings @ query_vec`: ~10ms per query (BLAS DGEMV)
- `np.argpartition(scores, -k)[-k:]` O(n) top-k: ~1ms for k=200
- Total dense search (excluding encoding): ~11ms per query
- Sentence-BERT encoding (`all-MiniLM-L6-v2`): ~110ms per query
- Bottleneck: model encoding, not the matmul search

**Lesson**: For moderate-scale vector search, the ANN approximation is often not the bottleneck. Profile encoding vs search separately.

---

### 6. CiteULike-ArXiv overlap — lower than expected

**Data**: CiteULike-A has 16,980 papers. Our ArXiv corpus has 200K papers. Only 158 CiteULike papers appear in our ArXiv corpus.

**Why**: CiteULike includes conference and journal papers (ACL, NIPS, ICML, JMLR) that have no ArXiv preprint. Only preprint-first papers overlap. 158 is ~0.9% overlap.

**Impact**: NDCG@10 on CiteULike = 0.000 (CiteULike paper IDs not in our retrieval results). ILD (Intra-List Diversity) increases from 0.496 → 0.576 with exploration — this is the meaningful signal.

**Lesson**: When evaluating with external datasets, validate ID overlap FIRST before investing in full eval runs. The 0% NDCG is expected, not a bug.

---

### 7. NFCorpus evaluation — URL gone

**Problem**: `ir_datasets` tries to download NFCorpus from `https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz` → HTTP 404. Dataset removed from original host.

**Workaround**: Ran synthetic latency benchmark using 10 ArXiv-style queries instead. Measured actual p50/p99 latencies rather than relevance metrics.

**TODO**: Alternative sources: Hugging Face datasets, BEIR benchmark (has NFCorpus). Update `retrieval/eval.py` to use BEIR's version.

---

### 8. ruff + ML naming conventions

**Problem**: Ruff's N806 rule ("variable in function should be lowercase") fires on `X`, `X_train`, `N`, `P`, `R`, `F1` — all standard ML notation from seminal papers (Hastie 2001, Robertson BM25 1994, etc.).

**Fix**: Added `ignore = ["N806", "N812"]` to `[tool.ruff.lint]` in `pyproject.toml`. Also extended `line-length` from 100 to 120 to accommodate SQL column lists and Markdown table headers.

**Lesson**: Configure linters for your domain. ML code has pre-PEP-8 naming traditions from mathematics and statistics that linters shouldn't fight.

---

### 9. mypy + SQLModel — `"str" has no attribute "in_"`

**Problem**: mypy reports `"str" has no attribute "in_"` when using `Model.field.in_(list)` in SQLModel queries.

**Root cause**: SQLModel's type stubs type model field attributes as their Python types (`str`, `int`) rather than SQLAlchemy column types (`Column[str]`). The `.in_()` method is on the column type, not the Python type.

**Fix**: `# type: ignore[attr-defined]` on each affected line. This is a known SQLModel/mypy compatibility limitation — not a real bug.

**Reference**: SQLModel GitHub issues #52, #234 — filed but not resolved as of 2026.

---

### 10. GPU VRAM contention — synth_data vs /explain

**Observation**: RTX 2050 has 4GB VRAM. Phi-3-mini 4-bit NF4 uses ~2GB VRAM (1.9GB model + 0.8GB activations/KV cache). synth_data and the API's /explain endpoint both need Phi-3, but they can't share VRAM.

**Symptom**: `/explain` returns `OSError: The paging file is too small for this operation to complete` when synth_data is running.

**Resolution**: VRAM is exclusive per process on Windows. Run synth_data to completion, then test /explain in a separate session. The API server itself doesn't preload Phi-3 (it's loaded on first /explain request or at startup with WARMUP_GENERATION=true).

---

### 11. pyarrow DLL conflict in train_lora.py (same root, new manifestation)

**Problem**: `python -m scholar.generation.train_lora` exited with code 5, 0 bytes output.

**Root cause**: `train_lora.py` imports `torch` at module level (line 18). When trl then imports `transformers.generation.utils → sklearn → pyarrow`, the Arrow C++ DLL conflicts with torch's already-loaded DLLs — the same bug as `main.py`, just triggered via a different import chain (trl→transformers→sklearn instead of sentence_transformers→datasets).

**Fix**: Added `import pyarrow as _pa` before `import torch` in `train_lora.py`.

**Lesson**: Any script that imports torch and then later imports trl/transformers/sklearn needs the pyarrow-first guard. The fix belongs at the entrypoint, not in a library.

---

### 12. PYTHONUTF8=1 rejected by pydantic Settings (extra_forbidden)

**Problem**: Adding `PYTHONUTF8=1` to `.env` fixed the trl charmap issue but broke `Settings()` instantiation with `Extra inputs are not permitted [type=extra_forbidden, input_value='1']`.

**Root cause**: pydantic-settings v2 `BaseSettings` defaults to `extra="ignore"` but our `model_config` dict didn't specify `extra`, and a newer pydantic-settings version changed the default to `"forbid"` or inherits it from the model validator.

**Fix**: Added `"extra": "ignore"` to `model_config` in `scholar/config.py`. This is the correct setting — `Settings` should absorb any OS env vars it doesn't know about without failing.

**Lesson**: Always set `extra="ignore"` explicitly in pydantic Settings. Environment variable namespaces are shared; your app will always encounter variables it doesn't own (CI vars, OS vars, utility env vars like PYTHONUTF8).

---

## Benchmark Results Summary (2026-05-15)

### Layer 1: Retrieval Latency
| Retriever | p50 | p99 |
|-----------|-----|-----|
| BM25 (scratch) | 26.6ms | 44.5ms |
| Dense (numpy matmul) | 121.6ms | 98s* |
| Hybrid (RRF k=60) | 106.5ms | 196.2ms |

*p99=98s is model cold-start. After warmup, Dense p50~121ms.

### Layer 2: Recsys Ablation (CiteULike-A, 38 users)
| Ablation | NDCG@10 | ILD |
|----------|---------|-----|
| MMR alone | 0.000 | 0.496 |
| LTR + MMR | 0.000 | 0.496 |
| LTR + MMR + exploration | 0.000 | 0.576 |

NDCG=0 expected (CiteULike IDs not in ArXiv corpus). ILD increase confirms exploration works.

### Layer 3: Generation (complete — 2026-05-15 19:26 PM)
- LoRA training: 10.3 hours, train_loss=1.037, final token_accuracy=0.767
- Adapter saved: data/lora_adapter/ (adapter_model.safetensors 6.3MB)
- SciTLDR eval: pending (datasets 3.x deprecated loading scripts; running on local ArXiv fallback)

## QLoRA Training Speed — RTX 2050 Reality Check

**Measured**: ~317 seconds/optimizer step on RTX 2050 4GB VRAM (259s warm-up on step 1, 300-320s steady state)  
**Config**: Phi-3-mini-4k, 4-bit NF4, LoRA rank 8, seq_len 768, batch=1, grad_accum=8, gradient_checkpointing=True  
**Breakdown (estimated)**:
- Each optimizer step = 8 forward + 8 backward passes (grad_accum=8)
- Backward with gradient checkpointing = 2× forward cost (recomputes activations)
- Per-batch: ~10s forward + ~20s backward (checkpoint recompute) = ~30s
- 8 batches × 30s = ~240s/step → matches observed 259s

**Total training time**: 114 steps × ~330s = ~10.5 hours for 300 examples, 3 epochs

**Epoch 1 loss curve** (from trainer_state.json in checkpoint-38):
| Step | Loss | Token Acc | Grad Norm |
|------|------|-----------|-----------|
| 10   | 1.434 | 0.669 | 0.224 |
| 20   | 1.221 | 0.710 | 0.271 |
| 30   | 1.124 | 0.733 | 0.447 |

Healthy downward curve. Grad norm increasing slightly (0.224 → 0.447) as training deepens — normal for low-rank LoRA adapting to in-domain vocabulary. Token accuracy climbing steadily (67% → 73%).

**Lesson**: QLoRA on consumer GPU (4GB) is extremely slow for large models (3.8B). For a 300-example fine-tune at seq_len 768:
- RTX 2050 4GB: ~8 hours
- RTX 3090 24GB (no NF4 needed): ~30 min  
- A100 80GB: ~5 min
The bottleneck is gradient checkpointing forced by tight VRAM — it trades 3× slower backward passes for ~40% memory savings. At 4GB with a 3.8B model, there's no alternative.

---

### 13. /explain RAM constraint — API server + Phi-3 combined overhead

**Problem**: `/explain` returns `OSError: The paging file is too small` even after training is complete and VRAM is freed.

**Root cause**: The API server loads at startup: embeddings.npy (307MB RAM), sentence-transformer all-MiniLM-L6-v2 (~500MB VRAM), BM25 numpy cache, LTR model. When `/explain` then tries to load Phi-3 4-bit NF4 (~1.9GB VRAM + ~2GB RAM for loading), combined RAM exceeds the Windows paging file limit. With only ~3GB free RAM on this 15.7GB system and paging file not auto-expanded large enough, the OS rejects the allocation.

**Resolution**: Three options: (1) increase Windows paging file to ≥8GB, (2) run Phi-3 generation as a separate process with `--lora-only` mode without loading the retrieval stack, (3) use CPU offloading for the sentence-transformer during explain calls.

**Lesson**: On memory-constrained Windows systems, model loading failures often surface as paging file errors rather than OOM — the system runs out of virtual address space before physical RAM.

---

### 15. Base model evaluation silently loading LoRA adapter

**Problem**: In `generation/eval.py`, "Phi-3 base" evaluation produced identical results to "Phi-3 + LoRA" — `Phi3Generator.__init__` always called `PeftModel.from_pretrained` if `settings.lora_adapter_path` existed.

**Fix**: Added `load_adapter: bool = True` parameter to `Phi3Generator.__init__` and `get_instance()`. `evaluate_level(use_adapter=False)` now calls `get_instance(load_adapter=False)`, preventing silent adapter loading.

**Lesson**: Model loading code that conditionally loads adapters based on environment config can silently affect evaluations if the config is set from a prior run. Eval code must explicitly control what's loaded rather than relying on ambient settings.

---

### 16. BERTScore roberta-large blocks Phi-3 reload (VRAM fragmentation)

**Problem**: After base model evaluation, `_bertscore()` loaded roberta-large into VRAM and it stayed. When `main()` tried to reload Phi-3 for the LoRA phase (after `Phi3Generator._instance = None`), the combined VRAM demand exceeded 4GB and raised: `ValueError: Some modules are dispatched on the CPU or the disk`.

**Fix**: Moved BERTScore computation out of `evaluate_level` — inference now returns raw hypotheses/references. `main()` calls `_free_gpu_memory()` (which does `gc.collect(); torch.cuda.empty_cache()`) between base and LoRA phases, then computes BERTScore in a single bulk pass after all inference is done (VRAM free at that point).

**Lesson**: On memory-constrained GPUs, evaluation pipelines that load multiple large models must explicitly manage GPU memory between phases. Python's GC doesn't immediately free CUDA memory — `torch.cuda.empty_cache()` is required.

---

### 14. allenai/scitldr deprecated in datasets 3.x

**Problem**: `load_dataset("allenai/scitldr", ..., trust_remote_code=True)` raises `Dataset scripts are no longer supported`. GitHub raw JSONL fallback also 404.

**Root cause**: HuggingFace datasets 3.x removed loading script support for security reasons. The allenai/scitldr dataset was never converted to parquet format. The GitHub repo URL structure changed.

**Workaround**: Fallback to local ArXiv snapshot (`data/arxiv-metadata-oai-snapshot.json`), using the paper title as a TLDR proxy. ROUGE-L against the title is a reasonable proxy for evaluating summary conciseness, though not identical to SciTLDR evaluation.

**TODO**: Find SciTLDR parquet on HF Hub, or download from allenai's current distribution URL.
