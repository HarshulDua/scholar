# Scholar — Implementation Guide

> 6-week build plan. Each phase: what you'll learn, files to create, key concepts, deliverables, gotchas, and the interview question each phase prepares you for.

---

## 0. Setup (Day 0–1)

### Repo structure (create this upfront)

```
scholar/
├── README.md                    # Public-facing, the README_TEMPLATE
├── CLAUDE.md                    # Claude Code project context
├── pyproject.toml               # uv or poetry
├── docker-compose.yml           # Postgres + API + UI
├── .env.example                 # DB URLs, API keys (no secrets committed)
│
├── docs/
│   ├── DESIGN_DOC.md            # The full design doc
│   ├── decisions.md             # ADRs as you go
│   ├── learning-log.md          # Daily notes — your interview prep gold
│   ├── benchmarks.md            # All metric tables
│   └── INTERVIEW.md             # Rehearsed Q&A
│
├── scholar/                     # The actual Python package
│   ├── __init__.py
│   ├── config.py                # Pydantic settings, env loading
│   │
│   ├── ingest/                  # Phase 1
│   │   ├── arxiv_loader.py      # Stream the Kaggle dump → Postgres
│   │   ├── embeddings.py        # Compute sentence-BERT embeddings
│   │   └── citeulike_loader.py  # Phase 2 — user history
│   │
│   ├── retrieval/               # Phase 1
│   │   ├── bm25.py              # Custom BM25
│   │   ├── dense.py             # FAISS index wrapper
│   │   ├── hybrid.py            # RRF fusion
│   │   └── eval.py              # NDCG/MRR/MAP harness
│   │
│   ├── recsys/                  # Phase 2
│   │   ├── mult_vae.py          # PyTorch Mult-VAE
│   │   ├── train_vae.py
│   │   ├── ltr.py               # LightGBM LambdaRank
│   │   ├── features.py          # Feature engineering for LTR
│   │   ├── mmr.py               # Diversity reranking
│   │   ├── exploration.py       # Epsilon-greedy / fresh start
│   │   └── eval.py              # NDCG, ILD, topic entropy
│   │
│   ├── generation/              # Phase 3
│   │   ├── prompts.py           # Prompt templates per level
│   │   ├── synth_data.py        # Generate LoRA training data
│   │   ├── train_lora.py        # QLoRA training script
│   │   ├── inference.py         # Inference wrapper
│   │   └── eval.py              # ROUGE, BERTScore
│   │
│   ├── api/                     # Phase 4
│   │   ├── main.py              # FastAPI app
│   │   ├── routes/
│   │   │   ├── search.py
│   │   │   ├── recommend.py
│   │   │   ├── explain.py
│   │   │   └── user.py
│   │   └── schemas.py           # Pydantic models
│   │
│   ├── ui/                      # Phase 4
│   │   └── gradio_app.py        # The demo UI
│   │
│   └── db/
│       ├── schema.sql
│       ├── migrations/
│       └── models.py            # SQLAlchemy / SQLModel
│
├── notebooks/                   # Experiments, ablations, plots
│   ├── 01_dataset_eda.ipynb
│   ├── 02_retrieval_ablation.ipynb
│   ├── 03_vae_training_curves.ipynb
│   ├── 04_filter_bubble_sim.ipynb
│   └── 05_lora_eval.ipynb
│
├── scripts/                     # One-off setup
│   ├── download_arxiv.sh
│   ├── build_index.py
│   └── deploy_demo.sh
│
├── tests/                       # pytest
│   ├── test_bm25.py
│   ├── test_mult_vae.py
│   └── test_e2e.py
│
└── assets/                      # Screenshots, GIFs for the README
    ├── architecture.png
    └── demo.gif
```

### Environment

```bash
# Python 3.11
uv venv && source .venv/bin/activate
uv pip install fastapi uvicorn gradio sqlmodel asyncpg pgvector \
               rank-bm25 faiss-cpu sentence-transformers \
               torch torchvision \
               lightgbm scikit-learn ranx \
               transformers peft bitsandbytes accelerate datasets \
               pandas pyarrow tqdm rich \
               pytest ruff mypy
```

### Day 0 deliverable
- Repo initialized with structure above
- `docker-compose.yml` brings up Postgres + pgvector
- `CLAUDE.md` and the design doc committed
- First commit ("scaffold").

---

## Phase 1: Retrieval (Weeks 1–2)

### Goals
- Working ingestion pipeline for ArXiv
- BM25 from scratch (not just calling rank_bm25)
- Dense retriever with FAISS
- Hybrid retrieval with RRF
- Eval harness with NDCG/MRR/MAP on a held-out query set

### Concepts to master before coding

- **TF-IDF and BM25** — read the original BM25 paper (Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"). Understand why BM25 has the `k1` and `b` parameters and what they do intuitively.
- **Inverted index** — implement one. It's a Python dict of `term → list[(doc_id, term_freq)]`. The data structure is trivial; getting query-time scoring right is the lesson.
- **Dense retrieval** — sentence-transformers' bi-encoder architecture. Read the SBERT paper (Reimers & Gurevych 2019). Understand why we encode queries and docs independently rather than as pairs (it's about scaling).
- **FAISS** — understand IndexFlatIP (exact), IndexHNSWFlat (approximate). HNSW is the workhorse at our scale.
- **Reciprocal Rank Fusion** — score = sum over retrievers of `1/(k + rank_i)`. Why k=60. Why this beats score normalization.
- **IR metrics** — NDCG (rewards correct order at top), MRR (one relevant doc, where is it?), MAP (averages precision at every recall point). Know when each one applies.

### Week 1 task list

1. Download ArXiv dump from Kaggle (`arxiv-metadata-oai-snapshot.json`).
2. Build `ingest/arxiv_loader.py`: stream JSON, filter to CS/stat/math categories, insert into Postgres.
3. Build `retrieval/bm25.py` from scratch. Don't import `rank_bm25` — write the inverted index and scoring yourself. ~200 lines.
4. Validate: pick 10 papers manually, write 10 queries, check that BM25 returns the right paper top-3 each time. Sanity check before scale.
5. Run BM25 on full corpus, measure: query latency, memory usage.

### Week 2 task list

1. Build `retrieval/dense.py`: encode all abstracts with `all-MiniLM-L6-v2`. Save to FAISS HNSW index. Persist to disk.
2. Build `retrieval/hybrid.py`: RRF fusion of BM25 top-200 and dense top-200.
3. Build `retrieval/eval.py`: load NFCorpus, run all three retrievers, compute NDCG@10 / MRR / MAP / Recall@100.
4. Run the ablation: BM25 alone vs dense alone vs hybrid. Save the numbers in `docs/benchmarks.md`.
5. Write a notebook (`02_retrieval_ablation.ipynb`) with the comparison table and short qualitative analysis (which queries does each retriever win on?).

### Phase 1 deliverables
- ArXiv corpus loaded into Postgres (~2M papers)
- Custom BM25 implementation
- Dense retriever via FAISS
- Hybrid retriever with RRF
- Eval table comparing all three
- Latency numbers documented
- ADR-003 committed
- ~10 commits across the two weeks

### Gotchas
- **Memory:** 2M abstracts × 384-dim FP32 embeddings = ~3GB. Use FP16 to halve. HNSW index is another ~2GB. Build incrementally; don't load all at once.
- **BM25 implementation traps:** off-by-one in average document length, log base for IDF (use natural log), how to handle terms not in vocabulary (return 0).
- **Don't trust score magnitudes across retrievers.** BM25 scores are unbounded; dense cosine sims are in [-1, 1]. RRF sidesteps this entirely; weighted sum requires careful normalization.

### Interview questions this phase prepares you for
- "Walk me through your retrieval architecture."
- "Why hybrid retrieval — when does BM25 beat dense, and vice versa?"
- "Why RRF and not weighted sum?"
- "What's the difference between NDCG and MAP, and when would you prefer each?"

---

## Phase 2: Recommendation + Anti-Bubble (Weeks 3–4)

### Goals
- Mult-VAE trained on CiteULike
- LightGBM LambdaRank reranker
- MMR diversity rerank with adaptive λ
- Epsilon-greedy exploration with "Fresh Start" feature
- Filter-bubble simulation

### Concepts to master before coding

- **VAE refresher** — ELBO, reparameterization, posterior collapse. You have this from CS F437. Re-read your notes.
- **Mult-VAE specifics** — Liang et al. 2018. Read the paper. Understand why they use a multinomial likelihood (papers are exclusive picks, not independent Bernoulli draws), and the KL annealing schedule.
- **Learning to Rank** — pointwise vs pairwise vs listwise losses. LambdaRank is listwise, optimizes for NDCG directly. Understand the lambda gradient (it's surprisingly elegant).
- **MMR** — the formula and the intuition: at each step, pick the candidate that maximizes `λ × relevance − (1−λ) × max_sim_to_already_picked`.
- **Exploration-exploitation** — epsilon-greedy, UCB. Why exploration matters in recsys specifically (vs RL).

### Week 3 task list

1. Build `ingest/citeulike_loader.py`: load CiteULike user-paper interactions. ~8K users, ~150K interactions.
2. Build `recsys/mult_vae.py`: implement the architecture (encoder 2M→600→200 latent → decoder 200→600→2M with multinomial likelihood).
3. Build `recsys/train_vae.py`: KL-annealed training loop. Train for ~50 epochs (~30 min on RTX 2080).
4. Build `recsys/eval.py`: NDCG@10, HR@10, MRR on the held-out users.
5. Compare against baselines: random, popular, k-NN on averaged paper embeddings. Mult-VAE should win by ≥20% NDCG@10.
6. Write learning-log entry: explain WHY Mult-VAE wins (regularized latent, captures combinatorial interests).

### Week 4 task list

1. Build `recsys/features.py`: extract features for LTR — query-doc BM25 score, query-doc dense sim, user-doc latent sim (from Mult-VAE), paper recency, citation count, topic match.
2. Build `recsys/ltr.py`: LightGBM LambdaRank model. Train on synthetic relevance labels (combine clicks + topic match heuristic).
3. Build `recsys/mmr.py`: MMR with tunable λ. Add adaptive λ based on user history clustering coefficient.
4. Build `recsys/exploration.py`: epsilon-greedy. Add the "Fresh Start" function that re-initializes the user embedding to the prior.
5. Build the filter-bubble simulation: synthetic user, 20 sessions, log topic entropy. Compare with and without MMR + exploration.
6. Make plots: topic entropy over time, with vs without anti-bubble mechanisms. This is the gold figure for your README.

### Phase 2 deliverables
- Trained Mult-VAE checkpoint
- Trained LightGBM ranker
- MMR + exploration code
- Filter-bubble simulation notebook with the killer plot
- Anti-bubble metrics in `docs/benchmarks.md`
- ADR-002, ADR-005 committed
- ~10 commits

### Gotchas
- **Mult-VAE posterior collapse:** if KL term dominates, all users get the same z. Use KL annealing (start with weight 0, ramp up to 0.2 over 20 epochs). Anneal-to-something-less-than-1 is a published trick.
- **LightGBM with very sparse features:** make sure you've explicitly set categorical features or use one-hot. Default behavior can mislead.
- **MMR is O(K²) per query** — fine for K=10 reranks. Don't try to MMR over 1000 candidates.
- **Synthetic relevance labels are dangerous.** If you train LTR on labels derived from BM25 scores, LTR learns to imitate BM25 — useless. Make sure your labels have an independent signal (clicks, topic match against ground-truth user interests).

### Interview questions this phase prepares you for
- "Why a VAE for user representation, not a transformer or simple averaging?"
- "How does Mult-VAE handle the cold-start problem?"
- "What is posterior collapse and how did you prevent it?"
- "How do you balance personalization against filter bubbles?"
- "What metrics did you use beyond NDCG?"

---

## Phase 3: Generation with QLoRA (Weeks 5–6)

### Goals
- LoRA training corpus built (real + synthetic)
- QLoRA fine-tuning of Phi-3-mini
- Generation evaluation with ROUGE, BERTScore, small human eval
- Inference wrapped in FastAPI endpoint

### Concepts to master before coding

- **LoRA and QLoRA** — original LoRA paper (Hu et al. 2021), QLoRA paper (Dettmers et al. 2023). Understand: why low-rank matrices, why 4-bit quantization works, what NF4 vs FP4 is, what double quantization does.
- **Instruction tuning data format** — Phi-3 uses a specific chat template. Get this right or training silently fails to teach the model anything.
- **Generation evaluation** — ROUGE is shallow (n-gram overlap), BERTScore is semantic (cosine sim of token embeddings). Both have failure modes. Human eval, even small, is necessary.
- **Prompt engineering as a baseline** — before training, see how far you get with just good prompting. Surprisingly far. The LoRA bump should be measurable but not magical.

### Week 5 task list

1. Build `generation/prompts.py`: three prompt templates (undergrad / grad / researcher level).
2. Build `generation/synth_data.py`: take 5K ArXiv abstracts, call Claude/GPT-4 API to generate level-stratified summaries. Save as JSONL.
3. Combine with SciTLDR → ~10K total training examples.
4. Build `generation/train_lora.py`: QLoRA on Phi-3-mini-4k-instruct, batch size 1, gradient accumulation 8, gradient checkpointing on, learning rate 2e-4, ~3 epochs.
5. Run training (~6–10 hours on RTX 2080). Monitor loss curve. Validate every 200 steps.
6. Save the LoRA adapter (~25MB).

### Week 6 task list

1. Build `generation/inference.py`: load base model + adapter, generation wrapper with stop tokens etc.
2. Build `generation/eval.py`: ROUGE-L, BERTScore on SciTLDR test split.
3. Run small human eval: pick 50 outputs across three levels, rate on (factuality, level appropriateness, glossary usefulness). Save to spreadsheet.
4. Compare: base Phi-3-mini vs LoRA Phi-3-mini vs GPT-3.5 (ceiling). LoRA should close 40-60% of the gap.
5. Wire the whole pipeline together in `api/routes/explain.py`.
6. Build `ui/gradio_app.py`: search box, user-level dropdown, results cards, "Fresh Start" button, diversity slider.
7. Make the demo screenshot/GIF.

### Phase 3 deliverables
- LoRA adapter checkpoint
- Generation eval table
- End-to-end Gradio demo
- Recorded 2-minute Loom walkthrough
- ADR-004 committed
- ~10 commits

### Gotchas
- **QLoRA OOM:** even with all the tricks, Phi-3-mini at seq_len 1024 + batch 1 is tight on 8GB. If you OOM: drop seq_len to 768, use paged_adamw_8bit optimizer, reduce LoRA rank from 16 to 8. If still OOM: fall back to Phi-1.5 or TinyLlama.
- **Chat template:** Phi-3 has a specific format with `<|user|>` and `<|assistant|>` tokens. Use `tokenizer.apply_chat_template`, not manual concatenation. Getting this wrong silently breaks training.
- **Synth data quality:** if your synthesized summaries are bad, your LoRA learns garbage. Spot-check 100 samples manually before training.
- **Don't evaluate ROUGE in isolation.** A model that copies the first sentence of the abstract often wins ROUGE without producing a useful summary. Pair with BERTScore and human eval.

### Interview questions this phase prepares you for
- "Explain LoRA — why does training only low-rank matrices work?"
- "Why QLoRA over plain LoRA?"
- "How did you build your training data, and what's the bias risk from using a teacher model?"
- "How did you evaluate a generation model — what does each metric tell you, and where does each fail?"
- "If you had 100x more compute, what would change?"

---

## Phase 4: Polish + Ship (overlaps with Week 6)

### Tasks
- Write the README using the template
- Record demo GIF and Loom walkthrough
- Deploy demo (Fly.io, Railway, or HuggingFace Spaces — all have free tiers)
- Write the blog post (use the design doc as the skeleton)
- Polish ADRs
- Final commit: tagged v1.0.0

### Stretch (if time)
- Replace Gradio with React frontend
- Add a small contextual bandit (LinUCB) as the production-grade exploration layer
- Build a citation-graph view (papers as nodes, citations as edges, color by topic)
- Add real-time incremental Mult-VAE updates (extension paper, ~2 days of work)

---

## Cross-cutting: How to use Claude Code throughout

The single rule: **you write the design, Claude writes the boilerplate, you review every line.**

### Session loop (every time you sit down to code)

1. Open `learning-log.md`. Write what you're trying to build today in 2 sentences.
2. Open Claude Code in the repo. Ask: "Plan the implementation of X. List files to touch, functions to add, tests to write. Don't write code yet."
3. Read the plan. Push back. Adjust. Then say "implement."
4. As code lands, **write the commit message yourself**. If you can't write it without re-reading the code, stop and ask Claude to explain.
5. After non-trivial code: close Claude, open `learning-log.md`, write 5–10 lines explaining the new code in your own words.
6. Every decision worth defending → ADR in `docs/decisions.md` (10 lines: context, options, decision, consequences).

### One day a week: no-Claude day

Sunday or whenever fits. Write 100 lines of project code from scratch. Even small. This is your calibration that you actually understand what's happening, not just orchestrating.

### Claude Code commands you'll use most

- `/plan` — for any new feature, get a plan first
- `/test` — generate tests for a module you just wrote
- `/explain <file>` — when re-reading code you wrote 2 weeks ago

### When NOT to use Claude

- Writing your README's "What I learned" section
- Writing ADRs
- Writing interview prep answers
- Writing the blog post

These are *your* artifacts. They have to sound like you, not like Claude.

---

## Daily rhythm (aligned with your overall placement plan)

- **Morning:** classes + class assignments (your courses)
- **2-6pm:** course study + DSA (2 problems/day, rotating topics)
- **7-10pm:** Scholar (this project)
- **Sat full day:** Scholar deep work
- **Sun morning:** Scholar deep work, afternoon: review, plan next week, DL specialization
- **Late night, 30 min:** OS/DBMS/ML revision (separate from this project)

If you fall behind: drop Phase 4 stretch items first, then drop one baseline comparison, then drop the human eval (keep ROUGE+BERTScore alone). Do NOT drop the design doc, ADRs, or learning log — those *are* the interview prep.

---

## Final shipping checklist

Before declaring v1.0.0:

- [ ] All ADRs written (≥6)
- [ ] Benchmarks table populated for all 3 layers
- [ ] Filter-bubble simulation plot in the README
- [ ] Demo deployed and link works
- [ ] Loom walkthrough recorded (≤3 min)
- [ ] Blog post drafted (≥1500 words)
- [ ] INTERVIEW.md filled with rehearsed answers
- [ ] Repo README polished
- [ ] Code passes `ruff` and `mypy`
- [ ] Unit tests pass and coverage ≥ 60% on the core modules
- [ ] You can explain every file in the repo without re-reading it
