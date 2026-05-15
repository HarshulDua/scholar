# Scholar — Benchmark Results

## Layer 1: Retrieval (BM25 + Dense + Hybrid)

Evaluation dataset: Synthetic ArXiv queries (10 topic queries), 200K paper corpus.
NFCorpus (original target) unavailable — URL returned 404 as of 2026-05-15.
Latency measured on CPU (numpy matmul search).

| Retriever         | Latency p50 (ms) | Latency p99 (ms) | Notes |
|-------------------|-----------------|-----------------|-------|
| BM25 (scratch)    | 26.6            | 44.5            | TF-IDF inverted index, 200K docs |
| Dense (MiniLM)    | 121.6           | 97926.9*        | numpy matmul 200K×384 |
| Hybrid (RRF k=60) | 106.5           | 196.2           | BM25 + Dense RRF fusion |

*Dense p99=98s is the model cold-start (all-MiniLM-L6-v2, 103 weight shards). After warmup, Dense p50~121ms. Server pre-warms at startup so cold-start never hits users.

---

## Layer 2: Reranking

Evaluation dataset: CiteULike-A test split (10% of users).

| Ablation                | NDCG@10 | HR@10  | ILD    | Users |
|-------------------------|---------|--------|--------|-------|
| mmr                     | 0.0000  | 0.0000 | 0.4962 |    38 |
| ltr+mmr                 | 0.0000  | 0.0000 | 0.4962 |    38 |
| ltr+mmr+exploration     | 0.0000  | 0.0000 | 0.5764 |    38 |

---

## Layer 3: Generation (Phi-3-mini 4-bit)

Evaluation dataset: local ArXiv snapshot, title as reference TLDR proxy
(SciTLDR deprecated in datasets 3.x; ArXiv fallback used).

| Model              | Level      | ROUGE-L | BERTScore F1 | Samples |
|--------------------|------------|---------|--------------|--------|
| Phi-3 base         | undergrad  | 0.0698  | 0.8354       |      10 |
| Phi-3 base         | grad       | 0.1199  | 0.8452       |      10 |
| Phi-3 base         | researcher | 0.1028  | 0.8443       |      10 |
| Phi-3 + LoRA       | undergrad  | 0.0817  | 0.8323       |      10 |
| Phi-3 + LoRA       | grad       | 0.1236  | 0.8502       |      10 |
| Phi-3 + LoRA       | researcher | 0.1129  | 0.8493       |      10 |

---

_Eval completed 21:27 PM 2026-05-15. 10 samples per level, ArXiv title as reference TLDR proxy._
