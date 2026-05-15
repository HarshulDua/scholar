# Serving and UI — Complete Deep Dive

> Covers: ASGI, FastAPI architecture, async/await, Pydantic, Docker internals,
> Docker Compose, Gradio UI patterns, and Scholar's complete API and frontend implementation.

---

## 1. The Serving Problem

You've built a 3-layer ML pipeline in Python. How do you expose it to users?

**Options:**
1. **Script + CLI** — run `python search.py "my query"` — no concurrency, no web
2. **Flask (WSGI)** — simple, but blocks during I/O (one request at a time per worker)
3. **FastAPI (ASGI)** — async-native, OpenAPI docs auto-generated, Pydantic validation, high throughput
4. **gRPC** — binary protocol, better for internal microservices, harder to debug

Scholar uses FastAPI because:
- Non-blocking I/O: while waiting for PostgreSQL, other requests are processed
- Pydantic integration: request/response schemas validated automatically
- Auto-generated Swagger UI: `/docs` gives a working API explorer, free
- Production-grade: powers Uber, Netflix, and many ML serving platforms

---

## 2. WSGI vs ASGI — The Core Difference

### WSGI (Synchronous)

```
Flask/Django request handling:
1. Request arrives
2. Worker thread picks it up — all other requests queue
3. Worker calls DB query (blocks: 5ms waiting for DB response)
4. Worker returns response
5. Worker now free for next request

With 4 workers: max 4 concurrent requests.
During a 5ms DB wait: 1 worker is blocked, 3 serve others.
```

### ASGI (Asynchronous)

```
FastAPI/Uvicorn request handling:
1. Request arrives → coroutine starts
2. await DB query — coroutine SUSPENDS (not blocking!)
3. Event loop handles OTHER requests while waiting
4. DB responds → coroutine RESUMES
5. Returns response

With 1 worker: handles hundreds of concurrent requests.
During a 5ms DB wait: 0 threads blocked, event loop serves all others.
```

**asyncio event loop:**
```python
import asyncio

async def fetch_papers(query: str) -> list:
    # "await" suspends this coroutine, yields control to event loop
    result = await db.execute("SELECT * FROM papers WHERE ...")
    return result

async def main():
    # gather() runs coroutines concurrently — all 3 fire, event loop handles them
    results = await asyncio.gather(
        fetch_papers("transformers"),
        fetch_papers("diffusion"),
        fetch_papers("rl"),
    )
    # Total time: max(3 DB queries), not sum
```

**When async doesn't help:** CPU-bound work (like numpy matrix multiplication in BM25/FAISS)
blocks the event loop even with async. Scholar uses `asyncio.to_thread()` to run CPU-heavy
operations in a thread pool, keeping the event loop free.

---

## 3. FastAPI Architecture

### Application Setup with Lifespan

```python
# scholar/api/main.py
import pyarrow as _pa   # FIRST import — prevents DLL conflict on Windows
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP (runs once before serving) ──────────────────────
    _create_tables_sync()   # create Postgres tables via psycopg2

    # Load indexes into app.state (shared across all requests)
    app.state.bm25 = BM25.load(settings.bm25_path)
    app.state.dense = DenseRetriever()
    app.state.dense.load()   # loads 307MB embeddings.npy into RAM

    # Precompute id_to_idx once (avoid rebuilding per-request)
    app.state.id_to_idx = {
        pid: i for i, pid in enumerate(app.state.dense._paper_ids)
    }

    # Load LTR model
    app.state.ltr_model = LTRModel.load(settings.ltr_model_path)

    # Optionally warmup Phi-3
    if settings.warmup_generation:
        Phi3Generator.get_instance()   # loads 1.9GB model into VRAM

    yield   # ← server is now running and serving requests

    # ── SHUTDOWN (runs once after all requests complete) ─────────
    # Clean up resources if needed (DB connections auto-closed)

app = FastAPI(title="Scholar API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route groups
app.include_router(search_router, tags=["search"])
app.include_router(recommend_router, tags=["recommend"])
app.include_router(explain_router, tags=["explain"])
app.include_router(user_router, tags=["user"])

@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Why `app.state`?** FastAPI's `app.state` is a namespace shared across all requests.
BM25 and the dense index are loaded once at startup and accessed via `request.state.*` in routes.
This avoids reloading 307MB on every search request.

### Pydantic Request/Response Schemas

```python
# scholar/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    level: Literal["undergrad", "grad", "researcher"] = "grad"
    top_k: int = Field(default=10, ge=1, le=50)
    user_id: Optional[str] = None

class PaperOut(BaseModel):
    id: str
    title: str
    abstract: str
    authors: str
    categories: str
    citation_count: int
    score: float   # relevance score from retrieval/ranking

class SearchResponse(BaseModel):
    papers: list[PaperOut]
    query: str
    total: int
```

**Pydantic validation:**
```python
# When client sends: {"query": "", "top_k": 100}
# Pydantic raises ValidationError automatically:
#   query: min_length 1 (got 0)
#   top_k: le 50 (got 100)
# FastAPI converts this to HTTP 422 Unprocessable Entity with error details
# No manual input validation needed in route handlers
```

### Search Route

```python
# scholar/api/routes/search.py
@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    request: Request,          # FastAPI injects the HTTP request object
    background_tasks: BackgroundTasks,  # for fire-and-forget logging
    session: AsyncSession = Depends(get_async_session),  # DB session injection
):
    bm25 = request.state.bm25
    dense = request.state.dense

    # Run BM25 and dense retrieval concurrently
    bm25_results, dense_results = await asyncio.gather(
        asyncio.to_thread(bm25.query, req.query, top_k=200),    # CPU-bound → thread
        asyncio.to_thread(dense.query, req.query, top_k=200),   # CPU-bound → thread
    )

    # RRF fusion
    hybrid_results = reciprocal_rank_fusion(bm25_results, dense_results, k=60, top_k=200)
    top_ids = [pid for pid, _ in hybrid_results[:req.top_k]]

    # Fetch paper metadata from Postgres
    papers_map = await _get_papers_by_ids(session, top_ids)

    # Build response
    papers_out = [
        PaperOut(**paper.__dict__, score=hybrid_results[i][1])
        for i, pid in enumerate(top_ids)
        if pid in papers_map
        for paper in [papers_map[pid]]
    ]

    # Fire-and-forget: log click events (doesn't delay the response)
    background_tasks.add_task(
        _log_search_clicks, session, req.user_id, req.query, papers_out
    )

    return SearchResponse(papers=papers_out, query=req.query, total=len(papers_out))
```

### BackgroundTasks — Fire and Forget

```python
# Background tasks run AFTER the response is sent
# The user gets their results immediately; logging happens asynchronously

async def _log_search_clicks(session_factory, user_id, query, results):
    async with session_factory() as session:
        for rank, paper in enumerate(results, start=1):
            session.add(ClickLog(
                user_id=user_id,
                paper_id=paper.id,
                query=query,
                rank_position=rank,
            ))
        await session.commit()

# In the route:
background_tasks.add_task(
    _log_search_clicks, request.state.async_session_factory,
    req.user_id, req.query, papers_out
)
```

**When to use BackgroundTasks vs asyncio.create_task():**
- `BackgroundTasks`: runs after the response, managed by FastAPI/Starlette. Good for
  logging, cache warming, email notifications.
- `asyncio.create_task()`: starts immediately in the background, doesn't wait for response.
  Good for parallel work needed before the response.

---

## 4. Dependency Injection

```python
# scholar/db/__init__.py
from sqlalchemy.ext.asyncio import AsyncSession

async def get_async_session(request: Request) -> AsyncSession:
    session_factory = request.state.async_session_factory
    async with session_factory() as session:
        yield session   # session is auto-closed after the request
        # If exception: session is rolled back automatically

# In routes:
async def search(session: AsyncSession = Depends(get_async_session)):
    # session is ready to use, lifecycle managed by FastAPI
    result = await session.execute(select(Paper).where(...))
```

**Why dependency injection?**
- Routes don't need to know HOW to create a session — just declare they need one
- Session lifecycle (open/close/rollback) is handled in one place
- Testing: inject a mock session without changing route code
- `Depends()` calls the dependency function for each request, closes it after

---

## 5. Error Handling

```python
from fastapi import HTTPException

@router.get("/user/{user_id}")
async def get_user(user_id: str, session: AsyncSession = Depends(...)):
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    if user is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user

# FastAPI converts HTTPException to:
# HTTP 404 {"detail": "User abc not found"}
```

```python
# 503 when index not loaded
if not hasattr(request.state, "bm25") or request.state.bm25 is None:
    raise HTTPException(
        status_code=503,
        detail="Search index not loaded. Run scripts/build_index.py first."
    )
```

---

## 6. Docker — Containerization

### Core Theory

**What a container is:**
A container is a process running with Linux kernel isolation:
- **Namespaces:** each container gets its own view of process IDs, network interfaces,
  filesystem, hostname — looks like its own machine
- **cgroups:** limits CPU, memory, I/O for the container
- **OverlayFS:** layered filesystem — base image layers are shared read-only across containers,
  each container gets its own writable layer on top

**Container vs VM:**
```
VM: hypervisor virtualizes hardware → guest OS boots → apps run
  Overhead: ~1GB RAM for guest OS, seconds to boot, full OS processes

Container: host OS kernel shared → container gets isolated namespaces
  Overhead: ~10MB, milliseconds to start, no guest OS
```

### Scholar's Dockerfile

```dockerfile
# CPU-only Dockerfile (Dockerfile)
FROM python:3.11-slim

# Layer ordering: least-changing first for cache efficiency
WORKDIR /app

# Install system deps first (changes rarely)
RUN apt-get update && apt-get install -y libpq-dev gcc curl && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files before application code
# → Docker caches this layer. If only code changes, deps aren't reinstalled
COPY pyproject.toml .
RUN uv pip install --system -e ".[dev]"

# Copy application code last (changes most often)
COPY scholar/ ./scholar/
COPY scripts/ ./scripts/
COPY .env.example .env

EXPOSE 8000

CMD ["uvicorn", "scholar.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

**Layer caching:** Docker builds images as a stack of layers. If `pyproject.toml` hasn't changed,
the `RUN uv pip install` layer is cached — rebuilds take seconds instead of minutes.
**Copy `pyproject.toml` before `COPY . .`** — otherwise any code change invalidates the deps layer.

### Docker Compose — Multi-Container Orchestration

```yaml
# docker-compose.yml
version: "3.9"
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: scholar
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-changeme}
      POSTGRES_DB: scholar
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data    # named volume: data persists across restarts
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U scholar"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data       # bind mount: local ./data/ maps to /app/data in container
      - ./.env:/app/.env       # inject secrets without rebuilding
    environment:
      - DATABASE_URL=postgresql+asyncpg://scholar:${POSTGRES_PASSWORD:-changeme}@db:5432/scholar
    depends_on:
      db:
        condition: service_healthy   # wait for postgres healthcheck to pass

  ui:
    build: .
    command: ["python", "scholar/ui/gradio_app.py"]
    ports:
      - "7860:7860"
    depends_on:
      - api

volumes:
  pgdata:    # named volume definition — Docker manages its location
```

**Networking:** Docker Compose creates a default bridge network. Services find each other
by service name (`db`, `api`, `ui`) as DNS hostnames. The API container connects to
`postgresql://scholar:...@db:5432/scholar` — `db` resolves to the Postgres container's IP.

**Bind mount vs named volume:**
```
Named volume (pgdata):
  - Managed by Docker: stored in /var/lib/docker/volumes/
  - Persists between container restarts and docker compose down
  - Best for: database files you want to keep

Bind mount (./data:/app/data):
  - Maps a host directory into the container
  - Changes visible immediately from both sides
  - Best for: model files, indexes, development code
```

---

## 7. GPU Docker

```yaml
# docker-compose.gpu.yml — override for CUDA environment
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.cuda   # CUDA 12.1 base image
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - DEVICE=cuda
      - GENERATION_AVAILABLE=true
```

```dockerfile
# Dockerfile.cuda
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip libpq-dev gcc
RUN pip3 install uv

WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system -e ".[dev]"  # installs torch+cuda variant
COPY scholar/ ./scholar/
COPY scripts/ ./scripts/

CMD ["uvicorn", "scholar.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Running:** `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`

NVIDIA Container Toolkit maps the host GPU into the container via `/dev/nvidia*` devices
and the `nvidia` driver volume. The container sees a real CUDA device.

---

## 8. Gradio — Scholar's UI

### Architecture

```
User browser ←→ WebSocket ←→ Gradio (FastAPI under the hood) ←→ Python handler
```

Gradio is built on top of FastAPI + WebSockets. When you click "Search", a WebSocket
message fires, the Python handler runs, and results stream back to the browser.

### Scholar's Gradio App Structure

```python
# scholar/ui/gradio_app.py
import gradio as gr
import httpx

API_URL = "http://api:8000"  # Docker service DNS

async def search_papers(query: str, level: str, user_id: str) -> str:
    """Call Scholar API and return formatted HTML."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{API_URL}/search", json={
            "query": query,
            "level": level,
            "user_id": user_id or None,
            "top_k": 10,
        })
        resp.raise_for_status()
        data = resp.json()

    # Format papers as HTML cards
    cards = []
    for paper in data["papers"]:
        cards.append(f"""
        <div style="border:1px solid #ddd; padding:16px; margin:8px 0; border-radius:8px">
          <h4>{paper['title']}</h4>
          <p><strong>Authors:</strong> {paper['authors'][:100]}</p>
          <p><strong>Categories:</strong> {paper['categories']}</p>
          <p>{paper['abstract'][:300]}...</p>
          <p><em>Score: {paper['score']:.3f} | Citations: {paper['citation_count']}</em></p>
        </div>
        """)
    return "<hr>".join(cards) if cards else "No results found."

async def explain_paper(paper_id: str, level: str) -> str:
    """Get AI-generated explanation for a paper."""
    async with httpx.AsyncClient(timeout=120.0) as client:   # long timeout for LLM
        resp = await client.post(f"{API_URL}/explain", json={
            "paper_id": paper_id,
            "level": level,
        })
        if resp.status_code == 503:
            return "Explanation service not available (Phi-3 not loaded or VRAM constraint)."
        resp.raise_for_status()
        return resp.json().get("explanation", "No explanation generated.")

# Build the Gradio UI
with gr.Blocks(title="Scholar — ArXiv Paper Discovery") as demo:
    gr.Markdown("# Scholar — Personalized Research Paper Discovery")

    with gr.Row():
        with gr.Column(scale=2):
            query_box = gr.Textbox(label="Search Query", placeholder="diffusion models for image generation")
        with gr.Column(scale=1):
            level_select = gr.Dropdown(["undergrad", "grad", "researcher"], value="grad", label="Your Level")

    user_state = gr.State(value="")   # gr.State: per-user session state (not shared across users)

    with gr.Row():
        search_btn = gr.Button("Search", variant="primary")
        create_user_btn = gr.Button("Create User (Enable Personalization)")

    user_id_display = gr.Textbox(label="Your User ID", interactive=False)
    results_html = gr.HTML(label="Results")

    # Events
    search_btn.click(
        fn=search_papers,
        inputs=[query_box, level_select, user_state],
        outputs=[results_html],
    )

    create_user_btn.click(
        fn=create_user,       # calls POST /user, returns user_id
        outputs=[user_state, user_id_display],
    )

    with gr.Tab("Explain Paper"):
        paper_id_box = gr.Textbox(label="Paper ID")
        explain_btn = gr.Button("Get AI Explanation")
        explanation_out = gr.Markdown()
        explain_btn.click(
            fn=explain_paper,
            inputs=[paper_id_box, level_select],
            outputs=[explanation_out],
        )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

### gr.State — Session Isolation

```python
user_state = gr.State(value="")   # starts as empty string

# Each browser session gets its OWN copy of user_state
# User A's user_id is not visible to User B
# This is NOT a global variable — gr.State is per-connection
```

**Why this matters:** if you used `user_id = ""` as a global Python variable, every user
would share the same ID — a serious bug. `gr.State` creates a separate value for each
connected browser tab.

### Async Handlers

```python
# Gradio supports async handlers natively
async def search_papers(query, level, user_id) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(...)    # non-blocking HTTP
```

Without async, Gradio's thread pool would block while waiting for the API. With async handlers,
Gradio's event loop handles multiple users concurrently.

---

## 9. Scholar API Endpoints — Complete Reference

```
GET  /health
  → {"status": "ok"}

POST /search
  Body: {"query": str, "level": str, "top_k": int, "user_id": str|null}
  → {"papers": [...], "query": str, "total": int}
  Flow: BM25 + dense → RRF → Postgres fetch → BackgroundTask logging

POST /recommend
  Body: {"user_id": str, "limit": int}
  → {"papers": [...], "user_id": str}
  Flow: Mult-VAE embedding → dense retrieval → LTR → MMR → ε-greedy

POST /explain
  Body: {"paper_id": str, "level": str, "why": str}
  → {"explanation": str, "paper_id": str}
  Flow: fetch paper → build prompt → Phi3Generator.generate()

GET  /user/{user_id}
  → {"id": str, "created_at": str, "history_count": int}

POST /user
  → {"id": str, "created_at": str}   # creates new user with UUID

POST /user/{user_id}/history
  Body: {"paper_id": str}
  → {"added": true}
  Side effect: writes to user_history table, used by Mult-VAE
```

**OpenAPI docs:** `http://localhost:8000/docs` — auto-generated from Pydantic schemas.
Interactive: click "Try it out" on any endpoint, fill in the JSON, click Execute.

---

## 10. Performance Characteristics

| Endpoint | p50 | p99 | Bottleneck |
|----------|-----|-----|------------|
| /health | <1ms | <1ms | None |
| /search | ~120ms | ~200ms | Dense retrieval (numpy matmul) |
| /recommend | ~140ms | ~230ms | Dense retrieval + LTR feature extraction |
| /explain | 8-15s | 30s | Phi-3 generation (GPU), 45s+ (CPU) |

**Cold start:** First `/explain` request triggers Phi-3 load (~2-4 min). Subsequent: ~10s.
`WARMUP_GENERATION=true` pre-loads Phi-3 at server startup.

---

## 11. Interview Questions and Answers

**Q: What is ASGI and why does it matter for Scholar?**

A: ASGI (Asynchronous Server Gateway Interface) is the async upgrade to WSGI. Scholar's
retrieval involves multiple I/O-bound operations: two DB queries, potentially three model
inference calls. With WSGI (Flask/Django), each request blocks one thread — during a 5ms DB
query, that thread does nothing. With ASGI (FastAPI/uvicorn), the coroutine suspends during
`await db.execute(...)` and the event loop handles other requests. For Scholar serving
10 concurrent users, this means we can handle all 10 with one worker thread instead of needing
10 threads.

**Q: How does Scholar handle concurrent requests to the heavy /search endpoint?**

A: Two mechanisms. BM25 and dense retrieval are CPU-bound, so they can't await — they'd block
the event loop. Scholar wraps them with `asyncio.to_thread()`: `bm25_results, dense_results = await asyncio.gather(asyncio.to_thread(bm25.query, ...), asyncio.to_thread(dense.query, ...))`. This runs both in the thread pool concurrently (true parallelism for CPU-bound work). The DB fetch is then a pure async operation. Result: two CPU-intensive operations run in parallel, and the event loop stays free.

**Q: Walk me through a Docker Compose startup from `docker compose up`.**

A: Docker Compose reads docker-compose.yml. It starts `db` first (pgvector/pgvector:pg16 image).
The db healthcheck (`pg_isready`) runs every 10s; `api` waits until it passes. Once healthy,
Docker Compose starts `api` — it builds the Dockerfile image, creates the container, maps port
8000, mounts `./data:/app/data`. The api container's startup (lifespan function) runs:
psycopg2 creates tables, BM25 loads from data/bm25.pkl (~100MB), embeddings.npy loads (~307MB).
Server starts serving. ui container starts last, connects to api via Docker's DNS (`http://api:8000`).

**Q: What is Pydantic and what does it validate in Scholar?**

A: Pydantic is a Python library for data validation using type hints. FastAPI uses it for
request/response schemas. When a client POSTs to /search, Pydantic validates: query is a
string with min_length=1 and max_length=500, level is one of ["undergrad", "grad", "researcher"],
top_k is an integer between 1 and 50. If validation fails, FastAPI returns HTTP 422 with a
structured error message — no manual validation code needed in route handlers. Pydantic also
validates the response: if a route returns a dict missing a required field from PaperOut,
it raises a validation error before sending, catching internal bugs.

**Q: What is gr.State in Gradio and why do you need it for user_id?**

A: gr.State is per-session state in Gradio — each browser connection gets its own independent
copy. If we stored user_id as a Python global variable, all users would share the same ID,
which would corrupt personalization. gr.State creates a separate value for each user's browser
tab, initialized to the default (empty string). When a user clicks "Create User" and gets a
UUID back, that UUID is stored in their gr.State — invisible to all other users.

---

## 12. World-Class Resources

**FastAPI:**
- [FastAPI docs](https://fastapi.tiangolo.com/) — Sebastián Ramírez's comprehensive guide
- [Starlette docs](https://www.starlette.io/) — FastAPI's underlying ASGI framework
- [asyncio docs](https://docs.python.org/3/library/asyncio.html) — official Python async reference

**Docker:**
- [Docker docs — Dockerfile best practices](https://docs.docker.com/build/building/best-practices/)
- [Docker Compose docs](https://docs.docker.com/compose/)
- [NGINX NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) — GPU in Docker

**Gradio:**
- [Gradio docs](https://www.gradio.app/docs) — component reference and event system
- [Gradio + FastAPI integration](https://www.gradio.app/guides/sharing-your-app) — deploying to HuggingFace Spaces
- [Gradio streaming](https://www.gradio.app/guides/streaming-outputs) — for LLM text generation
