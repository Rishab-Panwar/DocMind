# Imp. concepts

---

## 1. Caching

**What:** Process-wide in-memory LLM cache using LangChain's `InMemoryCache`.  
**File:** `utils/llm_cache.py`, initialized in `api/main.py:37` (FastAPI lifespan) and `utils/model_loader.py:117` (ModelLoader constructor).

**How it works:**
- `set_llm_cache(InMemoryCache())` registers a global cache for all LangChain LLM calls.
- Cache key = full formatted prompt + LLM config (model, temperature). Identical prompt → served from cache, no API call.
- `_INITIALIZED` flag makes it idempotent — safe to call from multiple features.
- `LLM_CACHE_ENABLED=false` env toggle disables it without code changes.
- Cache clears on restart — it's a Python dict in RAM, no disk persistence.

**How to explain:**
> "LLM calls are expensive and slow. I implemented a process-wide in-memory cache so identical prompts are served from cache instead of hitting the API again. It's registered as a LangChain global, so all three features benefit automatically. I also added an env toggle so it can be disabled in testing."

**Most impactful for:** Document Chat (short repeated questions). For Analyze and Compare, the full document text is embedded in the prompt, so a cache hit only occurs if the exact same document is submitted again — an edge case, not the common path.

**Follow-ups:**
- Why not Redis? → In-memory is zero-infra for a single process. Redis is the natural upgrade for multi-instance deployments.
- Cache invalidation? → LangChain matches on exact prompt+model key. No explicit eviction needed — restart clears it.

---

## 2. Logging

**What:** Structured JSON logging using `structlog`, written to both console and a timestamped file.  
**File:** `logger/custom_logger.py`, exported as `GLOBAL_LOGGER` from `logger/__init__.py`.

**How it works:**
- Every log line is a JSON object with `timestamp`, `level`, `event`, and contextual key-value pairs.
- One shared `GLOBAL_LOGGER` instance imported everywhere — format and destination controlled in one place.
- Logs go to `logs/MM_DD_YYYY_HH_MM_SS.log` (new file per process start) AND stdout.
- `session_id` is passed manually at key boundary points (session creation, file save, chain invocation, final response). Internal utility logs (API key loading, model loading) don't carry it.

**How to explain:**
> "Instead of plain print statements or format strings, I used structlog to emit machine-readable JSON logs. Each log call passes structured context as key-value pairs, making logs searchable and parseable by tools like Datadog or CloudWatch. There's one global logger instance so the format is consistent across all features."

**Known gap:**
> "Session ID is logged at key boundaries but not on every line — internal utility code doesn't receive it. The fix would be structlog's `contextvars` binding to automatically inject session_id into every log line within a request scope."

**Follow-ups:**
- Why structlog over standard logging? → Standard logging buries context in message strings. Structlog emits JSON natively — fields are queryable.
- How to trace a single request? → Grep the log file for a `session_id` value.

---

## 3. Exception Handling

**What:** Custom `DocumentPortalException` that auto-captures file name, line number, and full traceback.  
**File:** `exception/custom_exception.py`, used across all features.

**How it works:**
- Walks the traceback to the **last frame** (where the error actually occurred) and extracts `co_filename` and `tb_lineno`.
- Stores full `traceback.format_exception(...)` string inside the exception.
- Accepts `sys`, an `Exception` object, or nothing (auto-captures via `sys.exc_info()`).
- Pattern everywhere: `log.error(...)` first, then `raise DocumentPortalException(...)` — structured log entry + rich exception propagates to API layer.

**Output format:**
```
Error in [src/document_analyzer/data_analysis.py] at line [57] | Message: Metadata extraction failed
Traceback:
  ...full stack trace...
```

**How to explain:**
> "Instead of bare `raise Exception('something failed')`, I built a custom exception that automatically captures the exact file and line number where the error occurred, plus the full stack trace. When it's logged or hits the API error handler, you immediately know where things broke without digging through raw tracebacks."

**Follow-ups:**
- Why walk to the last frame? → The last frame is where the error happened, not where it was caught — more actionable.
- Why not just `logging.exception()`? → That logs the traceback but doesn't carry it in the exception itself. My approach wraps it so callers also get full context if they catch and re-raise.
- What does the API layer do with it? → Propagates to FastAPI's exception handler which returns a structured error response.

---

## 4. Evaluation (DeepEval)

**What:** LLM evaluation using DeepEval framework across 6 RAG-specific metrics.  
**Files:** `eval/run_doc_chat_deepeval.py` (text RAG), `eval/run_mm_doc_chat_deepeval.py` (multimodal RAG).

**6 Metrics:**
| Metric | What it checks |
|---|---|
| AnswerRelevancy | Is the answer relevant to the question? |
| Faithfulness | Is the answer grounded in retrieved context? |
| ContextualPrecision | Are relevant chunks ranked at the top? |
| ContextualRecall | Did retrieval fetch all chunks needed to answer? |
| ContextualRelevancy | Are retrieved chunks relevant to the question? |
| Hallucination | Did the model introduce facts not in the context? |

**How the pipeline works:**
1. Ingest documents into FAISS using the same `ChatIngestor` as production
2. Pull golden Q&A pairs from Confident AI (DeepEval's cloud platform)
3. Run each question through the actual RAG pipeline → get `answer` + `retrieved_context`
4. Build `LLMTestCase` objects and score all 6 metrics via an LLM judge

**How to explain:**
> "I used DeepEval to evaluate the RAG pipeline across 6 metrics — 3 for retrieval quality (precision, recall, relevancy) and 3 for generation quality (answer relevancy, faithfulness, hallucination). Test cases are golden Q&A pairs stored in Confident AI, so evaluation is reproducible across runs. The eval also runs in CI on every push to main."

**Key distinction:**
> "Faithfulness and Hallucination sound the same but are different — Faithfulness checks if the answer is supported by the retrieved context, Hallucination checks if the model introduced facts that weren't in the context at all."

**Follow-ups:**
- How to run it? → `python eval/run_doc_chat_deepeval.py` after `deepeval login`
- Where are results? → Terminal score table + Confident AI dashboard with metric trends across runs.
- DeepEval jobs have `continue-on-error: true` in CI — eval failures don't block deployments.

---

## 5. Database

**Two separate storage systems:**

### SQLite — User Authentication
**File:** `auth/db.py`  
**What's stored:** User accounts (id, email, hashed password) via `fastapi-users` + SQLAlchemy.  
**Location:** `users.db` in project root — a local file.

- Async SQLAlchemy (`aiosqlite`) — doesn't block the FastAPI event loop.
- `DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./users.db")` — swap to Postgres in production with no code changes.

### FAISS — Document Vector Store
**Location:** `faiss_index/session_<id>/` — one folder per user session.

- Not a traditional DB — flat binary index file on disk.
- Created fresh per session when a user uploads documents.
- Each user's documents are isolated in their own session folder.

**How to explain:**
> "There are two storage layers. For authentication, SQLite via async SQLAlchemy — local file by default, but `DATABASE_URL` env var lets you point it at Postgres for production with no code changes. For documents, FAISS as a local vector index — each upload session gets its own folder, keeping users' data isolated."

**Follow-ups:**
- Why SQLite over Postgres from the start? → Zero-infra for development. The async driver integrates cleanly with FastAPI. Connection string is env-driven so upgrading is a config change, not a code change.
- Why not a managed vector DB like Pinecone? → FAISS is sufficient for single-server use and keeps everything local. Pinecone would be the upgrade path for multi-instance or large-scale deployments.

---

## 6. CI/CD

**Two GitHub Actions workflows:**

### CI (`ci.yml`) — runs on every push/PR to `dev` or `main`
| Job | What it does |
|---|---|
| `test` | Runs `pytest tests/` |
| `deepeval` | Runs RAG evaluation — `continue-on-error: true` |
| `deepeval-mm` | Runs multimodal evaluation — `continue-on-error: true` |

### CD (`aws.yml`) — triggers only after CI passes on `main`
```
Push to main → CI passes → Build Docker image → Push to ECR → Deploy to ECS Fargate
```

**AWS services:**
- **ECR** — stores Docker image (tagged with git commit SHA)
- **ECS Fargate** — runs the container serverlessly, no EC2 to manage
- `wait-for-service-stability: true` — pipeline waits to confirm new container is healthy

**How to explain:**
> "I have two GitHub Actions workflows. CI runs on every push — unit tests and DeepEval evaluation. CD triggers automatically only when CI passes on main — builds a Docker image tagged with the git commit SHA, pushes to ECR, updates the ECS Fargate task definition, and does a rolling deploy. Broken code can never deploy because CD is gated on CI."

**Key points:**
- Image tagged with `github.sha` — every deployment is traceable to an exact commit
- DeepEval failures don't block deployment (`continue-on-error: true`)
- `wait-for-service-stability: true` — confirms new container is healthy before marking deploy done
- CD only runs on `main` — feature branches only run tests, never deploy
