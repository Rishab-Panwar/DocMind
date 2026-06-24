<div align="center">

<img width="80" height="80" alt="ChatGPT Image May 29, 2026, 06_10_06 PM" src="https://github.com/user-attachments/assets/8f64a368-fdcd-472c-ae2e-8ca2911c2258">

# DocMind

*AI-powered document intelligence. Chat, analyze, compare, and query — in one platform.*

[Features](#features) · [Architecture](#system-architecture) · [Tech Stack](#tech-stack) · [Local Setup](#local-setup)

DocMind is an AI document platform with **conversational RAG**, **agentic query rewriting** via LangGraph, **multimodal PDF understanding**, structured metadata extraction, and a **built-in MCP server** for Claude Desktop and Cursor. Production-deployed on AWS ECS with a full CI/CD pipeline.

**Live:** [docmind.rishabai.me](https://docmind.rishabai.me)

</div>

---
<img width="1912" height="969" alt="image" src="https://github.com/user-attachments/assets/af0aad18-8596-48aa-a1a7-39a323097861" />

______
<img width="1916" height="966" alt="image" src="https://github.com/user-attachments/assets/74abb867-5eec-4d64-984c-45e9d0c0abbf" />


## Features

| Feature | Description |
|---|---|
| **Document Chat** | Conversational RAG with FAISS retrieval + FlashrankRerank + LangChain LCEL |
| **Agentic RAG** | LangGraph-powered agent that grades retrieved docs and rewrites queries when needed |
| **Multimodal Chat** | PDF chat with image, table, and text understanding via vision LLMs |
| **Document Analyzer** | Extracts structured metadata (title, summary, entities, dates) from any document |
| **Document Comparator** | Page-by-page diff between two documents with LLM-powered change detection |
| **MCP Server** | Connect Claude Desktop or Cursor directly to your indexed documents |

---

## System Architecture

```mermaid
graph TB
    subgraph Client["Client Layer"]
        Browser["Browser\n(Web UI)"]
        MCP["Claude Desktop / Cursor\n(MCP Client)"]
    end

    subgraph AWS["AWS Infrastructure"]
        ACM["ACM\nSSL Certificate"]
        ALB["Application Load Balancer\ndocmind.rishabai.me"]
        ECS["ECS Fargate\nFastAPI · Port 8080"]
        ECR["Amazon ECR\nDocker Registry"]
        SM["Secrets Manager\nAPI Keys"]
    end

    subgraph AppLayer["Application — api/main.py"]
        Auth["Auth\nfastapi-users · JWT · Cookie"]
        ChatIdx["POST /chat/index\nDocument Ingestion"]
        ChatQ["POST /chat/query\nConversational RAG"]
        AgentQ["POST /chat/agent_query\nAgentic RAG"]
        Analyze["POST /analyze\nDocument Analyzer"]
        Compare["POST /compare\nDocument Comparator"]
    end

    subgraph RAGLayer["RAG Pipeline"]
        Ingestor["ChatIngestor\nChunking · Embedding · FAISS"]
        LCEL["ConversationalRAG\nLCEL · FlashrankRerank"]
        LG["AgenticRAG — LangGraph\nretrieve → grade → rewrite → generate"]
        MM["MultiModal Pipeline\nunstructured · MultiVectorRetriever"]
    end

    subgraph Storage["Storage"]
        FAISS["FAISS Vector Index\nPer-session on disk"]
        SQLite["SQLite\nUser Auth DB"]
    end

    subgraph AI["AI / LLM Layer"]
        Embed["Google Gemini\ngemini-embedding-001"]
        LLM["LLM\nGroq · Google · OpenAI"]
        Cache["InMemoryCache\nLangChain LLM Cache"]
    end

    subgraph MCPSrv["MCP Server (Local)"]
        FastMCP["FastMCP\nask_document · simple_query · list_sessions"]
    end

    Browser -->|HTTPS| ACM
    ACM --> ALB
    ALB --> ECS
    ECS --> AppLayer
    ECR -->|Docker Image| ECS
    SM -->|API Keys| ECS

    Auth --- SQLite
    ChatIdx --> Ingestor
    ChatQ --> LCEL
    AgentQ --> LG
    ChatIdx --> MM

    Ingestor --> Embed
    Ingestor --> FAISS
    LCEL --> FAISS
    LG --> FAISS
    MM --> FAISS

    LCEL --> LLM
    LG --> LLM
    Analyze --> LLM
    Compare --> LLM
    LLM --- Cache

    MCP --> FastMCP
    FastMCP --> LG
```

---

## AWS Infrastructure

```mermaid
graph LR
    Dev["Developer\nPush to main"] --> GH["GitHub Actions\nCI/CD Pipeline"]

    subgraph CI["CI — ci.yml"]
        Tests["pytest\nUnit Tests"]
        DE["DeepEval\nRAG Evaluation"]
        DEMM["DeepEval MM\nMultimodal Evaluation"]
    end

    subgraph CD["CD — aws.yml (triggers after CI)"]
        Build["Build Docker Image\n(tagged with git SHA)"]
        Push["Push to ECR"]
        Deploy["Update ECS Task Definition\nRolling Deploy"]
        Wait["wait-for-service-stability"]
    end

    subgraph AWSInfra["AWS — us-east-1"]
        ECR2["Amazon ECR\nContainer Registry"]
        ECS2["ECS Fargate\nServerless Container"]
        ALB2["Application Load Balancer"]
        SM2["Secrets Manager\napikeyliveclass"]
        ACM2["ACM\nSSL Cert"]
        Route53["Namecheap DNS\ndocmind.rishabai.me"]
    end

    GH --> CI
    CI -->|Pass| CD
    Build --> Push
    Push --> ECR2
    ECR2 --> Deploy
    Deploy --> ECS2
    Deploy --> Wait
    ALB2 --> ECS2
    ACM2 --> ALB2
    Route53 -->|CNAME| ALB2
    SM2 -->|Inject at runtime| ECS2
```

---

## Tech Stack

### AI & LLM
| Component | Technology |
|---|---|
| LLM Providers | Groq (`llama-3.1-8b-instant`), Google (`gemini-2.5-flash`), OpenAI (`gpt-4o-mini`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) via REST |
| RAG Framework | LangChain LCEL |
| Agentic RAG | LangGraph (retrieve → grade → rewrite → generate) |
| Reranking | FlashrankRerank (cross-encoder, runs locally) |
| Multimodal | `unstructured`, `MultiVectorRetriever`, vision LLMs |
| LLM Cache | LangChain `InMemoryCache` — process-wide, env-toggleable |
| Evaluation | DeepEval — 6 RAG metrics (AnswerRelevancy, Faithfulness, Hallucination, Contextual Precision/Recall/Relevancy) |
| MCP Server | FastMCP — exposes RAG as tools for Claude Desktop / Cursor |

### Backend
| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Authentication | `fastapi-users` · JWT + Cookie backends · bcrypt |
| Vector Store | FAISS (per-session, disk-based) |
| Auth Database | SQLite via async SQLAlchemy (`aiosqlite`) |
| Logging | `structlog` — structured JSON logs, dual output (console + file) |
| Exception Handling | Custom `DocumentPortalException` — auto-captures file, line, full traceback |

### Infrastructure
| Component | Technology |
|---|---|
| Containerization | Docker |
| Container Registry | Amazon ECR |
| Compute | AWS ECS Fargate (serverless) |
| Load Balancer | AWS Application Load Balancer |
| SSL | AWS Certificate Manager (ACM) |
| Secrets | AWS Secrets Manager |
| DNS | Namecheap → CNAME → ALB |
| CI/CD | GitHub Actions (2 workflows: `ci.yml` + `aws.yml`) |

---

## Project Structure

```
document_p/
├── api/
│   └── main.py                    # FastAPI entrypoint — all routes
├── src/
│   ├── document_chat/
│   │   ├── retrieval.py           # ConversationalRAG (LCEL + reranker)
│   │   ├── agent_rag.py           # AgenticRAG (LangGraph)
│   │   └── multimodal/            # Multimodal RAG pipeline
│   ├── document_analyzer/         # Metadata extraction
│   ├── document_compare/          # Document diff
│   └── document_ingestion/        # Chunking, FAISS indexing
├── mcp_server/
│   └── server.py                  # FastMCP server (local use)
├── eval/
│   ├── run_doc_chat_deepeval.py   # DeepEval — text RAG
│   └── run_mm_doc_chat_deepeval.py # DeepEval — multimodal RAG
├── auth/                          # fastapi-users setup
├── utils/                         # ModelLoader, config, cache
├── prompt/                        # Prompt registry
├── logger/                        # structlog global logger
├── exception/                     # Custom exception
├── config/config.yaml             # LLM + embedding config
├── .github/workflows/
│   ├── ci.yml                     # Tests + DeepEval on every push
│   └── aws.yml                    # Build + Deploy on main
└── Dockerfile
```

---

## Local Setup

```bash
# 1. Clone and install
git clone https://github.com/Rishab-Panwar/documet_p.git
cd documet_p
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Fill in: GROQ_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY

# 3. Run the API
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open [http://localhost:8080](http://localhost:8080)

### MCP Server (Claude Desktop integration)

```bash
pip install mcp
python mcp_server/server.py
```

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "docmind": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server/server.py"]
    }
  }
}
```

Available tools: `ask_document` (agentic RAG), `simple_query` (fast RAG), `list_sessions`

---

## CI/CD Pipeline

```
Push to main
  └── ci.yml
        ├── test         → pytest tests/
        ├── deepeval     → RAG evaluation (non-blocking)
        └── deepeval-mm  → Multimodal RAG evaluation (non-blocking)
              └── [all pass] → aws.yml
                                ├── Build Docker image (tagged with git SHA)
                                ├── Push to Amazon ECR
                                ├── Update ECS Task Definition
                                └── Rolling deploy → wait-for-service-stability
```

- DeepEval jobs have `continue-on-error: true` — eval failures never block deployment
- Docker image tagged with `${{ github.sha }}` — every deploy traceable to a commit
- CD only runs on `main` — feature branches run tests, never deploy

---

## Evaluation — DeepEval

RAG quality is measured on every CI run using [DeepEval](https://github.com/confident-ai/deepeval):

| Metric | Measures |
|---|---|
| Answer Relevancy | Is the answer relevant to the question? |
| Faithfulness | Is the answer grounded in retrieved context? |
| Contextual Precision | Are relevant chunks ranked at the top? |
| Contextual Recall | Were all necessary chunks retrieved? |
| Contextual Relevancy | Are retrieved chunks relevant? |
| Hallucination | Did the LLM introduce facts not in context? |

Judge model: `llama-3.3-70b-versatile` via Groq
