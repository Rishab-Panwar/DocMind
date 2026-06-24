import os
import re
import sys
from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log


# Tight patterns for EXPLICIT "list the indexed files" intent. Each is a phrase
# a summary/content question never contains, so this stays robust to typos
# (e.g. "sumarize both the files uploaded" does NOT match). When matched, we
# answer deterministically from the manifest so document content that happens to
# list filenames (a codebase doc, a Chroma DB) can't hijack the answer.
_LIST_PATTERNS = re.compile(
    r"""(?ix)
      \blist\s+(?:all\s+|the\s+)*(?:files?|docs?|documents?)
    | \bnames?\s+(?:of\s+)?(?:all\s+|the\s+|uploaded\s+|indexed\s+)*(?:files?|docs?|documents?)
    | \bhow\s+many\s+(?:files?|docs?|documents?)
    | \b(?:which|what)\s+(?:files?|docs?|documents?)\s+
        (?:do\s+you\s+have|have\s+you|are\s+(?:there|indexed|uploaded|available|loaded)|exist)
    | \b(?:files?|documents?)\s+(?:are\s+)?indexed\b
    """
)


def _is_file_list_query(q: str) -> bool:
    """True only for explicit 'list the indexed files' style questions."""
    return bool(q) and bool(_LIST_PATTERNS.search(q))


def _format_manifest(names: List[str]) -> str:
    """Render the list of indexed filenames as a compact reference line for the
    LLM, so 'which/how many files' questions can be answered completely
    regardless of phrasing — no brittle intent routing required."""
    if not names:
        return "(no files indexed)"
    return str(len(names)) + " file(s): " + ", ".join(names)


def contextualize_question(llm, question: str, history) -> str:
    """Rewrite a follow-up into a standalone question using recent chat history,
    so references like 'it'/'that'/'those' resolve. `history` is a list of
    {role, content} dicts. Returns the question unchanged if there is no history
    or on any error — so memory never breaks a question, it only enriches it."""
    if not history:
        return question
    lines = []
    for turn in history[-12:]:
        content = str((turn or {}).get("content", "")).strip()
        if not content:
            continue
        who = "User" if (turn.get("role") == "user") else "Assistant"
        lines.append(f"{who}: {content}")
    if not lines:
        return question
    prompt = ChatPromptTemplate.from_template(
        "Reformulate the user's follow-up so it stands on its own.\n"
        "Given the conversation and the follow-up, rewrite the follow-up as a "
        "STANDALONE question that carries over all needed context and resolves "
        "references like 'it', 'that', 'those', 'the same'. If it is already "
        "self-contained, return it unchanged. Output ONLY the rewritten "
        "question, nothing else.\n\n"
        "Conversation:\n{hist}\n\nFollow-up: {q}\nStandalone question:"
    )
    try:
        out = (prompt | llm | StrOutputParser()).invoke(
            {"hist": "\n".join(lines), "q": question}
        )
        return (out or "").strip() or question
    except Exception as e:
        log.warning("contextualize_question failed", error=str(e))
        return question


# Strips any literal "[Source: filename]" markers the LLM may echo from the
# context, keeping the answer clean while attribution stays in the prose.
_SOURCE_TAG = re.compile(r"\s*\[Source:\s*([^\]]+?)\]\s*")


def _strip_source_tags(text: str) -> str:
    if not text:
        return text
    # Replace inline "[Source: x]" with just "x"; collapse leftover whitespace.
    cleaned = _SOURCE_TAG.sub(lambda m: " " + m.group(1).strip() + " ", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:)\n])", r"\1", cleaned)
    return cleaned.strip()


# Uploads are saved as "<name>_<6 hex>.<ext>" (a uuid suffix keeps names unique).
# Strip that suffix for display so answers show the original-looking filename
# (rosmerta_753005.xlsx -> rosmerta.xlsx). Display-only; retrieval is unaffected.
def clean_filename(name: str) -> str:
    if not name:
        return name
    return re.sub(r"_[0-9a-f]{6}(\.[^.]+)$", r"\1", name)


# Per-index cache of the loaded FAISS vectorstore + parsed tables, keyed by
# index path and invalidated when the index file is rewritten (re-indexed).
_INDEX_CACHE: dict = {}


def _load_index_cached(model_loader, index_path: str, index_name: str):
    faiss_file = os.path.join(index_path, index_name + ".faiss")
    mtime = os.path.getmtime(faiss_file) if os.path.exists(faiss_file) else 0.0
    cached = _INDEX_CACHE.get(index_path)
    if cached and cached["mtime"] == mtime:
        return cached["vs"], cached["tables"]

    from langchain_community.vectorstores import FAISS as _FAISS
    embeddings = model_loader.load_embeddings()
    vectorstore = _FAISS.load_local(
        index_path, embeddings, index_name=index_name, allow_dangerous_deserialization=True
    )
    # Load any tabular files (csv/xlsx/db) into DataFrames for exact computation.
    tables = {}
    try:
        from src.document_chat.table_qa import load_tables, TABULAR_EXTS
        sources = {
            (d.metadata or {}).get("source")
            for d in vectorstore.docstore._dict.values()
            if (d.metadata or {}).get("source")
        }
        tabular = [s for s in sources if os.path.splitext(s)[1].lower() in TABULAR_EXTS]
        tables = load_tables(tabular) if tabular else {}
    except Exception as e:
        log.warning("table load failed", error=str(e))
    _INDEX_CACHE[index_path] = {"mtime": mtime, "vs": vectorstore, "tables": tables}
    return vectorstore, tables


def _unique_sources(vectorstore) -> List[str]:
    """Return the sorted unique source filenames stored in a FAISS docstore."""
    try:
        docs = vectorstore.docstore._dict.values()
        names = {
            clean_filename(os.path.basename((d.metadata or {}).get("source", "")))
            for d in docs
            if (d.metadata or {}).get("source")
        }
        return sorted(n for n in names if n)
    except Exception:
        return []


# Broad / whole-document questions (summaries, overviews, "all the X") need the
# retriever to cover most of the document, not just the top-k nearest chunks.
_BROAD_HINTS = re.compile(
    r"(?ix)\b("
    r"summar(?:y|ies|ize|ise|ising|izing)|overview|recap|tl;?dr|"
    r"all\s+the|all\s+of|each\b|every\b|list\s+all|"
    r"what\s+are\s+all|across\s+all|whole\s+(?:document|doc|file|report)"
    r")\b"
)


def looks_broad(question: str) -> bool:
    """True for whole-document questions (summaries/overviews/'all the ...')."""
    return bool(question) and bool(_BROAD_HINTS.search(question))


def adaptive_k(vectorstore, base_k: int, question: str, *, broad_cap: int = 60, specific_floor: int = 16) -> dict:
    """Pick retrieval breadth from the question and document size.

    - Broad questions (summaries) retrieve enough to span the whole document, so
      "summarize everything" doesn't miss sections that ranked below a small k.
    - Specific questions on a large document retrieve a bit wider than the small
      default so the right section isn't missed.
    - Small documents and normal queries stay at/under base_k, so latency on the
      common case (short docs / ledgers) is unchanged. Never asks for more chunks
      than the index holds. Returns {"k", "fetch_k"}.
    """
    try:
        total = len(vectorstore.docstore._dict)
    except Exception:
        total = base_k
    if looks_broad(question):
        k = min(total, broad_cap)
    else:
        k = min(total, max(base_k, specific_floor))
    k = max(1, k)
    return {"k": k, "fetch_k": max(20, k * 4)}


class RAGState(TypedDict):
    question: str          # the user's original question — NEVER mutated
    retrieval_query: str   # query used for vector search; rewrite updates only this
    documents: List[Document]
    answer: str
    rewrite_count: int
    relevant_count: int


class AgenticRAG:
    """
    LangGraph-based agentic RAG.

    Flow: retrieve → grade_docs → (rewrite → retrieve)* → generate

    If retrieved docs are irrelevant, the agent rewrites the query and retries
    up to MAX_REWRITES times before falling back to whatever it has.
    """

    MAX_REWRITES = 1

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.retriever = None
        self.vectorstore = None
        self.graph = None
        self._tables = {}
        # Reuse one ModelLoader for both LLM and embeddings (avoids a second
        # config/auth init per request).
        self._ml = ModelLoader()
        self.llm = self._ml.load_llm()
        log.info("AgenticRAG initialized", session_id=self.session_id)

    # ---------- Public API ----------

    def load_retriever_from_faiss(self, index_path: str, k: int = 10, index_name: str = "index"):
        try:
            # The vectorstore + parsed tables are expensive to load (~2s) and
            # identical across requests for the same index — cache them per index,
            # invalidated when the FAISS file is rewritten (re-indexed).
            vectorstore, self._tables = _load_index_cached(self._ml, index_path, index_name)
            # MMR diversifies retrieval so multi-document indexes return chunks
            # from every doc, not just the densest one (key for comparisons).
            # k can vary per request, so the retriever wrapper is built fresh.
            self.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": max(20, k * 4), "lambda_mult": 0.5},
            )
            self.vectorstore = vectorstore
            self._build_graph()
            log.info("AgenticRAG: FAISS retriever loaded", index_path=index_path, k=k, session_id=self.session_id)
        except Exception as e:
            log.error("AgenticRAG: failed to load retriever", error=str(e))
            raise DocumentPortalException("AgenticRAG retriever loading failed", sys)

    def invoke(self, question: str, chat_history=None) -> str:
        try:
            if self.graph is None:
                raise DocumentPortalException(
                    "Graph not initialized. Call load_retriever_from_faiss() first.", sys
                )
            # Resolve follow-ups ("how many days is that?") into a standalone
            # question first, so every downstream path — file listing, table
            # compute, and retrieval — benefits from the conversation context.
            if chat_history:
                question = contextualize_question(self.llm, question, chat_history)
                log.info("AgenticRAG: contextualized question", question=question, session_id=self.session_id)
            # Explicit "list the indexed files" questions are answered straight
            # from the manifest, so document content that itself lists filenames
            # (a codebase doc, a vector DB) can't hijack the answer.
            if _is_file_list_query(question):
                names = _unique_sources(self.vectorstore)
                if names:
                    return f"{len(names)} indexed file(s):\n" + "\n".join(f"- {n}" for n in names)
            # Computational questions over tabular files (totals/counts/max/min)
            # are answered by running real pandas, not RAG. Returns None for
            # descriptive/semantic questions, which fall through to retrieval.
            if self._tables:
                from src.document_chat.table_qa import answer_with_tables, looks_computational
                # Only spend the LLM 'decide' round-trip when the question shows
                # an aggregation/lookup signal; summaries/descriptions skip it.
                if looks_computational(question):
                    table_ans = answer_with_tables(self.llm, question, self._tables)
                    if table_ans is not None:
                        log.info("AgenticRAG: answered via table compute", session_id=self.session_id)
                        return table_ans
            # Adapt retrieval breadth to the question + document size: broad
            # "summarize everything" questions cover the whole doc; specific
            # questions on a large doc widen a little; small docs stay lean.
            if self.retriever is not None and self.vectorstore is not None:
                base_k = self.retriever.search_kwargs.get("k", 10)
                ak = adaptive_k(self.vectorstore, base_k, question)
                self.retriever.search_kwargs["k"] = ak["k"]
                self.retriever.search_kwargs["fetch_k"] = ak["fetch_k"]
                log.info("AgenticRAG: adaptive k", k=ak["k"], broad=looks_broad(question), session_id=self.session_id)
            initial: RAGState = {
                "question": question,
                "retrieval_query": question,
                "documents": [],
                "answer": "",
                "rewrite_count": 0,
                "relevant_count": 0,
            }
            result = self.graph.invoke(initial)
            return result["answer"]
        except Exception as e:
            log.error("AgenticRAG: invocation failed", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("AgenticRAG invocation failed", sys)

    # ---------- Graph nodes ----------

    def _retrieve(self, state: RAGState) -> dict:
        # Search with the (possibly rewritten) retrieval query, not the original
        # question — but the original question is what we ultimately answer.
        docs = self.retriever.invoke(state["retrieval_query"])
        docs = self._ensure_source_coverage(state["retrieval_query"], docs)
        log.info("AgenticRAG: retrieved", count=len(docs), session_id=self.session_id)
        return {"documents": docs}

    def _ensure_source_coverage(self, query: str, docs: list, per_source: int = 2) -> list:
        """Guarantee every indexed file contributes at least a couple of chunks,
        so a large document (e.g. a SQLite dump) can't crowd a small one out of
        retrieval for cross-document questions. Generalises to all file types."""
        try:
            all_sources = {
                (d.metadata or {}).get("source")
                for d in self.vectorstore.docstore._dict.values()
                if (d.metadata or {}).get("source")
            }
        except Exception:
            return docs
        if len(all_sources) <= 1:
            return docs
        present = {(d.metadata or {}).get("source") for d in docs}
        missing = [s for s in all_sources if s not in present]
        if not missing:
            return docs
        try:
            vec = self.vectorstore.embeddings.embed_query(query)
            total = len(self.vectorstore.docstore._dict)
        except Exception:
            return docs
        for src in missing:
            try:
                # fetch_k must span the whole index: FAISS filters AFTER taking
                # the fetch_k nearest, so a small fetch_k would never reach a
                # small file's chunks.
                extra = self.vectorstore.similarity_search_by_vector(
                    vec, k=per_source, filter={"source": src}, fetch_k=max(total, 50)
                )
                docs.extend(extra)
                log.info(
                    "AgenticRAG: source-coverage top-up",
                    source=os.path.basename(src), added=len(extra), session_id=self.session_id,
                )
            except Exception as e:
                log.warning("AgenticRAG: coverage top-up failed", source=src, error=str(e))
        return docs

    def _grade_documents(self, state: RAGState) -> dict:
        # Deterministic gate (no LLM call): rewrite the query only when retrieval
        # returned NOTHING. An LLM relevance grade was pure latency here — it
        # never filtered the context (we keep all chunks), and MMR + per-source
        # coverage already make retrieval reliable, so a grade call per query
        # only slowed things down (and mis-fired rewrites on synthesis questions).
        docs = state["documents"]
        return {"documents": docs, "relevant_count": len(docs)}

    def _rewrite_query(self, state: RAGState) -> dict:
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite this question into a search query that retrieves better "
            "documents from a vector store. Output ONLY the rewritten query, "
            "nothing else.\n"
            "Question: {question}\nSearch query:"
        )
        new_q = (rewrite_prompt | self.llm | StrOutputParser()).invoke({"question": state["question"]})
        log.info(
            "AgenticRAG: rewrote retrieval query",
            original=state["question"],
            rewritten=new_q,
            session_id=self.session_id,
        )
        # Update ONLY the retrieval query; the original question is preserved so
        # _generate always answers what the user actually asked.
        return {"retrieval_query": new_q, "rewrite_count": state["rewrite_count"] + 1}

    def _generation_setup(self, question: str, documents: list):
        """Build the (prompt, inputs) for answer generation."""
        # Prefix each chunk with its source filename so the model can attribute
        # content to the right file (per-file summaries, "which files" questions).
        context = "\n\n".join(
            f"[Source: {clean_filename(os.path.basename((d.metadata or {}).get('source', ''))) or 'unknown'}]\n{d.page_content}"
            for d in documents
        )
        # The file manifest is provided as a *reference* (not the headline) so
        # "which/how many files" questions are answerable regardless of phrasing.
        manifest = _format_manifest(_unique_sources(self.vectorstore))
        gen_prompt = ChatPromptTemplate.from_template(
            "Answer the user's question using the context chunks below. Each "
            "chunk is prefixed with [Source: filename].\n"
            "- For a summary or any content question, synthesize a real answer "
            "from the chunks. Cover EVERY file the question refers to. Never "
            "respond with just the number of files.\n"
            "- Attribute facts by naming the file naturally (e.g. \"according to "
            "resume.pdf\"). Do NOT output the literal \"[Source: ...]\" markup.\n"
            "- Treat the files as INDEPENDENT. Do not assume or invent any "
            "relationship between them, and never attribute one file's content "
            "to another file or its project unless the documents explicitly say "
            "so. Answer using only the file(s) the question is actually about.\n"
            "- If the question asks which, how many, or the names of the files "
            "that exist / are indexed / are uploaded, answer ONLY from this list "
            "of files in this session: {manifest}. Do NOT list filenames that "
            "merely appear inside the document text.\n"
            "- For a summary, overview, or 'what is this' question, ALWAYS "
            "synthesize an answer from the context above — never reply 'I don't "
            "know'. Only reply 'I don't know' when the question asks for a "
            "specific fact that is genuinely absent from the context.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return gen_prompt, {"context": context, "question": question, "manifest": manifest}

    def _generate(self, state: RAGState) -> dict:
        if not state["documents"]:
            return {"answer": "I couldn't find relevant information in the documents to answer your question."}
        prompt, inputs = self._generation_setup(state["question"], state["documents"])
        answer = _strip_source_tags((prompt | self.llm | StrOutputParser()).invoke(inputs))
        log.info("AgenticRAG: generated answer", preview=answer[:100], session_id=self.session_id)
        return {"answer": answer}

    # ---------- Graph routing ----------

    def _decide_next(self, state: RAGState) -> str:
        # Re-retrieve only when NOTHING graded relevant (the retrieval genuinely
        # missed). Otherwise generate using all retrieved context.
        if state.get("relevant_count", 0) == 0 and state["rewrite_count"] < self.MAX_REWRITES:
            return "rewrite"
        return "generate"

    # ---------- Graph construction ----------

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade_docs", self._grade_documents)
        g.add_node("rewrite", self._rewrite_query)
        g.add_node("generate", self._generate)

        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade_docs")
        g.add_conditional_edges(
            "grade_docs",
            self._decide_next,
            {"rewrite": "rewrite", "generate": "generate"},
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", END)

        self.graph = g.compile()
        log.info("AgenticRAG: graph compiled", session_id=self.session_id)