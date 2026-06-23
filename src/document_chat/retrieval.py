import sys
import os
import re
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from src.document_chat.agent_rag import (
    _unique_sources, _strip_source_tags, _format_manifest, _is_file_list_query,
    contextualize_question,
)

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors import FlashrankRerank
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id

            # Load LLM and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "mmr",
        search_kwargs: Optional[Dict[str, Any]] = None,
        use_reranker: bool = True,
        reranker_top_n: int = 6,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.

        When use_reranker=True and flashrank is installed, wraps the base FAISS
        retriever with FlashrankRerank so the top-k chunks are re-scored by a
        cross-encoder before being passed to the LLM.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                if search_type == "mmr":
                    # MMR (Maximal Marginal Relevance) diversifies the retrieved
                    # chunks so a multi-document index returns chunks from *all*
                    # docs instead of clustering on the densest one. fetch_k is the
                    # candidate pool MMR re-ranks down to k; lambda_mult=0.5 balances
                    # relevance vs. diversity.
                    search_kwargs = {"k": k, "fetch_k": max(20, k * 4), "lambda_mult": 0.5}
                else:
                    search_kwargs = {"k": k}

            self._base_retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self.vectorstore = vectorstore

            if use_reranker and _RERANKER_AVAILABLE:
                compressor = FlashrankRerank(top_n=reranker_top_n)
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self._base_retriever,
                )
                log.info("Reranker enabled", top_n=reranker_top_n, session_id=self.session_id)
            else:
                self.retriever = self._base_retriever
                if use_reranker and not _RERANKER_AVAILABLE:
                    log.warning("flashrank not installed — reranker disabled", session_id=self.session_id)

            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history=None) -> str:
        """Invoke the LCEL pipeline. `chat_history` is a list of {role, content}
        dicts; follow-ups are resolved to a standalone question first."""
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            # Resolve follow-ups into a standalone question using conversation
            # history before any retrieval or short-circuit runs.
            if chat_history:
                user_input = contextualize_question(self.llm, user_input, chat_history)
            # Explicit "list the indexed files" questions → answer from the
            # manifest, not vector search / document content.
            if _is_file_list_query(user_input) and getattr(self, "vectorstore", None) is not None:
                names = _unique_sources(self.vectorstore)
                if names:
                    return f"{len(names)} indexed file(s):\n" + "\n".join(f"- {n}" for n in names)
            # Question already resolved above, so the chain runs with empty history.
            payload = {"input": user_input, "chat_history": []}
            answer = self.chain.invoke(payload)
            answer = _strip_source_tags(answer)
            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."
            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    def _format_docs(self, docs) -> str:
        # Prefix each chunk with its source filename so the LLM can attribute
        # content to the right document (per-file summaries in a multi-doc index).
        parts = []
        for d in docs:
            content = getattr(d, "page_content", str(d))
            src = (getattr(d, "metadata", {}) or {}).get("source", "")
            name = os.path.basename(src) if src else "unknown"
            parts.append(f"[Source: {name}]\n{content}")
        # Prepend the full file manifest so "which/how many files" questions are
        # answerable from context regardless of phrasing — no intent routing.
        manifest = _format_manifest(_unique_sources(getattr(self, "vectorstore", None)))
        return manifest + "\n\n" + "\n\n".join(parts)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
        

    def get_retrieved_context(self, question: str, k: Optional[int] = None) -> str:
        """
        Retrieve relevant documents for a question and return the context as formatted text.
        
        Args:
            question: The question to retrieve context for
            k: Number of documents to retrieve (optional, uses retriever default if not provided)
            
        Returns:
            str: Formatted text containing the retrieved document content
            
        Raises:
            DocumentPortalException: If retriever is not set or retrieval fails
        """
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set. Call load_retriever_from_faiss first.", sys)
            
            # Temporarily update k on the base retriever (works for both plain and compressed)
            base = getattr(self, "_base_retriever", self.retriever)
            if k is not None:
                original_k = base.search_kwargs.get("k")
                base.search_kwargs["k"] = k

            docs = self.retriever.invoke(question)

            if k is not None and original_k is not None:
                base.search_kwargs["k"] = original_k
            
            # Format and return the context
            context = self._format_docs(docs)
            log.info("Retrieved context", 
                    session_id=self.session_id, 
                    num_docs=len(docs), 
                    context_length=len(context))
            
            return context
            
        except Exception as e:
            log.error("Failed to retrieve context", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to retrieve context", sys)
