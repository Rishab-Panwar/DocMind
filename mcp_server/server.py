"""
DocVault MCP Server

Exposes the document portal's RAG capabilities as MCP tools so that
Claude Desktop, Cursor, and any MCP-compatible client can query your
indexed documents directly.

Tools:
  - ask_document   : agentic RAG query (grade + rewrite loop)
  - simple_query   : single-pass RAG query (faster)
  - list_sessions  : list all indexed sessions

Usage (stdio transport, works with Claude Desktop):
    python mcp_server/server.py

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "docvault": {
          "command": "python",
          "args": ["<absolute-path>/mcp_server/server.py"]
        }
      }
    }
"""

import os
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

from src.document_chat.agent_rag import AgenticRAG
from src.document_chat.retrieval import ConversationalRAG
from utils.llm_cache import init_llm_cache
from logger import GLOBAL_LOGGER as log

init_llm_cache()

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

mcp = FastMCP("DocVault")


@mcp.tool()
def ask_document(question: str, session_id: str) -> str:
    """
    Ask a question about documents indexed in a session using agentic RAG.

    The agent retrieves documents, grades their relevance, rewrites the query
    if needed, and generates a grounded answer.

    Args:
        question:   The question to answer from the documents.
        session_id: Session ID returned by the /chat/index endpoint.

    Returns:
        Answer grounded in the indexed documents.
    """
    index_dir = os.path.join(FAISS_BASE, session_id)
    if not os.path.isdir(index_dir):
        return (
            f"No indexed documents found for session '{session_id}'. "
            "Please index documents first via the /chat/index endpoint."
        )
    try:
        rag = AgenticRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, index_name=FAISS_INDEX_NAME)
        return rag.invoke(question)
    except Exception as e:
        log.error("MCP ask_document failed", error=str(e), session_id=session_id)
        return f"Error querying documents: {e}"


@mcp.tool()
def simple_query(question: str, session_id: str) -> str:
    """
    Single-pass RAG query — faster than ask_document but no relevance grading.

    Args:
        question:   The question to answer.
        session_id: Session ID returned by the /chat/index endpoint.

    Returns:
        Answer from the indexed documents.
    """
    index_dir = os.path.join(FAISS_BASE, session_id)
    if not os.path.isdir(index_dir):
        return f"No indexed documents found for session '{session_id}'."
    try:
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, index_name=FAISS_INDEX_NAME)
        return rag.invoke(question, chat_history=[])
    except Exception as e:
        log.error("MCP simple_query failed", error=str(e), session_id=session_id)
        return f"Error querying documents: {e}"


@mcp.tool()
def list_sessions() -> list:
    """
    List all document sessions that have been indexed and are ready to query.

    Returns:
        List of session ID strings.
    """
    if not os.path.isdir(FAISS_BASE):
        return []
    return [
        d for d in os.listdir(FAISS_BASE)
        if os.path.isdir(os.path.join(FAISS_BASE, d))
    ]


if __name__ == "__main__":
    mcp.run()