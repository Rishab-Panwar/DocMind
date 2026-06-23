import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "llama-3.3-70b-versatile"

from logger import GLOBAL_LOGGER as log

from src.document_ingestion.data_ingestion import ChatIngestor
from src.document_chat.retrieval import ConversationalRAG


DEEPEVAL_INPUT_DIR = os.getenv("DEEPEVAL_INPUT_DIR", "data_deep_eval")
# Defaults align with the notebook usage
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "notebook/eval_data")
FAISS_BASE = os.getenv("FAISS_BASE", "notebook/eval_faiss_index")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")
DATASET_ALIAS = os.getenv("DEEPEVAL_DATASET_ALIAS", "test_doc_chat")


class LocalFileAdapter:
    def __init__(self, file_path: str):
        self.name = os.path.basename(file_path)
        self._file_path = file_path

    def read(self) -> bytes:
        with open(self._file_path, "rb") as f:
            return f.read()

    # For compatibility with save_uploaded_files
    def getbuffer(self) -> bytes:
        return self.read()


def list_supported_files(root: Path) -> List[Path]:
    exts = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"}
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def query_rag(question: str, session_id: str, k: int = 5) -> dict:
    index_dir = os.path.join(FAISS_BASE, session_id)
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"FAISS index not found at: {index_dir}")

    rag = ConversationalRAG(session_id=session_id)
    rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)
    answer = rag.invoke(question, chat_history=[])
    context = rag.get_retrieved_context(question, k=k)
    return {"answer": answer, "context": context}


def main():
    # Load env locally like in the notebook / ModelLoader
    if os.getenv("ENV", "local").lower() != "production":
        load_dotenv()
        log.info("Running in LOCAL mode: .env loaded")

    # 1) Build or load FAISS index from the specified directory
    data_dir = Path(DEEPEVAL_INPUT_DIR)
    assert data_dir.exists(), f"Input dir not found: {data_dir}"
    paths = list_supported_files(data_dir)
    if not paths:
        log.error("No supported files found in input directory", dir=str(data_dir))
        print("No supported files found in input directory.")
        sys.exit(1)

    # Ingest and index
    chat_ingestor = ChatIngestor(temp_base=UPLOAD_BASE, faiss_base=FAISS_BASE)
    adapters = [LocalFileAdapter(str(p)) for p in paths]
    chat_ingestor.built_retriver(adapters)
    log.info("Ingestion complete", session_id=chat_ingestor.session_id)

    # 2) Pull dataset from Confident AI (with fallback)
    dataset = EvaluationDataset()
    try:
        dataset.pull(alias=DATASET_ALIAS)
    except Exception as e:
        log.warning("Failed to pull dataset; falling back to local goldens", error=str(e))

    goldens = getattr(dataset, "goldens", []) or []
    if not goldens:
        class _G:
            def __init__(self, q: str, exp: str = ""):
                self.input = q
                self.expected_output = exp
        goldens = [
            _G("What is this document about?"),
        ]

    # 3) For each golden, query RAG and build test cases
    for golden in goldens[:1]:
        try:
            result = query_rag(golden.input, session_id=chat_ingestor.session_id)
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=result["answer"],
                expected_output=golden.expected_output,
                retrieval_context=[result["context"]],
                context=[result["context"]],
            )
            dataset.add_test_case(test_case)
        except Exception as e:
            log.error("Failed to build test case", error=str(e), question=golden.input)

    # 4) Evaluate with a subset of metrics to stay within Groq free-tier TPM
    judge = GroqJudge()
    metrics = [
        AnswerRelevancyMetric(model=judge),
        FaithfulnessMetric(model=judge),
        ContextualRelevancyMetric(model=judge),
    ]

    try:
        evaluate(test_cases=dataset.test_cases, metrics=metrics)
    except Exception as e:
        if "Invalid API key" in str(e) or "ConfidentApiError" in type(e).__name__:
            log.warning("DeepEval results upload skipped (no Confident AI key) — evaluation complete", error=str(e))
        else:
            raise


if __name__ == "__main__":
    main()


