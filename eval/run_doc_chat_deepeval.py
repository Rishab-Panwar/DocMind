"""Simple answer-quality evaluation with DeepEval.

Self-contained on purpose: it runs a few inline question/answer/context cases
through DeepEval's LLM-judged metrics, using the SAME Vertex Gemini the app uses
as the judge (via the GCP service-account credentials). No external dataset and
no separate Groq/OpenAI key required — it demonstrates the evaluation pipeline
and scores grounded answers. Runs in CI when the GCP credentials are present.
"""
import os
import sys
import json

# DeepEval prints a rich results table with unicode glyphs; force UTF-8 so it
# doesn't crash on a non-UTF-8 console (e.g. Windows cp1252). No-op on Linux/CI.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_vertexai import ChatVertexAI


def _project() -> str | None:
    """Resolve the GCP project: prefer env, else read it from the SA key file."""
    proj = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT")
    if proj:
        return proj
    sa = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa and os.path.exists(sa):
        try:
            return json.load(open(sa)).get("project_id")
        except Exception:
            return None
    return None


class VertexJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by Vertex Gemini (same provider the app uses)."""

    def __init__(self):
        self.model = ChatVertexAI(
            model=os.getenv("EVAL_JUDGE_MODEL", "gemini-2.5-flash"),
            project=_project(),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            temperature=0,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "vertex-gemini"


# A few self-contained cases: the question, the answer a grounded RAG system
# should give, and the retrieved context it was grounded in.
CASES = [
    LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        retrieval_context=["France is a country in Western Europe. Its capital and largest city is Paris."],
    ),
    LLMTestCase(
        input="What is the closing balance and is it a debit or credit balance?",
        actual_output="The closing balance is 1,435,756, and it is a credit balance.",
        retrieval_context=["Ledger totals: total debit 4,749,500; total credit 3,313,744; closing balance 1,435,756 (credit)."],
    ),
    LLMTestCase(
        input="Who prepared the report and when?",
        actual_output="The report was prepared by the Analytics team in March 2026.",
        retrieval_context=["This quarterly performance report was prepared by the Analytics team in March 2026."],
    ),
]


def main():
    if os.getenv("ENV", "local").lower() != "production":
        load_dotenv()

    judge = VertexJudge()
    metrics = [
        AnswerRelevancyMetric(model=judge, threshold=0.5),
        FaithfulnessMetric(model=judge, threshold=0.5),
    ]

    try:
        evaluate(test_cases=CASES, metrics=metrics)
    except Exception as e:
        # Confident AI upload is optional; don't fail the run without that key.
        if "Invalid API key" in str(e) or "ConfidentApiError" in type(e).__name__:
            print("DeepEval results upload skipped (no Confident AI key) — evaluation complete.")
        else:
            raise
    print("DeepEval (Vertex Gemini judge) completed.")


if __name__ == "__main__":
    main()
