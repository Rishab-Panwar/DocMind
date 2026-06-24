import os
import platform
import sys
import json
from typing import List
import requests as _requests
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from utils.llm_cache import init_llm_cache

_GOOGLE_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
)


class GoogleRestEmbeddings(Embeddings):
    """Calls Google AI embedContent REST endpoint directly.

    gemini-embedding-* models only expose embedContent (sync, one text at a time)
    and asyncBatchEmbedContent. We use embedContent and loop over texts.
    """

    def __init__(self, model: str, api_key: str):
        self._model_id = model.removeprefix("models/")
        self._api_key = api_key
        self._url = _GOOGLE_EMBED_URL.format(model=self._model_id)

    def _embed_one(self, text: str) -> List[float]:
        resp = _requests.post(
            self._url,
            params={"key": self._api_key},
            json={"content": {"parts": [{"text": text}]}},
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Google embedding API error {resp.status_code}: {resp.text}"
            )
        return resp.json()["embedding"]["values"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

# macOS OpenMP duplicate runtime workaround
if platform.system() == "Darwin" and os.environ.get("KMP_DUPLICATE_LIB_OK") is None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Fallback to individual env vars
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # Final check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        use_vertex = os.getenv("USE_VERTEX", "false").lower() in {"1", "true", "yes"}
        if missing and use_vertex:
            # Vertex AI authenticates via the service account (GOOGLE_APPLICATION_
            # CREDENTIALS), not these API keys, so their absence is expected.
            log.info("API keys not set; using Vertex AI service-account auth", missing_keys=missing)
        elif missing:
            # Enforce strictly only in production or when explicitly requested
            strict_mode = os.getenv("ENV", "local").lower() == "production" or os.getenv("STRICT_API_KEYS", "false").lower() in {"1", "true", "yes"}
            if strict_mode:
                log.error("Missing required API keys", missing_keys=missing)
                raise DocumentPortalException("Missing API keys", sys)
            else:
                log.warning("Missing API keys - continuing in non-strict mode", missing_keys=missing)

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})


    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    # Process-wide caches: LLM/embeddings clients are reusable and expensive to
    # build (auth, connection setup), so we build them once per process.
    _LLM_CACHE: dict = {}
    _EMB_CACHE = None
    _VERTEX_INITED = False

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        # Ensure LLM cache is initialized for non-API entry points as well
        init_llm_cache()

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

        # Vertex AI mode: route Google embeddings + LLM through Vertex AI
        # (service-account auth via GOOGLE_APPLICATION_CREDENTIALS) instead of
        # the AI Studio Gemini Developer API key.
        self._use_vertex = os.getenv("USE_VERTEX", "false").lower() in {"1", "true", "yes"}
        self._vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT")
        self._vertex_location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_LOCATION", "us-central1")
        if self._use_vertex:
            # Initialise the Vertex SDK with REST transport by default. The gRPC
            # transport (HTTP/2) is frequently blocked by firewalls/VPNs/AV and
            # fails with "503 ... tcp handshaker shutdown"; REST over plain HTTPS
            # is reliable. Override with VERTEX_TRANSPORT=grpc if your network is
            # fine and you want gRPC's slightly lower latency.
            transport = os.getenv("VERTEX_TRANSPORT", "rest")
            if not ModelLoader._VERTEX_INITED:
                try:
                    import vertexai
                    vertexai.init(
                        project=self._vertex_project,
                        location=self._vertex_location,
                        api_transport=transport,
                    )
                    ModelLoader._VERTEX_INITED = True
                except Exception as e:
                    log.warning("vertexai.init failed", error=str(e))
            log.info(
                "Vertex AI mode enabled",
                project=self._vertex_project,
                location=self._vertex_location,
                transport=transport,
            )

    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI (v1 stable API).

        Cached process-wide: the embeddings client is stateless and reusable, and
        building it (Vertex client init / auth) costs ~2s — paying that on every
        request was a major latency source.
        """
        if ModelLoader._EMB_CACHE is not None:
            return ModelLoader._EMB_CACHE
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name, vertex=self._use_vertex)
            if self._use_vertex:
                from langchain_google_vertexai import VertexAIEmbeddings
                emb = VertexAIEmbeddings(
                    model_name=model_name.removeprefix("models/"),
                    project=self._vertex_project,
                    location=self._vertex_location,
                )
            else:
                emb = GoogleRestEmbeddings(
                    model=model_name,
                    api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                )
            ModelLoader._EMB_CACHE = emb
            return emb
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model.

        Cached process-wide per provider: the chat client is reusable and keeps
        the authenticated connection warm, so calls don't pay cold-start auth on
        every request.
        """
        provider_key = os.getenv("LLM_PROVIDER", "openai")
        cached = ModelLoader._LLM_CACHE.get(provider_key)
        if cached is not None:
            return cached

        llm_block = self.config["llm"]
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)
        # Gemini 2.5 models "think" before answering by default, which adds
        # several seconds per call. Our steps (routing, extraction, grounded
        # answering) don't need it, so default the thinking budget to 0 (off).
        # Override per deployment via config llm.<provider>.thinking_budget.
        thinking_budget = llm_config.get("thinking_budget", 0)

        log.info("Loading LLM", provider=provider, model=model_name, thinking_budget=thinking_budget)

        if provider == "google":
            if self._use_vertex:
                from langchain_google_vertexai import ChatVertexAI
                llm = ChatVertexAI(
                    model=model_name,
                    project=self._vertex_project,
                    location=self._vertex_location,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_budget=thinking_budget,
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_budget=thinking_budget,
                )
        elif provider == "groq":
            llm = ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"),  # type: ignore
                temperature=temperature,
            )
        elif provider == "openai":
            llm = ChatOpenAI(
                model=model_name,
                api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")

        ModelLoader._LLM_CACHE[provider_key] = llm
        return llm


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")
