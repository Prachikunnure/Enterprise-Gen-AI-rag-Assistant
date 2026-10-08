"""
Application configuration loaded from environment variables and sensible defaults.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env from project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Directories
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"

VECTOR_INDEX_PATH = PROCESSED_DIR / "vector.index"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

# Embedding model (sentence-transformers)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# all-MiniLM-L6-v2 produces 384-dimensional vectors
EMBEDDING_DIMENSION = 384

# Text chunking (word-based for readability; ~500 words ≈ manageable context)
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "500"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))

# Retrieval
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
# Cosine similarity on normalized vectors is in [0, 1]; filter weak matches
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# LLM (OpenAI-compatible API)
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Conversation: number of recent turns passed to the LLM for follow-ups
MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "4"))


def ensure_directories() -> None:
    """Create data directories if they do not exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
