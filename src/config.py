"""Central configuration. Values can be overridden with environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = Path(os.getenv("POLICY_DIR", ROOT / "data" / "policies"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "data" / "chromadb"))
COLLECTION_NAME = "lending_policies"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")

# Retrieval
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
RRF_K = 60  # reciprocal-rank-fusion constant

# Agent
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "6"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8000"))
