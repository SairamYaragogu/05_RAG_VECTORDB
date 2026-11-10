
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# PDF documents directory
POLICY_DIR = BASE_DIR / "data" / "policies"

# Vector database directory
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ollama model
LLM_MODEL = "llama3.2"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB collection
COLLECTION_NAME = "akki_company_policies"

# Retrieval settings
TOP_K = 4

# Text splitting settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
