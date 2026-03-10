import os
from pathlib import Path
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_BASE_DIR / ".env", override=True)

class Config : 

    BASE_DIR = _BASE_DIR
    DATA_PATH = os.path.join(BASE_DIR , "data/security-and-technology-policies")
    VECTOR_DB_PATH = os.path.join(BASE_DIR , "db" , "chroma_db")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    LLM_MODEL_NAME = "llama-3.1-8b-instant"
    TEMPERATURE = 0.0 

    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150 

    RETRIEVER_K = 6 

    @staticmethod
    def validate_config():
        """Ensures critical keys are present before starting."""
        if not Config.GROQ_API_KEY:
            raise ValueError("ERROR: GROQ_API_KEY not found in .env file.")
        if not os.path.exists(Config.DATA_PATH):
            os.makedirs(Config.DATA_PATH)
            print(f"Created missing data directory at: {Config.DATA_PATH}")

if __name__ == "__main__":
    Config.validate_config()
    print("Configuration is valid and directories are ready.")