import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)


class Settings:

    # Application
    APP_NAME: str = os.getenv("APP_NAME", "AXA Sinistre Agent POC")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))

    # OCR Configuration
    TESSERACT_PATH: Optional[str] = os.getenv("TESSERACT_PATH")
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "fra,eng")

    # Streamlit Configuration
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", str(LOGS_DIR / "app.log"))

    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    INPUT_DIR: Path = INPUT_DIR
    OUTPUT_DIR: Path = OUTPUT_DIR
    LOGS_DIR: Path = LOGS_DIR

    def validate(self) -> bool:
        """Validation de la configuration"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY doit être définie dans le fichier .env")
        return True


settings = Settings()

if __name__ != "__main__":
    settings.validate()