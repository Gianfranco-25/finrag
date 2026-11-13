"""Configuración leída de variables de entorno (o del archivo .env)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")


@dataclass(frozen=True)
class Config:
    # Proveedor de modelos: "ollama" (local y gratis), "openai" o "azure".
    proveedor: str = os.getenv("PROVEEDOR", "ollama")
    modelo_chat: str = os.getenv("MODELO_CHAT", "qwen3:8b")
    modelo_embeddings: str = os.getenv("MODELO_EMBEDDINGS", "qwen3-embedding:0.6b")
    url_ollama: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

    # Azure OpenAI: los modelos se piden por el nombre del despliegue.
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    # Almacén de vectores: "pgvector" (Postgres) o "memoria" (para probar sin base de datos).
    almacen: str = os.getenv("ALMACEN", "pgvector")
    url_bd: str = os.getenv("DATABASE_URL", "postgresql+psycopg://finrag:finrag@localhost:5433/finrag")
    coleccion: str = os.getenv("COLECCION", "documentacion")

    carpeta_docs: Path = Path(os.getenv("CARPETA_DOCS", str(RAIZ / "docs")))
    trozo_max: int = int(os.getenv("TROZO_MAX", "800"))
    trozo_solape: int = int(os.getenv("TROZO_SOLAPE", "100"))
    k: int = int(os.getenv("K", "4"))
    # Por debajo de esta similitud un trozo no se considera relevante.
    umbral: float = float(os.getenv("UMBRAL", "0.35"))


def config() -> Config:
    return Config()
