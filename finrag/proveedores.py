"""Crea el modelo de chat y el de embeddings según el proveedor configurado.

Cambiar de Ollama a OpenAI o Azure OpenAI es solo cambiar variables de entorno: el resto del código
trabaja con las interfaces comunes de LangChain (BaseChatModel y Embeddings).
"""
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Config


def modelo_chat(cfg: Config) -> BaseChatModel:
    if cfg.proveedor == "ollama":
        from langchain_ollama import ChatOllama
        # reasoning=False: qwen3 responde directamente, sin el bloque de "pensamiento".
        return ChatOllama(model=cfg.modelo_chat, base_url=cfg.url_ollama, temperature=0, reasoning=False)
    if cfg.proveedor == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=cfg.modelo_chat, temperature=0)
    if cfg.proveedor == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(azure_deployment=cfg.modelo_chat, azure_endpoint=cfg.azure_endpoint,
                               api_version=cfg.azure_version, temperature=0)
    raise ValueError(f"Proveedor desconocido: {cfg.proveedor}")


def modelo_embeddings(cfg: Config) -> Embeddings:
    if cfg.proveedor == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=cfg.modelo_embeddings, base_url=cfg.url_ollama)
    if cfg.proveedor == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=cfg.modelo_embeddings)
    if cfg.proveedor == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(azure_deployment=cfg.modelo_embeddings, azure_endpoint=cfg.azure_endpoint,
                                     api_version=cfg.azure_version)
    raise ValueError(f"Proveedor desconocido: {cfg.proveedor}")
