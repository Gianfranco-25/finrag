"""Ingesta: lee los documentos, los trocea por secciones y guarda sus embeddings en el almacén de vectores.

    python -m finrag.ingesta          (re)indexa la carpeta docs/
"""
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import Config, config
from .proveedores import modelo_embeddings

_memoria: dict[str, InMemoryVectorStore] = {}


def cargar_documentos(carpeta: Path) -> list[Document]:
    docs = []
    for ruta in sorted(carpeta.glob("*.md")):
        docs.append(Document(page_content=ruta.read_text(encoding="utf-8"), metadata={"fuente": ruta.name}))
    if not docs:
        raise FileNotFoundError(f"No hay documentos .md en {carpeta}")
    return docs


def _tiene_contenido(texto: str, minimo_palabras: int = 15) -> bool:
    """Descarta trozos formados solo por títulos (#) y notas (>)."""
    lineas = [l for l in texto.splitlines() if l.strip() and not l.lstrip().startswith(("#", ">"))]
    return len(" ".join(lineas).split()) >= minimo_palabras


def trocear(docs: list[Document], trozo_max: int = 800, solape: int = 100) -> list[Document]:
    """Primero por títulos de Markdown (cada trozo sabe a qué sección pertenece) y después por tamaño."""
    por_titulo = MarkdownHeaderTextSplitter([("#", "titulo"), ("##", "seccion")], strip_headers=False)
    por_tamano = RecursiveCharacterTextSplitter(chunk_size=trozo_max, chunk_overlap=solape)
    trozos = []
    for doc in docs:
        for parte in por_tamano.split_documents(por_titulo.split_text(doc.page_content)):
            if not _tiene_contenido(parte.page_content):
                continue  # solo título o avisos: añadiría ruido a la búsqueda
            parte.metadata["fuente"] = doc.metadata["fuente"]
            parte.metadata.setdefault("seccion", parte.metadata.get("titulo", ""))
            trozos.append(parte)
    for i, t in enumerate(trozos):
        # Id estable: si se vuelve a indexar el mismo texto, se sobrescribe en lugar de duplicarse.
        huella = hashlib.sha256(f"{t.metadata['fuente']}|{i}|{t.page_content}".encode()).hexdigest()[:32]
        t.id = huella
        t.metadata["id"] = huella
    return trozos


def almacen(cfg: Config, embeddings: Embeddings | None = None) -> VectorStore:
    embeddings = embeddings or modelo_embeddings(cfg)
    if cfg.almacen == "memoria":
        if cfg.coleccion not in _memoria:
            _memoria[cfg.coleccion] = InMemoryVectorStore(embeddings)
        return _memoria[cfg.coleccion]
    if cfg.almacen == "pgvector":
        from langchain_postgres import PGVector
        return PGVector(embeddings=embeddings, collection_name=cfg.coleccion, connection=cfg.url_bd, use_jsonb=True)
    raise ValueError(f"Almacén desconocido: {cfg.almacen}")


def indexar(cfg: Config, embeddings: Embeddings | None = None) -> int:
    """Borra la colección y vuelve a indexar todos los documentos. Devuelve el número de trozos."""
    trozos = trocear(cargar_documentos(cfg.carpeta_docs), cfg.trozo_max, cfg.trozo_solape)
    store = almacen(cfg, embeddings)
    if cfg.almacen == "pgvector":
        store.delete_collection()
        store.create_collection()
    elif cfg.coleccion in _memoria:
        _memoria[cfg.coleccion] = store = InMemoryVectorStore(store.embeddings)
    store.add_documents(trozos, ids=[t.id for t in trozos])
    return len(trozos)


if __name__ == "__main__":
    cfg = config()
    n = indexar(cfg)
    print(f"Indexados {n} trozos de {cfg.carpeta_docs} en '{cfg.coleccion}' ({cfg.almacen}).")
