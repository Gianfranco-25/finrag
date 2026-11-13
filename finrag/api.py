"""API REST y página web.

    uvicorn finrag.api:app --reload      abre http://localhost:8000
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from .agente import crear_agente
from .config import RAIZ, Config, config
from .ingesta import almacen, indexar
from .proveedores import modelo_chat
from .rag import preguntar


class Pregunta(BaseModel):
    pregunta: str = Field(min_length=3, max_length=1000)
    modo: Literal["rag", "agente"] = "rag"


class FuenteOut(BaseModel):
    n: int
    fuente: str
    seccion: str
    fragmento: str
    similitud: float


class RespuestaOut(BaseModel):
    respuesta: str
    fuentes: list[FuenteOut]
    citadas: list[int]


def crear_app(cfg: Config | None = None, store: VectorStore | None = None, llm: BaseChatModel | None = None) -> FastAPI:
    cfg = cfg or config()
    estado: dict = {}

    @asynccontextmanager
    async def ciclo(_app: FastAPI):
        estado["store"] = store or almacen(cfg)
        estado["llm"] = llm or modelo_chat(cfg)
        if cfg.almacen == "memoria" and store is None:
            indexar(cfg)  # en memoria no hay nada guardado: se indexa al arrancar
            estado["store"] = almacen(cfg)
        estado["agente"] = crear_agente(estado["store"], estado["llm"], cfg.k, cfg.umbral)
        yield

    app = FastAPI(title="finrag", description="Asistente RAG sobre documentación financiera", lifespan=ciclo)

    @app.get("/", include_in_schema=False)
    def inicio():
        return FileResponse(RAIZ / "static" / "index.html")

    @app.get("/api/salud")
    def salud():
        return {"ok": True, "proveedor": cfg.proveedor, "modelo": cfg.modelo_chat, "almacen": cfg.almacen}

    @app.post("/api/preguntar", response_model=RespuestaOut)
    def api_preguntar(p: Pregunta):
        try:
            if p.modo == "agente":
                r = estado["agente"](p.pregunta)
            else:
                r = preguntar(p.pregunta, estado["store"], estado["llm"], cfg.k, cfg.umbral)
        except Exception as e:  # el modelo o la base de datos no responden
            raise HTTPException(503, f"No se pudo generar la respuesta: {type(e).__name__}") from e
        return RespuestaOut(respuesta=r.respuesta, fuentes=[FuenteOut(**vars(f)) for f in r.fuentes],
                            citadas=[f.n for f in r.citadas])

    return app


app = crear_app()
