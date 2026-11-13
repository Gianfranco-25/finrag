"""Tests sin red ni modelos: embeddings de bolsa de palabras y un modelo de chat falso."""
import math
import re
import unicodedata
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.vectorstores import InMemoryVectorStore

from finrag.agente import _tin_documentado, cuota_francesa
from finrag.api import crear_app
from finrag.config import Config
from finrag.ingesta import cargar_documentos, trocear
from finrag.rag import SIN_RESPUESTA, Respuesta, Fuente, preguntar, recuperar

DIM = 512


class BolsaDePalabras(Embeddings):
    """Embedding determinista: cuenta palabras en un vector de tamaño fijo. Suficiente para probar la recuperación."""

    def _vec(self, texto: str) -> list[float]:
        plano = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
        v = [0.0] * DIM
        for palabra in re.findall(r"[a-z0-9]{3,}", plano):
            v[hash(palabra) % DIM] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def cfg():
    return replace(Config(), almacen="memoria", umbral=0.1, k=3)


@pytest.fixture(scope="module")
def store(cfg):
    s = InMemoryVectorStore(BolsaDePalabras())
    trozos = trocear(cargar_documentos(cfg.carpeta_docs))
    s.add_documents(trozos, ids=[t.id for t in trozos])
    return s


def test_trocear_conserva_fuente_y_seccion(cfg):
    trozos = trocear(cargar_documentos(cfg.carpeta_docs), trozo_max=400, solape=50)
    assert len(trozos) > 10
    assert len({t.id for t in trozos}) == len(trozos)
    for t in trozos:
        assert t.metadata["fuente"].endswith(".md")
        assert len(t.page_content) <= 400
    secciones = {t.metadata["seccion"] for t in trozos}
    assert "Amortización anticipada" in secciones


def test_recupera_el_documento_correcto(store):
    fuentes = recuperar(store, "¿Cuánto cuesta la segunda tarjeta de débito?", k=3, umbral=0.1)
    assert fuentes[0].fuente == "comisiones_cuentas.md"
    assert fuentes[0].seccion == "Tarjeta de débito"


def test_sin_contexto_relevante_no_llama_al_modelo(store):
    llm = FakeListChatModel(responses=[])  # fallaría si se le llamara
    r = preguntar("¿Cuál es la capital de Francia?", store, llm, k=3, umbral=0.99)
    assert r.respuesta == SIN_RESPUESTA
    assert r.fuentes == []


def test_respuesta_con_citas(store):
    llm = FakeListChatModel(responses=["La segunda tarjeta cuesta 20 € al año [1]."])
    r = preguntar("¿Cuánto cuesta la segunda tarjeta de débito?", store, llm, k=3, umbral=0.1)
    assert "20 €" in r.respuesta
    assert [f.n for f in r.citadas] == [1]


def test_citadas_ignora_numeros_inexistentes():
    f = Fuente(1, "a.md", "s", "texto", 0.9)
    assert Respuesta("Dato [1] y otro [7].", [f]).citadas == [f]


@pytest.mark.parametrize("importe,tin,meses", [(12000, 6.95, 48), (3000, 8.45, 12), (60000, 6.95, 96)])
def test_cuota_francesa_amortiza_el_capital(importe, tin, meses):
    r = cuota_francesa(importe, tin, meses)
    i = tin / 100 / 12
    valor_actual = sum(r["cuota_mensual"] / (1 + i) ** t for t in range(1, meses + 1))
    assert valor_actual == pytest.approx(importe, abs=1)
    assert r["total_pagado"] == pytest.approx(r["cuota_mensual"] * meses, abs=0.01)


def test_cuota_sin_intereses_y_datos_invalidos():
    assert cuota_francesa(1200, 0, 12)["cuota_mensual"] == 100
    with pytest.raises(ValueError):
        cuota_francesa(0, 5, 12)


def test_solo_se_calcula_con_tin_documentado():
    fuentes = [Fuente(1, "p.md", "s", "TIN del 6,95 % con nómina y TIN del 8,45 % sin ella.", 0.9)]
    assert _tin_documentado(6.95, fuentes) and _tin_documentado(8.45, fuentes)
    assert not _tin_documentado(5, fuentes)      # inventado
    assert not _tin_documentado(95, fuentes)     # no confunde 95 con 6,95
    assert not _tin_documentado(6.95, [])        # sin búsqueda previa


def test_api(cfg, store):
    llm = FakeListChatModel(responses=["Son 15 días hábiles [1]."])
    with TestClient(crear_app(cfg, store=store, llm=llm)) as c:
        assert c.get("/api/salud").json()["ok"] is True
        assert c.get("/").status_code == 200
        r = c.post("/api/preguntar", json={"pregunta": "¿Plazo de respuesta para reclamar un cargo no autorizado?"})
        assert r.status_code == 200
        d = r.json()
        assert d["respuesta"] == "Son 15 días hábiles [1]."
        assert d["citadas"] == [1] and d["fuentes"]
        assert c.post("/api/preguntar", json={"pregunta": "?"}).status_code == 422
        assert c.post("/api/preguntar", json={"pregunta": "hola que tal", "modo": "otro"}).status_code == 422
