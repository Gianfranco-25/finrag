"""Agente con herramientas: decide si buscar en la documentación, calcular una cuota o ambas cosas.

Ejemplo: «¿Cuánto pagaría al mes por 12.000 € a 48 meses con nómina domiciliada?»
  1. busca el TIN en la documentación (6,95 % con nómina)
  2. calcula la cuota con el sistema francés
  3. responde citando la fuente
"""
import re

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore

from .rag import Fuente, Respuesta, formatear_contexto, preguntar, recuperar

SISTEMA = (
    "Eres el asistente interno de una entidad financiera. Para cualquier dato sobre productos, comisiones, "
    "plazos o normas usa SIEMPRE primero la herramienta buscar_documentacion; no respondas de memoria ni "
    "supongas tipos de interés. Para calcular una cuota: 1) busca el TIN del producto en la documentación, "
    "2) llama a calcular_cuota_prestamo con ese TIN. Cita cada dato con el número de su fragmento, por "
    "ejemplo [1] o [2]. Si la documentación no tiene la respuesta, dilo. Responde en español, breve."
)


def _tin_documentado(tin: float, fuentes: list[Fuente]) -> bool:
    """El TIN aparece literalmente en algún fragmento consultado (6,95 % o 6.95 %)."""
    texto = " ".join(f.fragmento for f in fuentes)
    variantes = {f"{tin:g}", f"{tin:g}".replace(".", ",")}
    return any(re.search(rf"(?<![\d,.]){re.escape(v)} ?%", texto) for v in variantes)


def cuota_francesa(importe: float, tin_anual: float, meses: int) -> dict:
    """Cuota mensual constante (sistema francés), total de intereses y total pagado."""
    if importe <= 0 or meses <= 0:
        raise ValueError("El importe y el plazo deben ser positivos")
    i = tin_anual / 100 / 12
    cuota = round(importe / meses if i == 0 else importe * i / (1 - (1 + i) ** -meses), 2)
    total = round(cuota * meses, 2)
    return {"cuota_mensual": cuota, "intereses_totales": round(total - importe, 2), "total_pagado": total}


def crear_agente(store: VectorStore, llm: BaseChatModel, k: int = 4, umbral: float = 0.35):
    """Devuelve una función pregunta -> Respuesta. Las fuentes que consulta el agente se acumulan aparte."""

    def responder(pregunta: str) -> Respuesta:
        fuentes: list[Fuente] = []

        @tool
        def buscar_documentacion(consulta: str) -> str:
            """Busca en la documentación interna del banco (comisiones, préstamos, reclamaciones, prevención
            del blanqueo). Devuelve fragmentos numerados [n] para citar."""
            nuevas = recuperar(store, consulta, k, umbral)
            for f in nuevas:  # numeración global entre búsquedas
                f.n = len(fuentes) + 1
                fuentes.append(f)
            return formatear_contexto(nuevas) if nuevas else "Sin resultados relevantes."

        @tool
        def calcular_cuota_prestamo(importe: float, tin_anual: float, meses: int) -> dict | str:
            """Calcula la cuota mensual de un préstamo con el sistema francés. tin_anual en porcentaje
            (por ejemplo 6.95) y obtenido ANTES con buscar_documentacion; meses = plazo en meses."""
            # Salvaguarda: no se calcula con un tipo de interés que no venga de la documentación.
            if not _tin_documentado(tin_anual, fuentes):
                return (f"Error: el TIN {tin_anual:g} % no aparece en la documentación consultada. "
                        "Busca primero el TIN del producto con buscar_documentacion.")
            return cuota_francesa(importe, tin_anual, meses)

        agente = create_agent(llm, tools=[buscar_documentacion, calcular_cuota_prestamo], system_prompt=SISTEMA)
        resultado = agente.invoke({"messages": [{"role": "user", "content": pregunta}]})
        r = Respuesta(str(resultado["messages"][-1].content).strip(), fuentes)
        if not r.citadas:
            # Salvaguarda: una respuesta sin fuentes citadas no se da por buena; responde la cadena RAG,
            # que siempre busca antes y, si no hay nada relevante, dice que no lo sabe.
            return preguntar(pregunta, store, llm, k, umbral)
        return r

    return responder
