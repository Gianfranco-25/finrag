"""Cadena RAG: recupera los trozos relevantes y genera una respuesta que cita sus fuentes con [n].

Si ningún trozo supera el umbral de similitud, responde que no lo sabe sin llamar al modelo:
en un entorno financiero es mejor no contestar que inventar.
"""
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

SIN_RESPUESTA = "No encuentro esa información en la documentación."

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres el asistente interno de una entidad financiera. Respondes preguntas usando SOLO los fragmentos de "
     "documentación que se te dan.\n"
     "Reglas:\n"
     "- Si los fragmentos no contienen la respuesta, responde exactamente: \"" + SIN_RESPUESTA + "\"\n"
     "- No inventes cifras, plazos ni condiciones. Copia los importes tal como aparecen.\n"
     "- Después de cada dato, cita su fragmento con su número entre corchetes, por ejemplo [2].\n"
     "- Responde en español, de forma breve y clara."),
    ("human", "Fragmentos:\n{contexto}\n\nPregunta: {pregunta}"),
])


@dataclass
class Fuente:
    n: int
    fuente: str
    seccion: str
    fragmento: str
    similitud: float


@dataclass
class Respuesta:
    respuesta: str
    fuentes: list[Fuente] = field(default_factory=list)

    @property
    def citadas(self) -> list[Fuente]:
        numeros = {int(n) for n in re.findall(r"\[(\d+)\]", self.respuesta)}
        return [f for f in self.fuentes if f.n in numeros]


def recuperar(store: VectorStore, pregunta: str, k: int, umbral: float) -> list[Fuente]:
    if isinstance(store, InMemoryVectorStore):  # ya devuelve similitud coseno (0..1)
        resultados: list[tuple[Document, float]] = store.similarity_search_with_score(pregunta, k=k)
    else:  # pgvector: convierte la distancia en relevancia (1 = idéntico)
        resultados = store.similarity_search_with_relevance_scores(pregunta, k=k)
    fuentes = []
    for doc, score in resultados:
        if score < umbral:
            continue
        fuentes.append(Fuente(n=len(fuentes) + 1, fuente=doc.metadata.get("fuente", "?"),
                              seccion=doc.metadata.get("seccion", ""), fragmento=doc.page_content,
                              similitud=round(float(score), 3)))
    return fuentes


def formatear_contexto(fuentes: list[Fuente]) -> str:
    return "\n\n".join(f"[{f.n}] ({f.fuente} › {f.seccion})\n{f.fragmento}" for f in fuentes)


def preguntar(pregunta: str, store: VectorStore, llm: BaseChatModel, k: int = 4, umbral: float = 0.35) -> Respuesta:
    fuentes = recuperar(store, pregunta, k, umbral)
    if not fuentes:
        return Respuesta(SIN_RESPUESTA)
    mensaje = (PROMPT | llm).invoke({"contexto": formatear_contexto(fuentes), "pregunta": pregunta})
    return Respuesta(str(mensaje.content).strip(), fuentes)
