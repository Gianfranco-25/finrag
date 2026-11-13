# finrag

Asistente RAG que responde preguntas sobre la documentación interna de una entidad financiera **citando las fuentes**,
y un agente que combina búsqueda en la documentación con cálculo de cuotas de préstamos.

> Los documentos de `docs/` son de **Banco Demo**, una entidad ficticia creada para este proyecto.

## Qué hace

- **RAG con fuentes citadas.** Recupera los fragmentos relevantes de la documentación (embeddings + pgvector) y genera
  una respuesta en la que cada dato lleva su cita `[n]`. La web muestra los fragmentos citados.
- **Agente con herramientas** (LangChain `create_agent`): decide cuándo buscar en la documentación y cuándo calcular
  una cuota con el sistema francés.
- **Salvaguardas pensadas para un entorno regulado:**
  - Si ningún fragmento supera el umbral de similitud, responde «No encuentro esa información» **sin llamar al modelo**.
  - La calculadora del agente **rechaza cualquier tipo de interés que no aparezca en la documentación consultada**:
    el modelo no puede inventarse un TIN.
  - Si el agente responde sin citar ninguna fuente, su respuesta se descarta y contesta la cadena RAG.
- **Evaluación automática** con preguntas de referencia: recuperación, respuesta correcta y citada, y rechazo de
  preguntas fuera de tema.
- **Proveedor intercambiable:** Ollama (local y gratis), OpenAI o Azure OpenAI, cambiando solo variables de entorno.

## Arquitectura

```
docs/*.md ──► ingesta ──► trozos por sección (Markdown + tamaño) ──► embeddings ──► Postgres + pgvector
                                                                                         │
navegador ──► FastAPI /api/preguntar ──┬─ modo rag:    recuperar (umbral) ──► LLM con citas
                                       └─ modo agente: LLM + herramientas
                                              ├─ buscar_documentacion  (pgvector)
                                              └─ calcular_cuota_prestamo (solo con TIN documentado)
```

| Pieza | Tecnología |
|---|---|
| Orquestación | LangChain 1.x (`create_agent`, prompts, splitters, `PGVector`) |
| Modelos | Ollama (`qwen3:8b` y `qwen3-embedding:0.6b`), OpenAI o Azure OpenAI |
| Base de datos vectorial | PostgreSQL 17 + pgvector (Docker) |
| API y web | FastAPI + HTML/JS sin dependencias |
| Calidad | pytest (sin red: embeddings y LLM falsos) + evaluación con modelo real |

## Resultados de la evaluación

Con `qwen3:8b` en local (MacBook Pro M1, 16 GB), sobre 15 preguntas de `eval/preguntas.json`:

| Métrica | Resultado |
|---|---|
| Recuperación del documento correcto | 13/13 |
| Respuesta correcta y con cita | 13/13 |
| Rechazo de preguntas fuera de la documentación | 2/2 |

## Cómo ejecutarlo

Requisitos: Python 3.12+, Docker y [Ollama](https://ollama.com).

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:8b && ollama pull qwen3-embedding:0.6b
docker compose up -d --wait               # Postgres + pgvector en el puerto 5433
./.venv/bin/python -m finrag.ingesta      # indexa docs/
./.venv/bin/uvicorn finrag.api:app        # http://localhost:8000
```

Tests y evaluación:

```bash
./.venv/bin/python -m pytest              # rápidos, sin red
./.venv/bin/python -m eval.evaluar        # con el modelo y la base de datos reales
```

Para usar OpenAI o Azure OpenAI, cambia `PROVEEDOR` y los modelos en `.env` (hay ejemplos dentro) y vuelve a indexar:
los embeddings de proveedores distintos no son compatibles entre sí.

Sin Docker: `ALMACEN=memoria` guarda los vectores en memoria y se indexa al arrancar la API.

## API

`POST /api/preguntar`

```json
{ "pregunta": "¿Cuánto cuesta la segunda tarjeta de débito?", "modo": "rag" }
```

```json
{
  "respuesta": "La segunda tarjeta de débito cuesta 20 € al año [1].",
  "fuentes": [{ "n": 1, "fuente": "comisiones_cuentas.md", "seccion": "Tarjeta de débito", "fragmento": "…", "similitud": 0.71 }],
  "citadas": [1]
}
```

Documentación interactiva en `http://localhost:8000/docs`.

## Estructura

```
finrag/config.py       configuración por variables de entorno
finrag/proveedores.py  modelos de chat y embeddings (Ollama / OpenAI / Azure OpenAI)
finrag/ingesta.py      carga, troceado e indexación en pgvector
finrag/rag.py          recuperación con umbral y respuesta con citas
finrag/agente.py       agente con herramientas y salvaguardas
finrag/api.py          API REST y página web
eval/                  preguntas de referencia y script de evaluación
tests/                 tests unitarios y de la API
```

## Siguientes pasos

- Documentos en PDF y actualización incremental (solo reindexar lo que cambia).
- Búsqueda híbrida (palabras clave + vectores) y reordenación de resultados.
- Memoria de conversación para preguntas de seguimiento.
- Despliegue en un VPS con Docker Compose.
