"""Evalúa el sistema con un conjunto de preguntas de referencia.

    python -m eval.evaluar

Mide tres cosas:
  - recuperación: el documento correcto está entre los fragmentos recuperados
  - respuesta: la respuesta contiene el dato esperado y cita al menos una fuente
  - rechazo: ante preguntas fuera de la documentación, responde que no lo sabe en lugar de inventar
"""
import json
import re
import time
from pathlib import Path

from finrag.config import config
from finrag.ingesta import almacen
from finrag.proveedores import modelo_chat
from finrag.rag import SIN_RESPUESTA, preguntar

CASOS = Path(__file__).with_name("preguntas.json")


def main():
    cfg = config()
    store, llm = almacen(cfg), modelo_chat(cfg)
    casos = json.loads(CASOS.read_text(encoding="utf-8"))
    rec = resp = rech = n_dentro = n_fuera = 0
    inicio = time.time()
    for c in casos:
        r = preguntar(c["pregunta"], store, llm, cfg.k, cfg.umbral)
        if c["fuente"] is None:
            n_fuera += 1
            ok = SIN_RESPUESTA.lower().rstrip(".") in r.respuesta.lower()
            rech += ok
            print(f"{'✓' if ok else '✗'} [fuera] {c['pregunta']}\n    → {r.respuesta[:120]}")
            continue
        n_dentro += 1
        ok_rec = any(f.fuente == c["fuente"] for f in r.fuentes)
        ok_resp = all(re.search(k, r.respuesta, re.I) for k in c["claves"]) and bool(r.citadas)
        rec += ok_rec
        resp += ok_resp
        print(f"{'✓' if ok_rec and ok_resp else '✗'} {c['pregunta']}\n    → {r.respuesta[:160]}")
    print(f"\nRecuperación: {rec}/{n_dentro}   Respuesta correcta y citada: {resp}/{n_dentro}   "
          f"Rechazo fuera de tema: {rech}/{n_fuera}   ({time.time() - inicio:.0f} s, {cfg.proveedor}/{cfg.modelo_chat})")


if __name__ == "__main__":
    main()
