# =============================================================================
# RESPONSABLE: Pili
# =============================================================================
# ETAPA 8 (parte espectral): compara la PSD parametrica AR con Lomb
#
# Que hace: para las mismas ventanas de caracteristicas.py (60 latidos validos),
# calcula lf, hf y lf_hf con Lomb (lomb_bandas) y con AR (psd_ar), y muestra:
#   - la correlacion de Spearman entre ambos metodos
#   - las medianas por clase (FA / noFA) con cada metodo
#   - el orden AR elegido por AIC
#
# Entra:  resultados/rr/<reg>.csv
# Sale:   resultados/comparacion_ar_lomb.csv  (una fila por ventana)
#
# Como correrlo (desde la carpeta principal del repo):
#   python codigo/comparar_ar_lomb.py            # todos los registros con RR
#   python codigo/comparar_ar_lomb.py 001 002    # solo esos

import os
import sys

import numpy as np
from scipy.stats import spearmanr

import config
import comun
from caracteristicas import lomb_bandas, psd_ar


def comparar_registro(reg):
    filas = comun.leer_csv(os.path.join(config.RES, "rr", f"{reg}.csv"))
    t = np.array([float(f["t_s"]) for f in filas])
    rr = np.array([float(f["rr_s"]) for f in filas])
    et = np.array([f["etiqueta"] for f in filas])
    ok = np.array([f["valido"] == "1" for f in filas])

    n = config.VENTANA_LATIDOS
    salida = []
    for i in range(0, len(rr) - n + 1, n):      # mismas ventanas que ventanas_registro
        sl = slice(i, i + n)
        etiquetas = set(et[sl])
        if len(etiquetas) != 1 or etiquetas.pop() not in ("FA", "noFA"):
            continue
        if ok[sl].mean() < config.MIN_FRACCION_VALIDA:
            continue
        m = ok[sl]
        lf_l, hf_l, lfhf_l = lomb_bandas(t[sl][m], rr[sl][m])
        lf_a, hf_a, lfhf_a, orden = psd_ar(t[sl][m], rr[sl][m])
        salida.append({"registro": reg, "t_inicio_s": round(float(t[i]), 1),
                       "etiqueta": et[i],
                       "lf_lomb": round(lf_l, 5), "hf_lomb": round(hf_l, 5),
                       "lf_hf_lomb": round(lfhf_l, 5),
                       "lf_ar": round(lf_a, 5), "hf_ar": round(hf_a, 5),
                       "lf_hf_ar": round(lfhf_a, 5), "orden_ar": orden})
    return salida


if __name__ == "__main__":
    regs = sys.argv[1:] or [r for r in comun.registros_a_procesar()
                            if r not in config.REGISTROS_EXCLUIDOS
                            and os.path.exists(os.path.join(config.RES, "rr", f"{r}.csv"))]
    filas = []
    for reg in regs:
        filas += comparar_registro(reg)
        print(f"  {reg}: {len(filas)} ventanas acumuladas", end="\r")

    columnas = list(filas[0].keys())
    comun.guardar_csv(filas, os.path.join(config.RES, "comparacion_ar_lomb.csv"), columnas)

    et = np.array([f["etiqueta"] for f in filas])
    print(f"\n{len(filas)} ventanas. Orden AR elegido: mediana "
          f"{np.median([f['orden_ar'] for f in filas]):.0f}\n")
    print(f"{'':8s}{'Spearman':>10s}   mediana noFA (Lomb / AR)   mediana FA (Lomb / AR)")
    for nom in ("lf", "hf", "lf_hf"):
        a = np.array([f[f"{nom}_lomb"] for f in filas], dtype=float)
        b = np.array([f[f"{nom}_ar"] for f in filas], dtype=float)
        v = np.isfinite(a) & np.isfinite(b)
        rho = spearmanr(a[v], b[v])[0]
        med = lambda x, e: np.nanmedian(x[et == e])
        print(f"{nom:8s}{rho:10.3f}   {med(a, 'noFA'):9.3f} / {med(b, 'noFA'):<9.3f}"
              f"        {med(a, 'FA'):9.3f} / {med(b, 'FA'):<9.3f}")
