# =============================================================================
# RESPONSABLE: Sofi
# ETAPA 9a: comparacion FA vs no-FA por caracteristica (TP completo)
#
# TODO (Sofi):
#   [ ] Estadistica descriptiva por clase (mediana, IQR) de cada caracteristica
#       usando los valores AGREGADOS POR SUJETO (ya lo hace por_sujeto()).
#   [ ] Test de hipotesis por caracteristica sobre los valores por sujeto
#       (p. ej. Wilcoxon de rangos con signo si cada sujeto tiene FA y no-FA,
#       o Mann-Whitney si no) y correccion por comparaciones multiples (Holm).
#   [ ] Guardar la tabla en resultados/estadistica.csv.
#
# Por que por sujeto: las ventanas de un mismo sujeto no son independientes;
# usar todas las ventanas infla artificialmente la significancia.
#
# Entra:  resultados/caracteristicas.csv
# Sale:   resultados/estadistica.csv
#
# Como correrlo:
#   python codigo/estadistica.py
# =============================================================================

import os
from collections import defaultdict

import numpy as np

import config
import comun
from caracteristicas import NOMBRES


def por_sujeto(filas):
    """Mediana de cada caracteristica por (sujeto, etiqueta).
    Devuelve {(sujeto, etiqueta): {caracteristica: mediana}}."""
    grupos = defaultdict(list)
    for f in filas:
        grupos[(f["sujeto"], f["etiqueta"])].append(f)
    salida = {}
    for clave, fs in grupos.items():
        salida[clave] = {n: float(np.nanmedian([float(f[n]) for f in fs]))
                         for n in NOMBRES}
    return salida


if __name__ == "__main__":
    filas = comun.leer_csv(os.path.join(config.RES, "caracteristicas.csv"))
    agregado = por_sujeto(filas)
    sujetos_fa = {s for (s, e) in agregado if e == "FA"}
    sujetos_nofa = {s for (s, e) in agregado if e == "noFA"}
    print(f"{len(filas)} ventanas -> {len(agregado)} pares (sujeto, ritmo)")
    print(f"sujetos con FA: {len(sujetos_fa)}, con noFA: {len(sujetos_nofa)}, "
          f"con ambos: {len(sujetos_fa & sujetos_nofa)}")

    # TODO (Sofi): descriptiva, tests y correccion de Holm a partir de 'agregado'
