# =============================================================================
# RESPONSABLE: Sofi
# ETAPA 9a: comparacion FA vs no-FA por caracteristica
#
# Que hace:
#   1. Resume cada caracteristica por sujeto: mediana de sus ventanas en FA y
#      mediana de sus ventanas en no-FA (por_sujeto).
#   2. Estadistica descriptiva por clase, sobre esos valores por sujeto:
#      mediana y rango intercuartil (Q1-Q3).
#   3. Test principal: Wilcoxon de rangos con signo, PAREADO, sobre los sujetos
#      que tienen FA y no-FA (cada sujeto es su propio control). Tamano del
#      efecto: correlacion biserial por rangos pareada (de -1 a 1).
#   4. Test de sensibilidad: Mann-Whitney con TODOS los sujetos (grupos
#      independientes), con el AUC = U / (n1 * n0).
#   5. Correccion de Holm por las comparaciones multiples (una por
#      caracteristica) para cada test.
#
# Por que por sujeto: las ventanas de un mismo sujeto no son independientes;
# usar todas las ventanas infla artificialmente la significancia.
# Por que tests no parametricos: las caracteristicas no son normales y hay
# valores atipicos.
#
# Entra:  resultados/caracteristicas.csv
# Sale:   resultados/estadistica.csv
#
# Como correrlo:
#   python codigo/estadistica.py
# =============================================================================

import os
import warnings
from collections import defaultdict

import numpy as np
from scipy import stats

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
        salida[clave] = {}
        for n in NOMBRES:
            valores = [float(f[n]) for f in fs if f[n] not in ("", "nan", "NaN")]
            salida[clave][n] = float(np.median(valores)) if valores else np.nan
    return salida


def holm(p):
    """Correccion de Holm. p: lista de p-valores; devuelve los ajustados."""
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p)
    m = len(p)
    ajustado = np.empty(m)
    maximo = 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, min(1.0, (m - rango) * p[i]))
        ajustado[i] = maximo
    return ajustado


def biserial_pareada(d):
    """Correlacion biserial por rangos para diferencias pareadas d (-1 a 1)."""
    d = d[d != 0]
    rangos = stats.rankdata(np.abs(d))
    mas, menos = rangos[d > 0].sum(), rangos[d < 0].sum()
    return (mas - menos) / (mas + menos)


def analizar(agregado):
    """Una fila por caracteristica con descriptiva y tests (sin Holm todavia)."""
    sujetos = {s for (s, _) in agregado}
    filas = []
    for n in NOMBRES:
        fa = {s: agregado[(s, "FA")][n] for s in sujetos if (s, "FA") in agregado}
        nofa = {s: agregado[(s, "noFA")][n] for s in sujetos if (s, "noFA") in agregado}
        fa = {s: v for s, v in fa.items() if not np.isnan(v)}
        nofa = {s: v for s, v in nofa.items() if not np.isnan(v)}

        fila = {"caracteristica": n}
        for nombre, d in (("FA", fa), ("noFA", nofa)):
            v = np.array(list(d.values()))
            fila[f"n_sujetos_{nombre}"] = len(v)
            fila[f"mediana_{nombre}"] = float(np.median(v))
            fila[f"q1_{nombre}"] = float(np.percentile(v, 25))
            fila[f"q3_{nombre}"] = float(np.percentile(v, 75))

        # Test principal: Wilcoxon pareado (sujetos con FA y no-FA)
        ambos = sorted(set(fa) & set(nofa))
        dif = np.array([fa[s] - nofa[s] for s in ambos])
        fila["n_pares"] = len(ambos)
        fila["mediana_dif_FA_menos_noFA"] = float(np.median(dif))
        fila["wilcoxon_p"] = float(stats.wilcoxon(dif).pvalue)
        fila["efecto_biserial"] = float(biserial_pareada(dif))

        # Sensibilidad: Mann-Whitney con todos los sujetos
        a, b = np.array(list(fa.values())), np.array(list(nofa.values()))
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        fila["mannwhitney_p"] = float(u.pvalue)
        fila["auc_sujetos"] = float(u.statistic / (len(a) * len(b)))
        filas.append(fila)

    for clave in ("wilcoxon_p", "mannwhitney_p"):
        ajustados = holm([f[clave] for f in filas])
        for f, aj in zip(filas, ajustados):
            f[clave + "_holm"] = float(aj)
    return filas


if __name__ == "__main__":
    filas = comun.leer_csv(os.path.join(config.RES, "caracteristicas.csv"))
    agregado = por_sujeto(filas)
    sujetos_fa = {s for (s, e) in agregado if e == "FA"}
    sujetos_nofa = {s for (s, e) in agregado if e == "noFA"}
    print(f"{len(filas)} ventanas -> {len(agregado)} pares (sujeto, ritmo)")
    print(f"sujetos con FA: {len(sujetos_fa)}, con noFA: {len(sujetos_nofa)}, "
          f"con ambos: {len(sujetos_fa & sujetos_nofa)}")

    tabla = analizar(agregado)
    columnas = list(tabla[0].keys())
    comun.guardar_csv(tabla, os.path.join(config.RES, "estadistica.csv"), columnas)

    print("\ncaracteristica   med_FA   med_noFA  pares  wilcoxon_holm  efecto  AUC_suj")
    for f in tabla:
        print(f"{f['caracteristica']:<14} {f['mediana_FA']:8.3f} {f['mediana_noFA']:9.3f} "
              f"{f['n_pares']:6d} {f['wilcoxon_p_holm']:13.2e} "
              f"{f['efecto_biserial']:7.2f} {f['auc_sujetos']:8.3f}")
    print("\nguardado en resultados/estadistica.csv")
