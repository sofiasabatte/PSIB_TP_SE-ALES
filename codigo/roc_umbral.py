# =============================================================================
# RESPONSABLE: Juani
# ETAPA 9b: curva ROC, AUC y umbral train/test (TP completo)
#
# TODO (Juani):
#   [ ] Curva ROC y AUC de cada caracteristica (FA = positivo) sobre train.
#   [ ] Elegir el umbral de cada caracteristica en train (p. ej. indice de
#       Youden: max Se + Esp - 1).
#   [ ] Evaluar ese umbral en test: sensibilidad y especificidad.
#   [ ] Opcional: regla combinada de Dash 2009 (nrmssd > 0,1 Y tpr en
#       0,54-0,77 Y shannon > 0,7) evaluada en test.
#   [ ] Guardar resultados/roc_umbral.csv y la figura de las curvas ROC.
#
# La particion es POR SUJETO (de Chazal et al. 2004): todas las ventanas de un
# sujeto quedan del mismo lado. particion_por_sujeto() ya lo hace.
#
# Entra:  resultados/caracteristicas.csv
# Sale:   resultados/roc_umbral.csv, resultados/figuras/roc.png
#
# Como correrlo:
#   python codigo/roc_umbral.py
# =============================================================================

import os

import numpy as np

import config
import comun

FRACCION_TEST = 0.3
SEMILLA = 0      # fija, para que la particion sea reproducible


def particion_por_sujeto(filas, fraccion_test=FRACCION_TEST, semilla=SEMILLA):
    """Separa las ventanas en train y test sin repartir sujetos."""
    sujetos = sorted({f["sujeto"] for f in filas})
    rng = np.random.default_rng(semilla)
    rng.shuffle(sujetos)
    n_test = int(round(fraccion_test * len(sujetos)))
    test = set(sujetos[:n_test])
    train_f = [f for f in filas if f["sujeto"] not in test]
    test_f = [f for f in filas if f["sujeto"] in test]
    return train_f, test_f


if __name__ == "__main__":
    filas = comun.leer_caracteristicas()
    train, test = particion_por_sujeto(filas)
    for nombre, parte in [("train", train), ("test", test)]:
        n_fa = sum(f["etiqueta"] == "FA" for f in parte)
        n_suj = len({f["sujeto"] for f in parte})
        print(f"{nombre}: {n_suj} sujetos, {len(parte)} ventanas ({n_fa} FA)")

    # TODO (Juani): ROC y AUC por caracteristica en train, umbral de Youden,
    # evaluacion en test (Se, Esp). sklearn.metrics.roc_curve sirve, o
    # calcularla a mano recorriendo umbrales.
