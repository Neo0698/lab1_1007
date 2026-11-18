"""Choix du modèle : effet de k et de la taille du jeu de données sur l'exactitude du kNN.

Lancement (depuis la racine du projet) : python -m src.choix_modele
"""

from __future__ import annotations

import os
import time

import pandas as pd
from matplotlib.figure import Figure

from .model import Data, ModeleVol, PoidsVariables

CHEMIN_CSV = os.path.join("donnees", "Airlines.csv")
TAILLES = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
VALEURS_K = range(20, 150, 5)


def score_selon_taille(df: pd.DataFrame, tailles: list[int], k: int = 31) -> pd.DataFrame:
    """Entraîne un kNN sur des échantillons de tailles croissantes et mesure ses scores.

    Chaque échantillon est tiré au hasard (graine fixe) puis découpé 80 % / 20 %
    comme dans ModeleVol.entrainer().

    Args:
        df: Jeu de données complet.
        tailles: Nombres de lignes à tester (les tailles plus grandes que df sont ignorées).
        k: Nombre de voisins du kNN.

    Returns:
        DataFrame avec les colonnes taille, exactitude_knn, exactitude_reference, temps_s.
    """
    tailles = sorted({t for t in tailles if t < len(df)} | {len(df)})
    resultats = []
    for taille in tailles:
        modele = ModeleVol(n=k)
        modele.donnees = Data(df.sample(n=taille, random_state=42))
        debut = time.perf_counter()
        scores = modele.entrainer()
        duree = time.perf_counter() - debut
        resultats.append({"taille": taille, **scores, "temps_s": duree})
        print(f"{taille:>9} lignes : kNN = {scores['exactitude_knn']:.3f}  ({duree:.1f} s)")
    return pd.DataFrame(resultats)


def graphique_taille(resultats: pd.DataFrame, chemin_sortie: str) -> str:
    """Enregistre le graphique exactitude et temps d'entraînement selon la taille (PNG)."""
    fig = Figure(figsize=(12, 5))
    ax_score, ax_temps = fig.subplots(1, 2)

    ax_score.plot(resultats["taille"], resultats["exactitude_knn"] * 100, marker="o",
                  color="#1565c0", linewidth=2, label="kNN (k=31)")
    ax_score.plot(resultats["taille"], resultats["exactitude_reference"] * 100, marker="s",
                  color="#9e9e9e", linestyle="--", label="Référence (classe majoritaire)")
    ax_score.set_xscale("log")
    ax_score.set(title="Exactitude selon la taille du jeu de données",
                 xlabel="Nombre de vols (échelle log)", ylabel="Exactitude sur le test (%)")
    ax_score.grid(True, alpha=0.3)
    ax_score.legend()

    ax_temps.plot(resultats["taille"], resultats["temps_s"], marker="o", color="#c62828",
                  linewidth=2)
    ax_temps.set_xscale("log")
    ax_temps.set(title="Temps d'entraînement + évaluation",
                 xlabel="Nombre de vols (échelle log)", ylabel="Temps (s)")
    ax_temps.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(chemin_sortie) or ".", exist_ok=True)
    fig.savefig(chemin_sortie, dpi=120)
    return chemin_sortie


def score_selon_poids(df: pd.DataFrame, combinaisons: dict[str, PoidsVariables],
                      k: int = 31) -> pd.DataFrame:
    """Entraîne un kNN pour chaque combinaison de poids et compare les exactitudes.

    Args:
        df: Jeu de données (un échantillon suffit pour comparer).
        combinaisons: Nom lisible -> poids des variables.
        k: Nombre de voisins du kNN.

    Returns:
        DataFrame trié par exactitude décroissante (colonnes nom, poids, exactitude_knn).
    """
    resultats = []
    for nom, poids in combinaisons.items():
        modele = ModeleVol(n=k, poids=poids)
        modele.donnees = Data(df)
        score = modele.entrainer()["exactitude_knn"]
        resultats.append({"nom": nom, "poids": poids.en_dict(), "exactitude_knn": score})
        print(f"{nom:<40} {score:.4f}")
    return pd.DataFrame(resultats).sort_values("exactitude_knn", ascending=False)


def score_selon_k(modele: ModeleVol, valeurs_k, chemin_sortie: str) -> str:
    """Compare l'exactitude du kNN selon k, avec et sans pondération sigmoïde (PNG).

    À appeler après modele.entrainer(). Long sur le jeu complet.
    """
    sans_poids = [modele.test_knn(n=k, weight=False) for k in valeurs_k]
    avec_poids = [modele.test_knn(n=k, weight=True) for k in valeurs_k]

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    ax.plot(list(valeurs_k), sans_poids, marker="o", color="#2e7d32", linewidth=2,
            label="Sans pondération")
    ax.plot(list(valeurs_k), avec_poids, marker="o", color="#8F00FF", linewidth=2,
            label="Pondération sigmoïde")
    ax.set(title="Score du modèle k-NN en fonction de k", xlabel="Valeur de k",
           ylabel="Score (exactitude)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin_sortie, dpi=120)
    return chemin_sortie


def main() -> None:
    """Trace l'exactitude du kNN selon la taille du jeu de données."""
    try:
        df = Data().charger_donnees(CHEMIN_CSV)
    except FileNotFoundError as e:
        print(f"{e}\nGénérez d'abord les données avec : python -m src.create_dataset")
        return
    resultats = score_selon_taille(df, TAILLES)
    resultats.to_csv(os.path.join("donnees", "score_vs_taille.csv"), index=False)
    chemin = graphique_taille(resultats, os.path.join("donnees", "score_vs_taille.png"))
    print(f"--- Graphique enregistré dans '{chemin}' ---")


if __name__ == "__main__":
    main()
