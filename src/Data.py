from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

from .dataset import Dataset

# Colonnes utilisées comme caractéristiques (identique à src/model.py)
FEATURES = ["Airline_Encoded", "Origin_Taux", "DayOfWeek", "Time", "Length", "Length_Carre"]


class Data(Dataset):
    """Jeu de données de vols : chargement, découpage entraînement/test et exploration.

    Args:
        data: DataFrame déjà chargé (facultatif).
        airlines: Liste des compagnies connues (facultatif).
    """

    def __init__(self, data: pd.DataFrame | None = None, airlines: list[str] | None = None) -> None:
        self._data = data
        self._liste_airlines: list[str] = list(airlines) if airlines else []
        self._liste_aeroports: list[str] = []
        self._features = list(FEATURES)
        self._df_train: pd.DataFrame | None = None
        self._df_test: pd.DataFrame | None = None

    # --- Accès en lecture --------------------------------------------------
    @property
    def data(self) -> pd.DataFrame | None:
        """Données brutes chargées depuis le CSV."""
        return self._data

    @property
    def liste_airlines(self) -> list[str]:
        """Compagnies connues (triées)."""
        return self._liste_airlines

    @property
    def liste_aeroports(self) -> list[str]:
        """Aéroports de départ connus (triés)."""
        return self._liste_aeroports

    @property
    def features(self) -> list[str]:
        """Colonnes utilisées comme caractéristiques par le modèle."""
        return self._features

    @features.setter
    def features(self, colonnes: list[str]) -> None:
        self._features = list(colonnes)

    def _verifier_decoupage(self) -> None:
        """Lève RuntimeError si train_test_split() n'a pas encore été appelé."""
        if self._df_train is None or self._df_test is None:
            raise RuntimeError("Données non découpées : appeler train_test_split() d'abord.")

    @property
    def X_train(self) -> pd.DataFrame:
        """Caractéristiques de l'ensemble d'entraînement."""
        self._verifier_decoupage()
        return self._df_train[self._features]

    @property
    def y_train(self) -> pd.Series:
        """Cible (Delay) de l'ensemble d'entraînement."""
        self._verifier_decoupage()
        return self._df_train["Delay"]

    @property
    def X_test(self) -> pd.DataFrame:
        """Caractéristiques de l'ensemble de test."""
        self._verifier_decoupage()
        return self._df_test[self._features]

    @property
    def y_test(self) -> pd.Series:
        """Cible (Delay) de l'ensemble de test."""
        self._verifier_decoupage()
        return self._df_test["Delay"]

    @property
    def df_train(self) -> pd.DataFrame:
        """Ensemble d'entraînement complet (toutes les colonnes)."""
        self._verifier_decoupage()
        return self._df_train

    # --- Modification de l'état --------------------------------------------
    def charger_donnees(self, path: str) -> pd.DataFrame:
        """Lit le CSV et le garde en mémoire.

        erreur:
            FileNotFoundError: si le fichier n'existe pas.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Erreur : le fichier '{path}' est introuvable.")
        self._data = pd.read_csv(path)
        return self._data

    def charger_airlines(self, airlines: list[str]) -> None:
        """Remplace la liste des compagnies connues."""
        self._liste_airlines = list(airlines)

    def charger_aeroports(self, aeroports: list[str]) -> None:
        """Remplace la liste des aéroports de départ connus."""
        self._liste_aeroports = list(aeroports)

    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2,
                         graine: int = 42) -> None:
        """Découpe df en ensembles d'entraînement et de test (graine fixée : reproductible)."""
        self._df_train, self._df_test = train_test_split(df, test_size=test_size,
                                                         random_state=graine)

    def transformer(self, fonction: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """Applique fonction (DataFrame -> DataFrame) aux ensembles d'entraînement et de test."""
        self._verifier_decoupage()
        self._df_train = fonction(self._df_train)
        self._df_test = fonction(self._df_test)

    
    def resume_qualite(self) -> pd.DataFrame:
        """Résume la qualité des données, une ligne par colonne.

        retourne:
            DataFrame avec le type, le nombre de valeurs manquantes, le min et le max.

        erreur:
            RuntimeError: si aucune donnée n'est chargée.
        """
        if self._data is None:
            raise RuntimeError("Aucune donnée chargée : appeler charger_donnees() d'abord.")
        return pd.DataFrame({
            "type": self._data.dtypes.astype(str),
            "manquantes": self._data.isna().sum(),
            "min": self._data.min(numeric_only=True),
            "max": self._data.max(numeric_only=True),
        })

    def graphique_exploration(self, chemin_sortie: str) -> str:
        """Enregistre un graphique d'exploration du taux de retard (PNG).

        Quatre vues : par heure de départ, par jour de la semaine, par compagnie
        et distribution des durées de vol selon le retard.

        Args:
            chemin_sortie: Chemin du fichier image à créer.

        retourne:
            Le chemin du fichier enregistré.

        erreur:
            RuntimeError: si aucune donnée n'est chargée.
        """
        if self._data is None:
            raise RuntimeError("Aucune donnée chargée : appeler charger_donnees() d'abord.")
        df = self._data.dropna()
        par_heure = df.groupby(df["Time"] // 60)["Delay"].mean() * 100
        par_jour = df.groupby("DayOfWeek")["Delay"].mean() * 100
        par_compagnie = df.groupby("Airline")["Delay"].mean() * 100
        compagnies = sorted(par_compagnie.index, key=lambda c: par_compagnie[c])

        fig = Figure(figsize=(13, 9))
        (ax_heure, ax_jour), (ax_cie, ax_duree) = fig.subplots(2, 2)

        ax_heure.bar(par_heure.index, par_heure.values, color="#c62828")
        ax_heure.set(title="Taux de retard selon l'heure de départ",
                     xlabel="Heure de départ (h)", ylabel="Vols en retard (%)")

        ax_jour.bar(par_jour.index, par_jour.values, color="#ef6c00")
        ax_jour.set_xticks(range(1, 8), ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"])
        ax_jour.set(title="Taux de retard selon le jour", xlabel="Jour de la semaine",
                    ylabel="Vols en retard (%)")

        ax_cie.barh(compagnies, [par_compagnie[c] for c in compagnies], color="#1565c0")
        ax_cie.set(title="Taux de retard par compagnie", xlabel="Vols en retard (%)",
                   ylabel="Compagnie")

        for delay, couleur, nom in ((0, "#2e7d32", "À l'heure"), (1, "#c62828", "En retard")):
            ax_duree.hist(df.loc[df["Delay"] == delay, "Length"], bins=40, alpha=0.6,
                          color=couleur, label=nom)
        ax_duree.set(title="Durée des vols selon le retard", xlabel="Durée du vol (min)",
                     ylabel="Nombre de vols")
        ax_duree.legend()

        fig.tight_layout()
        dossier = os.path.dirname(chemin_sortie)
        if dossier:
            os.makedirs(dossier, exist_ok=True)
        fig.savefig(chemin_sortie, dpi=120)
        return chemin_sortie

    # --- Méthodes spéciales --------------------------------------------------
    def __add__(self, autre: "Data") -> "Data":
        """Renvoie un nouveau Data qui met bout à bout les lignes des deux jeux."""
        if not isinstance(autre, Data):
            return NotImplemented
        lignes = pd.concat([self._data, autre.data], ignore_index=True)
        compagnies = sorted(set(self._liste_airlines) | set(autre.liste_airlines))
        return Data(lignes, compagnies)

    def __repr__(self) -> str:
        n = 0 if self._data is None else len(self._data)
        return f"Data({n} lignes, {len(self._liste_airlines)} compagnies)"
