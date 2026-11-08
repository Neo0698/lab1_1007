from __future__ import annotations

import os
import json
from typing import Callable

import joblib

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from .dataset import Dataset
from .exceptions import DonneeInvalideError
from .outils import chronometrer

def sigmoid_weights(distances):
    # On peut inverser et centrer selon un seuil, par exemple :
    # Plus la distance est grande, plus le poids tend vers 0.
    return 1 / (1 + np.exp(distances - 1.0))

COLONNES_CSV = {"Airline", "Origin", "DayOfWeek", "Time", "Length", "Delay"}
FEATURES = ["Airline_Encoded", "Origin_Taux", "DayOfWeek", "Time", "Length", "Length_Carre"]


def taux_retard_lisse(origines: pd.Series, retards: pd.Series, poids: int = 20) -> pd.Series:
    """Calcule le taux de retard de chaque aéroport, lissé vers le taux global.

    taux = (retards de l'aéroport + poids × taux global) / (vols de l'aéroport + poids)
    Un petit aéroport (peu de vols) reste proche du taux global ; un grand aéroport
    garde son propre taux. Calcul vectorisé (pandas/NumPy, sans boucle Python).

    Args:
        origines: Aéroport de départ de chaque vol.
        retards: 1 si le vol est en retard, 0 sinon (même index que origines).
        poids: Nombre de « vols fictifs » au taux global ajoutés à chaque aéroport.

    Returns:
        Série indexée par aéroport, valeurs entre 0 et 1.

    Raises:
        ValueError: si aucun vol n'est fourni.
    """
    if len(retards) == 0:
        raise ValueError("Impossible de calculer un taux de retard sans aucun vol.")
    taux_global = retards.mean()
    stats = retards.groupby(origines).agg(["sum", "count"])
    return (stats["sum"] + poids * taux_global) / (stats["count"] + poids)


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

        Raises:
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

    # --- Exploration ---------------------------------------------------------
    def resume_qualite(self) -> pd.DataFrame:
        """Résume la qualité des données, une ligne par colonne.

        Returns:
            DataFrame avec le type, le nombre de valeurs manquantes, le min et le max.

        Raises:
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

        Returns:
            Le chemin du fichier enregistré.

        Raises:
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


class vol:
    def __init__(self,k_duree=1, k_aeroport=1, k_day=1):
        return


class ModeleVol:
    """Modèle de prédiction de retard de vol basé sur un kNN.

    Contient les données (Data) et toutes les transformations apprises sur
    l'entraînement : encodeur des compagnies, taux de retard des aéroports, scaler.

    Args:
        chemin_csv: Chemin du jeu de données.
        n: Nombre de voisins du kNN.
    """

    def __init__(self, chemin_csv: str = "donnees/Airlines.csv", n: int = 31) -> None:
        self._chemin_csv = chemin_csv
        self.nb_neighbors = n
        self.donnees = Data()
        self._le: LabelEncoder | None = None
        self._taux_aeroports: dict[str, float] | None = None
        self._taux_global: float | None = None
        self._scaler: StandardScaler | None = None
        self._knn: KNeighborsClassifier | None = None
        self._X_train_scaled: np.ndarray | None = None
        self._X_test_scaled: np.ndarray | None = None
        self._metadonnees: dict | None = None

    @property
    def _liste_airlines(self) -> list[str]:
        return self.donnees.liste_airlines

    @property
    def metadonnees(self) -> dict:
        """Métadonnées du modèle chargé ou sauvegardé.

        Raises:
            RuntimeError: si aucun modèle n'a été chargé ni sauvegardé.
        """
        if self._metadonnees is None:
            raise RuntimeError("Aucun modèle chargé : appeler charger() ou sauvegarder() d'abord.")
        return self._metadonnees

    # ---------------------------------------------------------
    # Préparation
    # ---------------------------------------------------------
    def preparer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode la compagnie, crée Length_Carre et retourne le DataFrame prêt.

        Raises:
            ValueError: si une colonne attendue (dont Origin) manque dans les données.
        """
        manquantes = COLONNES_CSV - set(df.columns)
        if manquantes:
            raise ValueError(f"Colonnes manquantes dans les données : {sorted(manquantes)}. "
                             "Régénérez le CSV avec : python -m src.create_dataset")
        df = df.copy().dropna()
        self._le = LabelEncoder()
        df["Airline_Encoded"] = self._le.fit_transform(df["Airline"])
        df["Length_Carre"] = df["Length"] ** 2  # permet d'apporter plus d'importance sur la duree du vol
        self.donnees.charger_airlines(sorted(df["Airline"].unique()))
        return df

    def _ajouter_taux_aeroport(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renvoie une copie de df avec la colonne Origin_Taux (taux de retard de l'aéroport).

        Un aéroport absent de l'entraînement reçoit le taux global.
        """
        df = df.copy()
        df["Origin_Taux"] = df["Origin"].map(self._taux_aeroports).fillna(self._taux_global)
        return df

    # ---------------------------------------------------------
    # Entraînement et évaluation
    # ---------------------------------------------------------
    @chronometrer
    def entrainer(self) -> dict:
        """Entraîne le kNN, l'évalue sur le test et retourne les scores.

        Raises:
            FileNotFoundError: si le CSV est introuvable.
            ValueError: si une colonne attendue manque dans le CSV.
        """
        if self.donnees.data is None:
            self.donnees.charger_donnees(self._chemin_csv)
        self.donnees.train_test_split(self.preparer(self.donnees.data))

        # Taux de retard par aéroport : calculé sur l'entraînement SEULEMENT (pas de fuite du test)
        df_train = self.donnees.df_train
        taux = taux_retard_lisse(df_train["Origin"], df_train["Delay"])
        self._taux_aeroports = {str(a): float(t) for a, t in taux.items()}
        self._taux_global = float(df_train["Delay"].mean())
        self.donnees.charger_aeroports(sorted(self._taux_aeroports))
        self.donnees.transformer(self._ajouter_taux_aeroport)

        # Normalisation : apprise sur l'entraînement, appliquée au test
        self._scaler = StandardScaler()
        self._X_train_scaled = self._scaler.fit_transform(self.donnees.X_train)
        self._X_test_scaled = self._scaler.transform(self.donnees.X_test)

        # Référence (classe majoritaire)
        reference = DummyClassifier(strategy="most_frequent")
        reference.fit(self.donnees.X_train, self.donnees.y_train)
        score_reference = reference.score(self.donnees.X_test, self.donnees.y_test)

        # Modèle kNN
        self._knn = KNeighborsClassifier(n_neighbors=self.nb_neighbors, metric="euclidean")
        self._knn.fit(self._X_train_scaled, self.donnees.y_train)
        score_knn = self._knn.score(self._X_test_scaled, self.donnees.y_test)

        print(f"Référence (classe majoritaire) : {score_reference:.3f}")
        print(f"kNN (k={self.nb_neighbors})                : {score_knn:.3f}")

        return {
            "exactitude_knn": round(float(score_knn), 4),
            "exactitude_reference": round(float(score_reference), 4),
        }

    def test_knn(self, n: int = 5, metric: str = "euclidean", weight: bool = False) -> float:
        """Entraîne un kNN avec d'autres réglages et renvoie son exactitude sur le test.

        À appeler après entrainer() (utilise les mêmes données normalisées).

        Raises:
            RuntimeError: si entrainer() n'a pas été appelé.
        """
        if self._X_train_scaled is None:
            raise RuntimeError("Appeler entrainer() avant test_knn().")
        if weight:
            self._knn = KNeighborsClassifier(n_neighbors=n, metric=metric, weights=sigmoid_weights)
        else:
            self._knn = KNeighborsClassifier(n_neighbors=n, metric=metric)
        self._knn.fit(self._X_train_scaled, self.donnees.y_train)
        return float(self._knn.score(self._X_test_scaled, self.donnees.y_test))

    # ---------------------------------------------------------
    # Prédiction
    # ---------------------------------------------------------
    def _preparer_vol(self, airline_choisie: str, aeroport: str, jour_semaine: int,
                      heure: int, minutes: int, duree: float) -> np.ndarray:
        """Valide les saisies et renvoie le vol normalisé.

        Raises:
            RuntimeError: si le modèle n'est pas entraîné ou chargé.
            DonneeInvalideError: si une saisie est inconnue ou hors intervalle.
        """
        if (self._knn is None or self._scaler is None or self._le is None
                or self._taux_aeroports is None):
            raise RuntimeError("Le modèle doit être entraîné avant de prédire.")
        if airline_choisie not in self._liste_airlines:
            raise DonneeInvalideError(f"Compagnie inconnue : '{airline_choisie}'.")
        if aeroport not in self._taux_aeroports:
            raise DonneeInvalideError(f"Aéroport de départ inconnu : '{aeroport}'.")
        if not (1 <= jour_semaine <= 7):
            raise DonneeInvalideError("Le jour de la semaine doit être entre 1 et 7.")
        if not (0 <= heure <= 23 and 0 <= minutes <= 59):
            raise DonneeInvalideError("L'heure doit être entre 00h00 et 23h59.")
        if duree <= 0:
            raise DonneeInvalideError("La durée du vol doit être positive.")

        valeurs = {
            "Airline_Encoded": self._le.transform([airline_choisie])[0],
            "Origin_Taux": self._taux_aeroports[aeroport],
            "DayOfWeek": jour_semaine,
            "Time": (heure * 60) + minutes,
            "Length": duree,
            "Length_Carre": duree ** 2,
        }
        features = self.donnees.features
        nouveau_vol = pd.DataFrame([[valeurs[f] for f in features]], columns=features)
        return self._scaler.transform(nouveau_vol)

    def predire(self, airline_choisie: str, aeroport: str, jour_semaine: int,
                heure: int, minutes: int, duree: float) -> np.ndarray:
        """Retourne les probabilités [p_a_l_heure, p_retard] pour un vol.

        Raises:
            RuntimeError: si le modèle n'est pas entraîné ou chargé.
            DonneeInvalideError: si une saisie est invalide.
        """
        x = self._preparer_vol(airline_choisie, aeroport, jour_semaine, heure, minutes, duree)
        return self._knn.predict_proba(x)[0]

    def voisins(self, airline_choisie: str, aeroport: str, jour_semaine: int,
                heure: int, minutes: int, duree: float) -> pd.DataFrame:
        """Retourne les k vols d'entraînement les plus proches, avec aéroport, retard et distance."""
        x = self._preparer_vol(airline_choisie, aeroport, jour_semaine, heure, minutes, duree)
        distances, indices = self._knn.kneighbors(x)
        lignes = self.donnees.df_train.iloc[indices[0]]
        vols = lignes[self.donnees.features].copy()
        vols.insert(0, "Origin", lignes["Origin"].values)
        vols["Delay"] = lignes["Delay"].values
        vols["Distance"] = distances[0]
        return vols

    # ---------------------------------------------------------
    # Chargement et sauvegarde
    # ---------------------------------------------------------
    def charger(self, dossier: str = "modeles") -> None:
        """Charge le modèle, le scaler, le label encoder et les métadonnées sauvegardés.

        Reconstruit aussi l'ensemble d'entraînement (même découpage, random_state=42)
        pour que voisins() fonctionne sans ré-entraîner le modèle.

        Args:
            dossier: Chemin vers le dossier contenant les fichiers .joblib et .json.

        Raises:
            FileNotFoundError: si un des fichiers du modèle ou le CSV est absent.
        """
        for nom in ("knn.joblib", "scaler.joblib", "label_encoder.joblib", "metadonnees.json"):
            if not os.path.exists(os.path.join(dossier, nom)):
                raise FileNotFoundError(f"Fichier de modèle manquant : {os.path.join(dossier, nom)}")

        with open(os.path.join(dossier, "metadonnees.json"), "r", encoding="utf-8") as f:
            self._metadonnees = json.load(f)
        self.donnees.features = self._metadonnees["features"]
        self._taux_aeroports = self._metadonnees["taux_retard_aeroports"]
        self._taux_global = self._metadonnees["taux_retard_global"]

        # Même découpage (random_state=42) que lors de l'entraînement → mêmes voisins
        self.donnees.train_test_split(self.preparer(self.donnees.charger_donnees(self._chemin_csv)))
        self.donnees.transformer(self._ajouter_taux_aeroport)

        self._knn = joblib.load(os.path.join(dossier, "knn.joblib"))
        self._scaler = joblib.load(os.path.join(dossier, "scaler.joblib"))
        self._le = joblib.load(os.path.join(dossier, "label_encoder.joblib"))
        self.donnees.charger_airlines(self._metadonnees["airlines_possibles"])
        self.donnees.charger_aeroports(self._metadonnees["aeroports_possibles"])

    def sauvegarder(self, scores: dict, dossier: str = "modeles") -> None:
        """Sauvegarde le modèle (joblib) et les métadonnées (JSON).

        Raises:
            RuntimeError: si le modèle n'est pas entraîné.
        """
        if self._knn is None or self._scaler is None or self._le is None:
            raise RuntimeError("Le modèle doit être entraîné avant d'être sauvegardé.")

        os.makedirs(dossier, exist_ok=True)
        joblib.dump(self._knn, os.path.join(dossier, "knn.joblib"))
        joblib.dump(self._scaler, os.path.join(dossier, "scaler.joblib"))
        joblib.dump(self._le, os.path.join(dossier, "label_encoder.joblib"))

        metadonnees = {
            "features": self.donnees.features,
            "cible": "Delay",
            "airlines_possibles": [str(a) for a in self.donnees.liste_airlines],
            "aeroports_possibles": self.donnees.liste_aeroports,
            "taux_retard_aeroports": self._taux_aeroports,
            "taux_retard_global": self._taux_global,
            "dataset": self._chemin_csv,
            "intervalle_dayofweek": [1, 7],
            "intervalle_time": [0, 1439],
            "intervalle_length": [0, 655],
            "exactitude_knn": scores["exactitude_knn"],
            "exactitude_reference": scores["exactitude_reference"],
        }
        with open(os.path.join(dossier, "metadonnees.json"), "w", encoding="utf-8") as f:
            json.dump(metadonnees, f, ensure_ascii=False, indent=2)
        self._metadonnees = metadonnees

        print(f"--- Modèle et métadonnées sauvegardés dans '{dossier}/' ---")


def main() -> None:
    """Point d'entrée : explore les données, puis entraîne, évalue et sauvegarde le modèle."""
    modele = ModeleVol("donnees/Airlines.csv")
    try:
        modele.donnees.charger_donnees("donnees/Airlines.csv")
        print(modele.donnees.resume_qualite())
        chemin = modele.donnees.graphique_exploration(os.path.join("donnees", "exploration.png"))
        print(f"--- Graphique d'exploration enregistré dans '{chemin}' ---")
        scores = modele.entrainer()
        modele.sauvegarder(scores)
    except FileNotFoundError as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()
