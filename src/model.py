from __future__ import annotations

import os
import json


import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from .Data import Data
from .exceptions import DonneeInvalideError
from .outils import chronometrer

def sigmoid_weights(distances):

    return 1 / (1 + np.exp(distances - 1.0))


COLONNES_CSV = {"Airline", "Origin", "DayOfWeek", "Time", "Length", "Delay"}
# Fichiers écrits par sauvegarder() et relus par charger()
FICHIERS_MODELE = ("knn.joblib", "scaler.joblib", "label_encoder.joblib", "metadonnees.json")


def taux_retard_lisse(origines: pd.Series, retards: pd.Series, poids: int = 20) -> pd.Series:
    """Calcule le taux de retard de chaque aéroport, en fonction de toutes les lignes contenant cette aeroport

   
    Args:
        origines: Aéroport de départ de chaque vol.
        retards: 1 si le vol est en retard, 0 sinon (même index que origines).
        poids: Nombre de « vols fictifs » au taux global ajoutés à chaque aéroport.

    retourne:
        Série indexée par aéroport, valeurs entre 0 et 1.

    erreur:
        ValueError: si aucun vol n'est fourni.
    """
    if len(retards) == 0:
        raise ValueError("Impossible de calculer un taux de retard sans aucun vol.")
    taux_global = retards.mean()
    stats = retards.groupby(origines).agg(["sum", "count"])
    return (stats["sum"] + poids * taux_global) / (stats["count"] + poids)


class PoidsVariables:
    """Poids de chaque variable dans la distance du kNN (1 = poids normal, 0 = ignorée).

    Les poids s'appliquent après la normalisation : un poids k multiplie
    l'importance de la variable dans la distance euclidienne par k².

    Args:
        k_duree: Poids de la durée du vol (Length et Length_Carre).
        k_aeroport: Poids du taux de retard de l'aéroport de départ.
        k_jour: Poids du jour de la semaine.
        k_heure: Poids de l'heure de départ.
        k_compagnie: Poids de la compagnie.

    erreur:
        ValueError: si un poids est négatif.
    """

    def __init__(self, k_duree: float = 1.0, k_aeroport: float = 1.0, k_jour: float = 1.0,
                 k_heure: float = 1.0, k_compagnie: float = 1.0) -> None:
        if min(k_duree, k_aeroport, k_jour, k_heure, k_compagnie) < 0:
            raise ValueError("Les poids doivent être positifs ou nuls.")
        self._poids = {
            "Airline_Encoded": k_compagnie,
            "Origin_Taux": k_aeroport,
            "DayOfWeek": k_jour,
            "Time": k_heure,
            "Length": k_duree,
            "Length_Carre": k_duree,
        }

    @classmethod
    def depuis_dict(cls, poids: dict) -> "PoidsVariables":
        """Recrée les poids à partir du dictionnaire enregistré dans le JSON."""
        return cls(k_duree=poids.get("Length", 1.0), k_aeroport=poids.get("Origin_Taux", 1.0),
                   k_jour=poids.get("DayOfWeek", 1.0), k_heure=poids.get("Time", 1.0),
                   k_compagnie=poids.get("Airline_Encoded", 1.0))

    def appliquer(self, X: np.ndarray, features: list[str]) -> np.ndarray:
        """Multiplie chaque colonne de X (déjà normalisé) par son poids."""
        vecteur = np.array([self._poids.get(f, 1.0) for f in features]) # vectorisation des poids pour chaque feature
        return np.asarray(X) * vecteur                # applique les poids aux colonnes correspondantes de X

    def en_dict(self) -> dict:
        """Poids à enregistrer dans le JSON."""
        return dict(self._poids)

def poids_retenus() -> PoidsVariables:
    """Poids choisis avec choix_modele.score_selon_poids (jeu complet, k=31).

    Le jour de la semaine influence très peu le retard : on réduit fortement son poids (0,2).
    La compagnie compte 3 fois plus et la durée 2 fois moins.
    Exactitude : 64,5 % contre 63,4 % avec tous les poids à 1 (65,3 % avec un jour à 0).
    """
    return PoidsVariables(k_jour=0.2, k_compagnie=3, k_duree=0.5)


class ModeleVol:
    """Modèle de prédiction de retard de vol basé sur un kNN.

    Contient les données (Data) et toutes les transformations apprises sur
    l'entraînement : encodeur des compagnies, taux de retard des aéroports, scaler.

    Args:
        chemin_csv: Chemin du jeu de données.
        n: Nombre de voisins du kNN.
        poids: Poids des variables dans la distance (par défaut, poids_retenus()).
    """

    def __init__(self, chemin_csv: str = "donnees/Airlines.csv", n: int = 31,
                 poids: PoidsVariables | None = None) -> None:
        self._chemin_csv = chemin_csv
        self.nb_neighbors = n
        self.poids = poids if poids is not None else poids_retenus()
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

        erreur:
            RuntimeError: si aucun modèle n'a été chargé ni sauvegardé.
        """
        if self._metadonnees is None:
            raise RuntimeError("Aucun modèle chargé : appeler charger() ou sauvegarder() d'abord.")
        return self._metadonnees

    # ---------------------------------------------------------
    # Préparation
    # ---------------------------------------------------------
    def preparer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apprend l'encodage des compagnies, puis renvoie les vols prêts (voir _ajouter_colonnes).

        erreur:
            ValueError: si une colonne attendue (dont Origin) manque dans les données.
        """
        df = self._nettoyer(df)
        self._le = LabelEncoder().fit(df["Airline"])
        self.donnees.charger_airlines([str(a) for a in self._le.classes_])  # classes_ : déjà triées
        return self._ajouter_colonnes(df)

    @staticmethod
    def _nettoyer(df: pd.DataFrame) -> pd.DataFrame:
        """Vérifie les colonnes du CSV et retire les lignes incomplètes.

        erreur:
            ValueError: si une colonne attendue (dont Origin) manque dans les données.
        """
        manquantes = COLONNES_CSV - set(df.columns)
        if manquantes:
            raise ValueError(f"Colonnes manquantes dans les données : {sorted(manquantes)}. "
                             "Régénérez le CSV avec : python -m src.create_dataset")
        return df.dropna()

    def _ajouter_colonnes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renvoie une copie de df avec Airline_Encoded (encodeur déjà appris) et Length_Carre."""
        df = df.copy()
        df["Airline_Encoded"] = self._le.transform(df["Airline"])
        df["Length_Carre"] = df["Length"] ** 2  # permet d'apporter plus d'importance sur la duree du vol
        return df

    def _ajouter_taux_aeroport(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renvoie une copie de df avec la colonne Origin_Taux (taux de retard de l'aéroport).

        Un aéroport absent de l'entraînement reçoit le taux global.
        """
        df = df.copy()
        df["Origin_Taux"] = df["Origin"].map(self._taux_aeroports).fillna(self._taux_global)
        return df

    def _transformation_espace_knn(self, X: pd.DataFrame) -> np.ndarray:
        """Normalise X avec le scaler appris, puis applique les poids : c'est l'espace du kNN."""
        return self.poids.appliquer(self._scaler.transform(X), self.donnees.features)

    # ---------------------------------------------------------
    # Entraînement et évaluation
    # ---------------------------------------------------------
    @chronometrer
    def entrainer(self) -> dict:
        """Entraîne le kNN, l'évalue sur le test et retourne les scores.

        erreur:
            FileNotFoundError: si le CSV est introuvable.
            ValueError: si une colonne attendue manque dans le CSV.
        """
        if self.donnees.data is None:
            self.donnees.charger_donnees(self._chemin_csv)
        self.donnees.train_test_split(self.preparer(self.donnees.data)) # divise les données en ensembles d'entraînement et de test

        
        df_train = self.donnees.df_train

        taux = taux_retard_lisse(df_train["Origin"], df_train["Delay"])
        
        self._taux_aeroports = {str(a): float(t) for a, t in taux.items()}
        self._taux_global = float(df_train["Delay"].mean())
        self.donnees.charger_aeroports(sorted(self._taux_aeroports))
        self.donnees.transformer(self._ajouter_taux_aeroport) # ajoute le taux de retard par aéroport aux données tests et d'entraînement

        # Normalisation (apprise sur l'entraînement, appliquée au test), puis poids des variables
        self._scaler = StandardScaler().fit(self.donnees.X_train)
        self._X_train_scaled = self._transformation_espace_knn(self.donnees.X_train)
        self._X_test_scaled = self._transformation_espace_knn(self.donnees.X_test)
        
        reference = DummyClassifier(strategy="most_frequent")
        reference.fit(self.donnees.X_train, self.donnees.y_train) # entraîne le classifieur Dummy sur les données d'entraînement
        score_reference = reference.score(self.donnees.X_test, self.donnees.y_test) # score renvoie l'exactitude de la référence par rapport au test

        self._knn = KNeighborsClassifier(n_neighbors=self.nb_neighbors, metric="euclidean")
        self._knn.fit(self._X_train_scaled, self.donnees.y_train) # entraîne le kNN sur les données d'entraînement
        score_knn = self._knn.score(self._X_test_scaled, self.donnees.y_test) # score renvoie l'exactitude de knn par rapport au test

        

        return {
            "exactitude_knn": round(float(score_knn), 4),
            "exactitude_reference": round(float(score_reference), 4),
        }

    def test_knn(self, n: int = 5, metric: str = "euclidean", weight: bool = False) -> float:
        """Entraîne un kNN avec d'autres réglages et renvoie son exactitude sur le test.

        À appeler après entrainer() (utilise les mêmes données normalisées).

        erreur:
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

        erreur:
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
        return self._transformation_espace_knn(nouveau_vol)

    def predire(self, airline_choisie: str, aeroport: str, jour_semaine: int,
                heure: int, minutes: int, duree: float) -> np.ndarray:
        """Retourne les probabilités [p_a_l_heure, p_retard] pour un vol.

        erreur:
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
        lignes = self.donnees.df_train.iloc[indices[0]] # iloc permet de sélectionner les lignes par position dans le DataFrame
        vols = lignes[self.donnees.features].copy()
        vols.insert(0, "Origin", lignes["Origin"].values)
        vols["Delay"] = lignes["Delay"].values
        vols["Distance"] = distances[0]
        return vols

    
    def charger_csv(self) -> pd.DataFrame:
        """Renvoie les vols bruts du CSV du modèle, lus une seule fois.

        erreur:
            FileNotFoundError: si le CSV est introuvable.
        """
        if self.donnees.data is None:
            self.donnees.charger_donnees(self._chemin_csv)
        return self.donnees.data

    @staticmethod
    def _chemins_fichiers(dossier: str) -> dict[str, str]:
        """Chemin de chaque fichier du modèle dans dossier."""
        return {nom: os.path.join(dossier, nom) for nom in FICHIERS_MODELE}

    def sauvegarder(self, scores: dict, dossier: str = "modeles") -> None:
        """Sauvegarde le modèle (joblib) et les métadonnées (JSON).

        Le JSON réunit deux parties : celle de Data (colonnes, compagnies, aéroports)
        et celle du modèle (taux des aéroports, poids, scores).

        erreur:
            RuntimeError: si le modèle n'est pas entraîné.
        """
        if self._knn is None or self._scaler is None or self._le is None:
            raise RuntimeError("Le modèle doit être entraîné avant d'être sauvegardé.")

        os.makedirs(dossier, exist_ok=True)
        chemins = self._chemins_fichiers(dossier)
        joblib.dump(self._knn, chemins["knn.joblib"])
        joblib.dump(self._scaler, chemins["scaler.joblib"])
        joblib.dump(self._le, chemins["label_encoder.joblib"])

        metadonnees = {
            **self.donnees.en_dict(),   # partie Data
            "cible": "Delay",
            "dataset": self._chemin_csv,
            "taux_retard_aeroports": self._taux_aeroports,
            "taux_retard_global": self._taux_global,
            "poids_variables": self.poids.en_dict(),
            "intervalle_dayofweek": [1, 7],
            "intervalle_time": [0, 1439],
            "intervalle_length": [0, 655],
            "exactitude_knn": scores["exactitude_knn"],
            "exactitude_reference": scores["exactitude_reference"],
        }
        with open(chemins["metadonnees.json"], "w", encoding="utf-8") as f:
            json.dump(metadonnees, f, ensure_ascii=False, indent=2)
        self._metadonnees = metadonnees

        print(f"--- Modèle et métadonnées sauvegardés dans '{dossier}/' ---")

    def charger(self, dossier: str = "modeles") -> None:
        """Charge le modèle sauvegardé, puis reconstruit les vols d'entraînement.

        Args:
            dossier: Dossier contenant les fichiers .joblib et .json.

        erreur:
            FileNotFoundError: si un des fichiers du modèle ou le CSV est absent.
        """
        chemins = self._chemins_fichiers(dossier)
        for chemin in chemins.values():
            if not os.path.exists(chemin):
                raise FileNotFoundError(f"Fichier de modèle manquant : {chemin}")

        # 1. Partie modèle : objets scikit-learn et valeurs apprises
        self._knn = joblib.load(chemins["knn.joblib"])
        self._scaler = joblib.load(chemins["scaler.joblib"])
        self._le = joblib.load(chemins["label_encoder.joblib"])
        with open(chemins["metadonnees.json"], encoding="utf-8") as f:
            self._metadonnees = json.load(f)
        self._taux_aeroports = self._metadonnees["taux_retard_aeroports"]
        self._taux_global = self._metadonnees["taux_retard_global"]
        # Mêmes poids qu'à l'entraînement, sinon les distances ne correspondent plus
        self.poids = PoidsVariables.depuis_dict(self._metadonnees.get("poids_variables", {}))

        # 2. Partie Data : colonnes et listes connues
        self.donnees.restaurer(self._metadonnees)

        # 3. Vols d'entraînement, nécessaires à voisins()
        self._reconstruire_entrainement()

    def _reconstruire_entrainement(self) -> None:
        """Relit le CSV et refait le même découpage qu'à l'entraînement (graine 42).

        Utilise l'encodeur et les taux déjà chargés : rien n'est réappris.

        erreur:
            FileNotFoundError: si le CSV est absent.
        """
        df = self._nettoyer(self.charger_csv())
        self.donnees.train_test_split(self._ajouter_colonnes(df))
        self.donnees.transformer(self._ajouter_taux_aeroport)


def main() -> None:
    """Point d'entrée : explore les données, puis entraîne, évalue et sauvegarde le modèle."""
    modele = ModeleVol("donnees/Airlines.csv")
    try:
        modele.charger_csv()   # lu une fois, réutilisé par entrainer()
        print(modele.donnees.resume_qualite())
        chemin = modele.donnees.graphique_exploration(os.path.join("donnees", "exploration.png"))
        print(f"--- Graphique d'exploration enregistré dans '{chemin}' ---")
        scores = modele.entrainer()
        modele.sauvegarder(scores)
    except FileNotFoundError as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()
