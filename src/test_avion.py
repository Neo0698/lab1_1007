import os
import json
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier


class DonneeInvalideError(Exception):
    """Exception personnalisée levée quand une saisie utilisateur est invalide."""
    pass


class ModeleVol:
    """Modèle de prédiction de retard de vol.

    Compare plusieurs modèles simples (kNN, arbre, forêt aléatoire)
    sur le même split, garde le meilleur, et l'évalue honnêtement
    (référence + exactitude sur le test), puis le sauvegarde avec joblib
    et un fichier JSON de métadonnées.
    """

    def __init__(self, chemin_csv: str = "donnees/Airlines.csv") -> None:
        self._chemin_csv = chemin_csv
        self._le: LabelEncoder | None = None
        self._scaler: StandardScaler | None = None
        self._modele = None  # meilleur modèle après la comparaison
        self._comparaison: dict[str, float] = {}
        self._nom_modele_choisi: str = ""
        self._matrice_confusion: list[list[int]] = []
        self._liste_airlines: list[str] = []
        self._features = ["Airline_Encoded", "DayOfWeek", "Time", "Length", "Length_Carre"]
        # Renseignés par preparer(), puis écrits dans les métadonnées JSON
        self._colonnes: list[str] = []
        self._intervalles: dict[str, list[float]] = {}
        self._valeurs_categorielles: dict[str, list[str]] = {}

    # ---------------------------------------------------------
    # Chargement et exploration des données
    # ---------------------------------------------------------
    def charger_donnees(self) -> pd.DataFrame:
        """Charge le CSV et retourne le DataFrame.

        Lève FileNotFoundError si le fichier est introuvable.
        """
        if not os.path.exists(self._chemin_csv):
            raise FileNotFoundError(
                f"Erreur : le fichier '{self._chemin_csv}' est introuvable."
            )
        return pd.read_csv(self._chemin_csv)

    def preparer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode la compagnie, crée Length_Carre et retourne le DataFrame prêt.

        Ne modifie pas l'original (retourne une copie).
        """
        df = df.copy()
        df = df.dropna()  # nettoyage : on écarte les lignes incomplètes

        # --- Métadonnées, calculées depuis les données elles-mêmes ---
        # Noms des colonnes d'entrée (toutes sauf la cible)
        self._colonnes = [c for c in df.columns if c != "Delay"]
        # Valeurs possibles de la variable catégorielle
        self._valeurs_categorielles = {
            "Airline": sorted(df["Airline"].unique().tolist())
        }
        # Intervalles [min, max] observés pour chaque variable numérique
        self._intervalles = {
            colonne: [float(df[colonne].min()), float(df[colonne].max())]
            for colonne in ("DayOfWeek", "Time", "Length")
        }

        self._le = LabelEncoder()
        df["Airline_Encoded"] = self._le.fit_transform(df["Airline"])
        df["Length_Carre"] = df["Length"] ** 2
        self._liste_airlines = sorted(df["Airline"].unique())
        return df

    # ---------------------------------------------------------
    # Entraînement et évaluation honnête
    # ---------------------------------------------------------
    def entrainer(self) -> dict:
        """Compare plusieurs modèles sur le MÊME split et garde le meilleur.

        Candidats : kNN pour plusieurs valeurs de k, arbre de décision
        et forêt aléatoire, tous évalués sur le même ensemble de test
        (même graine), plus la référence (classe majoritaire).
        Le meilleur modèle devient self._modele, celui que predire()
        et sauvegarder() utiliseront.

        Retourne un dict avec, entre autres :
        'exactitude_knn' (le kNN à 5 voisins, pour mémoire),
        'exactitude_reference', 'comparaison' (nom -> exactitude),
        'meilleur_modele', 'exactitude_meilleur' et 'matrice_confusion'
        du meilleur modèle ([[vrais 0, faux 1], [faux 0, vrais 1]]).

        Limite honnête : le meilleur modèle est choisi d'après le score
        de test, ce qui rend ce score légèrement optimiste. Un jeu de
        validation séparé serait plus rigoureux.
        """
        df = self.preparer(self.charger_donnees())

        X = df[self._features]
        y = df["Delay"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Standardisation : ajustée sur l'entraînement SEULEMENT,
        # puis appliquée aux deux ensembles.
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        # --- Référence (étape 4) : classe majoritaire, sans intelligence ---
        reference = DummyClassifier(strategy="most_frequent")
        reference.fit(X_train, y_train)
        score_reference = reference.score(X_test, y_test)

        # --- Candidats, tous sur le même split ---
        candidats = {
            **{f"knn_k{k}": KNeighborsClassifier(n_neighbors=k)
               for k in (1, 3, 5, 11, 21, 31)},
            "arbre_decision": DecisionTreeClassifier(random_state=42),
            "foret_aleatoire": RandomForestClassifier(
                n_estimators=200, random_state=42),
        }

        self._comparaison = {}
        modeles_entraines = {}
        for nom, candidat in candidats.items():
            candidat.fit(X_train_scaled, y_train)
            self._comparaison[nom] = round(
                float(candidat.score(X_test_scaled, y_test)), 4)
            modeles_entraines[nom] = candidat

        # --- Choix du meilleur modèle sur ce split ---
        self._nom_modele_choisi = max(
            self._comparaison, key=self._comparaison.get)
        self._modele = modeles_entraines[self._nom_modele_choisi]
        score_meilleur = self._comparaison[self._nom_modele_choisi]

        # Matrice de confusion du meilleur modèle (lignes = réel,
        # colonnes = prédit, ordre des classes [0, 1])
        predictions = self._modele.predict(X_test_scaled)
        self._matrice_confusion = confusion_matrix(
            y_test, predictions, labels=[0, 1]).tolist()

        print(f"Référence (classe majoritaire) : {score_reference:.3f}")
        for nom, score in sorted(self._comparaison.items(),
                                 key=lambda item: item[1], reverse=True):
            print(f"{nom:<20} : {score:.3f}")
        print(f"--> Modèle retenu : {self._nom_modele_choisi} "
              f"({score_meilleur:.3f})")

        return {
            "exactitude_knn": self._comparaison["knn_k5"],
            "exactitude_reference": round(float(score_reference), 4),
            "comparaison": self._comparaison,
            "meilleur_modele": self._nom_modele_choisi,
            "exactitude_meilleur": score_meilleur,
            "matrice_confusion": self._matrice_confusion,
        }

    # ---------------------------------------------------------
    # Sauvegarde (étape 6) : joblib + métadonnées JSON
    # ---------------------------------------------------------
    def sauvegarder(self, scores: dict, dossier: str = "modeles") -> None:
        """Sauvegarde le modèle (joblib) et les métadonnées (JSON).

        Lève RuntimeError si le modèle n'a pas été entraîné.
        """
        if self._modele is None or self._scaler is None or self._le is None:
            raise RuntimeError("Le modèle doit être entraîné avant d'être sauvegardé.")

        os.makedirs(dossier, exist_ok=True)
        joblib.dump(self._modele, os.path.join(dossier, "modele.joblib"))
        joblib.dump(self._scaler, os.path.join(dossier, "scaler.joblib"))
        joblib.dump(self._le, os.path.join(dossier, "label_encoder.joblib"))

        metadonnees = {
            # Noms des colonnes du jeu de données et du modèle
            "colonnes": self._colonnes,
            "features": self._features,
            "cible": "Delay",
            # Valeurs possibles des variables catégorielles
            "variables_categorielles": self._valeurs_categorielles,
            "airlines_possibles": self._liste_airlines,
            # Intervalles des variables numériques, calculés par preparer()
            "variables_numeriques": {
                colonne: {"min": bornes[0], "max": bornes[1]}
                for colonne, bornes in self._intervalles.items()
            },
            "intervalle_dayofweek": self._intervalles.get("DayOfWeek"),
            "intervalle_time": self._intervalles.get("Time"),
            "intervalle_length": self._intervalles.get("Length"),
            # Métriques obtenues sur l'ensemble de test
            "metriques": {
                "exactitude_knn": scores["exactitude_knn"],
                "exactitude_reference": scores["exactitude_reference"],
            },
            "exactitude_knn": scores["exactitude_knn"],
            "exactitude_reference": scores["exactitude_reference"],
            # Comparaison des modèles et modèle retenu pour predire()
            "modele_choisi": scores.get("meilleur_modele",
                                         self._nom_modele_choisi),
            "exactitude_modele_choisi": scores.get("exactitude_meilleur"),
            "comparaison_modeles": scores.get("comparaison",
                                              self._comparaison),
            "matrice_confusion": scores.get("matrice_confusion",
                                            self._matrice_confusion),
        }
        with open(os.path.join(dossier, "metadonnees.json"), "w", encoding="utf-8") as f:
            json.dump(metadonnees, f, ensure_ascii=False, indent=2)

        print(f"--- Modèle et métadonnées sauvegardés dans '{dossier}/' ---")

    # ---------------------------------------------------------
    # Prédiction (appelée par l'interface)
    # ---------------------------------------------------------
    def predire(self, airline_choisie: str, jour_semaine: int,
                heure: int, minutes: int, duree: int) -> np.ndarray:
        """Retourne les probabilités [p_a_l_heure, p_retard] pour un vol.

        Lève DonneeInvalideError si une saisie est hors intervalle,
        RuntimeError si le modèle n'est pas prêt.
        """
        if self._modele is None or self._scaler is None or self._le is None:
            raise RuntimeError("Le modèle doit être chargé ou entraîné avant de prédire.")
        if airline_choisie not in self._liste_airlines:
            raise DonneeInvalideError(f"Compagnie inconnue : '{airline_choisie}'.")
        if not (1 <= jour_semaine <= 7):
            raise DonneeInvalideError("Le jour de la semaine doit être entre 1 et 7.")
        if not (0 <= heure <= 23 and 0 <= minutes <= 59):
            raise DonneeInvalideError("L'heure doit être entre 00h00 et 23h59.")
        if duree <= 0:
            raise DonneeInvalideError("La durée du vol doit être positive.")

        time_vol = (heure * 60) + minutes
        airline_encoded = self._le.transform([airline_choisie])[0]

        nouveau_vol = np.array([[
            airline_encoded, jour_semaine, time_vol, duree, duree ** 2
        ]])
        nouveau_vol_scaled = self._scaler.transform(nouveau_vol)

        probas = self._modele.predict_proba(nouveau_vol_scaled)[0]
        return probas  # [proba_a_l_heure, proba_retard]


def main() -> None:
    """Point d'entrée : entraîne, évalue et sauvegarde le modèle."""
    modele = ModeleVol("donnees/Airlines.csv")
    try:
        scores = modele.entrainer()
        modele.sauvegarder(scores)
    except FileNotFoundError as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    main()