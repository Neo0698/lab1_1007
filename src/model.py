import os
import json
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from src.dataset import Dataset
def sigmoid_weights(distances):
    # On peut inverser et centrer selon un seuil, par exemple :
    # Plus la distance est grande, plus le poids tend vers 0.
    return 1 / (1 + np.exp(distances - 1.0))

class DonneeInvalideError(Exception):
    """Exception personnalisée levée quand une saisie utilisateur est invalide."""
    pass
class Data(Dataset):
    def __init__(self,data=None,airlines=[]):
        self._liste_airlines: list[str] = airlines
        self._data = data
    @property
    def data(self):
        return self._data
    @property
    def liste_airlines(self):
        return self._liste_airlines
    def __add__(self,values):
        if(type(values)!=Data):
            raise ValueError("can only concentrate Data with Data")

        self._data+=values.data
        self._liste_airlines+=values.liste_airlines
    
    def charger_donnees(self,path:str) -> pd.DataFrame:
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Erreur : le fichier '{self._chemin_csv}' est introuvable.")
        self._data=pd.read_csv(path)



class ModeleVol:
    """Modèle de prédiction de retard de vol basé sur un kNN."""

    def __init__(self, chemin_csv: str = "donnees/Airlines.csv",n=31) -> None:
        self._chemin_csv = chemin_csv
        self._le: LabelEncoder | None = None
        self._scaler: StandardScaler | None = None
        self._knn: KNeighborsClassifier | None = None
        
        self._features = ["Airline_Encoded", "DayOfWeek", "Time", "Length", "Length_Carre"]
        self.nb_neighbors = n
        self.X_train = None  # ✅ Ajout pour la méthode voisins()
        self.y_train = None  # ✅ Ajout pour la méthode voisins()
        self.donnees=Data()
        self.donnees.charger_donnees(chemin_csv)

    # ---------------------------------------------------------
    # Chargement et préparation des données
    # ---------------------------------------------------------
    

    def preparer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode la compagnie, crée Length_Carre et retourne le DataFrame prêt."""
        df = df.copy().dropna()
        self._le = LabelEncoder()
        df["Airline_Encoded"] = self._le.fit_transform(df["Airline"])
        df["Length_Carre"] = df["Length"] ** 2  # Calcul vectorisé (NumPy)
        self._liste_airlines = sorted(df["Airline"].unique())
        return df

    # ---------------------------------------------------------
    # Entraînement et évaluation
    # ---------------------------------------------------------
    def entrainer(self) -> dict:
        """Entraîne le kNN, l'évalue sur le test et retourne les scores."""
        print("self.donnees.data()",self.donnees.data)
        df = self.preparer(self.donnees.data)
        
        X = df[self._features]
        y = df["Delay"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ✅ STOCKE X_train ET y_train POUR LA MÉTHODE voisins()
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()

        self._scaler = StandardScaler()
        self.X_train_scaled = self._scaler.fit_transform(X_train)
        self.X_test_scaled = self._scaler.transform(X_test)

        # Référence (classe majoritaire)
        reference = DummyClassifier(strategy="most_frequent")
        reference.fit(X_train, y_train)
        score_reference = reference.score(X_test, y_test)

        # Modèle kNN
        self._knn = KNeighborsClassifier(n_neighbors=self.nb_neighbors, metric="euclidean")
        self._knn.fit(self.X_train_scaled, self.y_train)
        score_knn = self._knn.score(self.X_test_scaled, self.y_test)

        print(f"Référence (classe majoritaire) : {score_reference:.3f}")
        print(f"kNN (k={self.nb_neighbors})                : {score_knn:.3f}")

        return {
            "exactitude_knn": round(float(score_knn), 4),
            "exactitude_reference": round(float(score_reference), 4),
        }
    def test_knn(self,n=5, metric="euclidean", weight=False):
        if(weight):
            self._knn = KNeighborsClassifier(n_neighbors=n, metric=metric, weights=sigmoid_weights)
        else:
            self._knn = KNeighborsClassifier(n_neighbors=n, metric=metric)
        self._knn.fit(self.X_train_scaled, self.y_train)
        score_knn = self._knn.score(self.X_test_scaled, self.y_test)
        return score_knn
    # ---------------------------------------------------------
    # Prédiction
    # ---------------------------------------------------------
    def _preparer_vol(self, airline_choisie: str, jour_semaine: int,
                      heure: int, minutes: int, duree: int) -> np.ndarray:
        """Valide les saisies et renvoie le vol normalisé."""
        if self._knn is None or self._scaler is None or self._le is None:
            raise RuntimeError("Le modèle doit être entraîné avant de prédire.")
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
        nouveau_vol = np.array([[airline_encoded, jour_semaine, time_vol, duree, duree ** 2]])
        return self._scaler.transform(nouveau_vol)

    def predire(self, airline_choisie, jour_semaine, heure, minutes, duree) -> np.ndarray:
        """Retourne les probabilités [p_a_l_heure, p_retard] pour un vol."""
        x = self._preparer_vol(airline_choisie, jour_semaine, heure, minutes, duree)
        return self._knn.predict_proba(x)[0]

    def voisins(self, airline_choisie, jour_semaine, heure, minutes, duree) -> pd.DataFrame:
        """Retourne les k vols d'entraînement les plus proches, avec leur retard et leur distance."""
        x = self._preparer_vol(airline_choisie, jour_semaine, heure, minutes, duree)
        distances, indices = self._knn.kneighbors(x)
        vols = self.X_train.iloc[indices[0]].copy()
        vols["Delay"] = self.y_train.iloc[indices[0]].values
        vols["Distance"] = distances[0]
        return vols

    def charger(self, dossier: str = "modeles") -> None:
        """Charge le modèle, le scaler et le label encoder depuis les fichiers sauvegardés.

        Args:
            dossier: Chemin vers le dossier contenant les fichiers .joblib et .json.
        """
        # Charger les fichiers joblib
        print("start charger___________________________________________")
        if((os.path.exists(os.path.join(dossier, "knn.joblib")) and os.path.exists(os.path.join(dossier, "scaler.joblib")) and os.path.exists(os.path.join(dossier, "label_encoder.joblib")) and os.path.exists(os.path.join(dossier, "metadonnees.json")))==False):
            raise ValueError("file not found")
        self._knn = joblib.load(os.path.join(dossier, "knn.joblib"))
        self._scaler = joblib.load(os.path.join(dossier, "scaler.joblib"))
        self._le = joblib.load(os.path.join(dossier, "label_encoder.joblib"))

        # Charger les métadonnées (pour _liste_airlines et _features)
        with open(os.path.join(dossier, "metadonnees.json"), "r", encoding="utf-8") as f:
            metadonnees = json.load(f)
        self.donnees.charger_airlines(metadonnees["airlines_possibles"])
        self._features = metadonnees["features"]



    # ---------------------------------------------------------
    # Sauvegarde
    # ---------------------------------------------------------
    def sauvegarder(self, scores: dict, dossier: str = "modeles") -> None:
        """Sauvegarde le modèle (joblib) et les métadonnées (JSON)."""
        if self._knn is None or self._scaler is None or self._le is None:
            raise RuntimeError("Le modèle doit être entraîné avant d'être sauvegardé.")

        os.makedirs(dossier, exist_ok=True)
        joblib.dump(self._knn, os.path.join(dossier, "knn.joblib"))
        joblib.dump(self._scaler, os.path.join(dossier, "scaler.joblib"))
        joblib.dump(self._le, os.path.join(dossier, "label_encoder.joblib"))

        metadonnees = {
            "features": self._features,
            "cible": "Delay",
            "airlines_possibles":self.donnees.airlines,
            "dataset":self.donnees.data,
            "intervalle_dayofweek": [1, 7],
            "intervalle_time": [0, 1439],
            "intervalle_length": [0, 655],
            "exactitude_knn": scores["exactitude_knn"],
            "exactitude_reference": scores["exactitude_reference"],
        }
        with open(os.path.join(dossier, "metadonnees.json"), "w", encoding="utf-8") as f:
            json.dump(metadonnees, f, ensure_ascii=False, indent=2)

        print(f"--- Modèle et métadonnées sauvegardés dans '{dossier}/' ---")


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
