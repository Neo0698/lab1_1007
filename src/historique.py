"""Historique des prédictions, enregistré dans un fichier CSV."""

import csv
import os
from datetime import datetime

import pandas as pd


class Historique:
    """Ajoute chaque prédiction à un fichier CSV et permet de relire l'historique.

    Args:
        chemin: Chemin du fichier CSV (créé au premier ajout s'il n'existe pas).
    """

    COLONNES = ["date", "compagnie", "aeroport", "jour", "heure", "duree_min", "proba_retard_pct"]

    def __init__(self, chemin: str = os.path.join("historique", "predictions.csv")) -> None:
        """Garde le chemin du fichier ; aucun fichier n'est créé à cette étape."""
        self._chemin = chemin

    def ajouter(self, compagnie: str, aeroport: str, jour: int, heure: int, minutes: int,
                duree: float, proba_retard: float) -> None:
        """Ajoute une ligne (date, saisies et probabilité de retard en %) au CSV.

        Crée le dossier et l'en-tête du fichier s'ils n'existent pas encore.

        Args:
            compagnie: Code de la compagnie.
            aeroport: Code de l'aéroport de départ.
            jour: Jour de la semaine (1 = lundi).
            heure: Heure de départ (0 à 23).
            minutes: Minutes de départ (0 à 59).
            duree: Durée du vol en minutes.
            proba_retard: Probabilité de retard, entre 0 et 1.

        erreur:
            OSError: si le fichier ne peut pas être écrit.
        """
        dossier = os.path.dirname(self._chemin)
        if dossier:
            os.makedirs(dossier, exist_ok=True)
        nouveau = not os.path.exists(self._chemin)
        with open(self._chemin, "a", newline="", encoding="utf-8") as f:
            ecrivain = csv.DictWriter(f, fieldnames=self.COLONNES)
            if nouveau:
                ecrivain.writeheader()
            ecrivain.writerow({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "compagnie": compagnie, "aeroport": aeroport, "jour": int(jour),
                "heure": f"{int(heure):02d}h{int(minutes):02d}",
                "duree_min": duree, "proba_retard_pct": round(proba_retard * 100, 2),
            })

    def lire(self) -> pd.DataFrame:
        """Renvoie l'historique (vide s'il n'existe pas encore)."""
        if not os.path.exists(self._chemin):
            return pd.DataFrame(columns=self.COLONNES)
        return pd.read_csv(self._chemin)
