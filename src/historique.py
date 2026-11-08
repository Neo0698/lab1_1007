import csv
import os
from datetime import datetime

import pandas as pd


class Historique:
    COLONNES = ["date", "compagnie", "aeroport", "jour", "heure", "duree_min", "proba_retard_pct"]

    def __init__(self, chemin: str = os.path.join("historique", "predictions.csv")) -> None:
        self._chemin = chemin

    def ajouter(self, compagnie: str, aeroport: str, jour: int, heure: int, minutes: int,
                duree: float, proba_retard: float) -> None:
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

    def __len__(self) -> int:
        return len(self.lire())