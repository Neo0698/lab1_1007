"""Point d'entrée de l'application : charge le modèle et lance l'interface Gradio.

Lancement (depuis la racine du projet) : python -m src.main
"""

import os

from .historique import Historique
from .interface import Interface
from .model import ModeleVol

CHEMIN_CSV = os.path.join("donnees", "Airlines.csv")
DOSSIER_MODELES = "modeles"


def preparer_modele(chemin_csv: str = CHEMIN_CSV, dossier: str = DOSSIER_MODELES) -> ModeleVol:
    """Charge le modèle sauvegardé ; s'il est absent, l'entraîne et le sauvegarde une fois.

    Args:
        chemin_csv: Chemin du jeu de données.
        dossier: Dossier du modèle sauvegardé (joblib + JSON).

    Returns:
        Un modèle prêt à prédire.

    Raises:
        FileNotFoundError: si le jeu de données lui-même est introuvable.
    """
    modele = ModeleVol(chemin_csv)
    try:
        modele.charger(dossier)
    except FileNotFoundError as e:
        print(f"{e} : entraînement du modèle…")
        modele.sauvegarder(modele.entrainer(), dossier=dossier)
        modele.charger(dossier)
    return modele


def main() -> None:
    """Prépare le modèle puis lance l'interface graphique."""
    try:
        modele = preparer_modele()
    except FileNotFoundError as e:
        print(f"Erreur : {e}\nGénérez d'abord les données avec : python -m src.create_dataset")
        return
    Interface(modele, Historique()).construire_interface().launch()


if __name__ == "__main__":
    main()
