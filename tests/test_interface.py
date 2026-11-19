from unittest.mock import MagicMock

import pandas as pd

from src.historique import Historique
from src.interface import Interface
from src.model import ModeleVol


def test_predire_affiche_le_graphique_et_enregistre_historique():
    modele = MagicMock(spec=ModeleVol)
    modele.metadonnees = {
        "airlines_possibles": ["AA"],
        "aeroports_possibles": ["JFK"],
        "intervalle_dayofweek": [1, 7],
        "intervalle_length": [0, 655],
    }
    modele.predire.return_value = [0.25, 0.75]
    voisins = pd.DataFrame({"Delay": [0, 1], "Time": [600, 660], "Length": [100, 200]})
    modele.voisins.return_value = voisins
    historique = MagicMock(spec=Historique)

    html, figure, tableau = Interface(modele, historique).predire("AA", "JFK", 3, 10, 30, 150)

    assert "À l'heure : 25.0%" in html
    assert "Retardé : 75.0%" in html
    assert tableau is voisins
    assert len(figure.axes[0].collections) == 3
    modele.predire.assert_called_once_with("AA", "JFK", 3, 10, 30, 150.0)
    modele.voisins.assert_called_once_with("AA", "JFK", 3, 10, 30, 150.0)
    historique.ajouter.assert_called_once_with("AA", "JFK", 3.0, 10.0, 30.0, 150.0, 0.75)
