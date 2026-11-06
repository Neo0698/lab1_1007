import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajoute src/ au path Python
sys.path.append(str(Path(__file__).parent.parent))

from src.model import ModeleVol, DonneeInvalideError

def fabriquer_vols(nb: int = 60) -> pd.DataFrame:
    lignes = []
    for i in range(nb):
        length = 60 + ((i * 53) % 540)          # entre 60 et 599 minutes
        lignes.append({
            "Airline": ["AA", "DL", "UA"][i % 3],
            "DayOfWeek": (i % 7) + 1,           # 1 à 7
            "Time": (i * 47) % 1440,            # minutes depuis minuit
            "Length": length,
            "Delay": 1 if length > 300 else 0,
        })
    return pd.DataFrame(lignes)

@pytest.fixture()
def chemin_csv(tmp_path) -> str:
    chemin = tmp_path / "Airlines.csv"
    fabriquer_vols().to_csv(chemin, index=False)
    return str(chemin)

@pytest.fixture(scope="module")
def modele_entraine(tmp_path_factory):
    """Un modèle entraîné une seule fois pour tous les tests de prédiction."""
    dossier = tmp_path_factory.mktemp("donnees")
    chemin = dossier / "Airlines.csv"
    fabriquer_vols().to_csv(chemin, index=False)
    modele = ModeleVol(str(chemin))
    scores = modele.entrainer()
    return modele, scores

# ----------------------------------------------------------------------
# charger_donnees
# ----------------------------------------------------------------------
def test_charger_donnees_fichier_introuvable():
    modele = ModeleVol("donnees/fichier_inexistant.csv")
    with pytest.raises(FileNotFoundError):
        modele.donnees.charger_donnees("donnees/fichier_inexistant.csv")

def test_charger_donnees_ok(chemin_csv):
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)  # ← Charger les données
    df = modele.donnees.data  # ← Accéder via donnees.data
    assert len(df) == 60
    assert {"Airline", "DayOfWeek", "Time", "Length", "Delay"} <= set(df.columns)

# ----------------------------------------------------------------------
# preparer
# ----------------------------------------------------------------------
def test_preparer_supprime_les_lignes_incompletes(chemin_csv):
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.donnees.data.copy()  # ← Copie pour éviter de modifier l'original
    df.loc[len(df)] = [None, None, None, None, None]   # ligne vide
    df_propre = modele.preparer(df)
    assert len(df_propre) == 60
    assert df_propre.isna().sum().sum() == 0

def test_preparer_cree_encodage_et_length_carre(chemin_csv):
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.preparer(modele.donnees.data)  # ← Utilise donnees.data
    # La nouvelle caractéristique est bien le carré de Length
    assert (df["Length_Carre"] == df["Length"] ** 2).all()
    # L'encodage donne un entier par compagnie (3 compagnies -> 0, 1, 2)
    assert set(df["Airline_Encoded"].unique()) == {0, 1, 2}
    # La liste des compagnies est triée
    assert modele.donnees.liste_airlines == ["AA", "DL", "UA"]  # ← liste_airlines

def test_preparer_ne_modifie_pas_le_dataframe_original(chemin_csv):
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.donnees.data.copy()  # ← Copie
    modele.preparer(df)
    assert "Airline_Encoded" not in df.columns
    assert "Length_Carre" not in df.columns

# ----------------------------------------------------------------------
# entrainer
# ----------------------------------------------------------------------
def test_entrainer_retourne_deux_scores_valides(modele_entraine):
    _, scores = modele_entraine
    assert set(scores) == {"exactitude_knn", "exactitude_reference"}
    for score in scores.values():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

def test_entrainer_knn_fait_mieux_que_la_reference(modele_entraine):
    _, scores = modele_entraine
    # Sur ces données, le retard dépend de Length : le kNN doit
    # battre la classe majoritaire, qui ne regarde rien.
    assert scores["exactitude_knn"] >= scores["exactitude_reference"]

# ----------------------------------------------------------------------
# predire : cas d'erreur avant les calculs
# ----------------------------------------------------------------------
def test_predire_avant_entrainement_leve_runtime_error(chemin_csv):
    modele = ModeleVol(chemin_csv)   # jamais entraîné
    with pytest.raises(RuntimeError):
        modele.predire("AA", 3, 10, 30, 200)

def test_predire_compagnie_inconnue(modele_entraine):
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("ZZ", 3, 10, 30, 200)

@pytest.mark.parametrize("jour", [0, 8])
def test_predire_jour_hors_intervalle(modele_entraine, jour):
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", jour, 10, 30, 200)

@pytest.mark.parametrize("heure,minutes", [(24, 0), (10, 60), (-1, 30)])
def test_predire_heure_invalide(modele_entraine, heure, minutes):
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", 3, heure, minutes, 200)

@pytest.mark.parametrize("duree", [0, -10])
def test_predire_duree_non_positive(modele_entraine, duree):
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", 3, 10, 30, duree)

# ----------------------------------------------------------------------
# predire : cas usuels et cas limites
# ----------------------------------------------------------------------
def test_predire_retourne_deux_probabilites(modele_entraine):
    modele, _ = modele_entraine
    probas = modele.predire("AA", 3, 10, 30, 200)
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (2,)
    assert all(0.0 <= p <= 1.0 for p in probas)
    # Les deux probabilités sont complémentaires
    assert float(probas.sum()) == pytest.approx(1.0)

def test_predire_vol_long_plus_a_risque_que_vol_court(modele_entraine):
    modele, _ = modele_entraine
    p_retard_court = modele.predire("AA", 3, 10, 30, 80)[1]
    p_retard_long = modele.predire("AA", 3, 10, 30, 590)[1]
    assert p_retard_long > p_retard_court

@pytest.mark.parametrize("jour,heure,minutes", [(1, 0, 0), (7, 23, 59)])
def test_predire_cas_limites_acceptes(modele_entraine, jour, heure, minutes):
    modele, _ = modele_entraine
    probas = modele.predire("DL", jour, heure, minutes, 120)
    assert probas.shape == (2,)

# ----------------------------------------------------------------------
# sauvegarder
# ----------------------------------------------------------------------
def test_sauvegarder_avant_entrainement_leve_runtime_error(chemin_csv, tmp_path):
    modele = ModeleVol(chemin_csv)   # jamais entraîné
    scores = {"exactitude_knn": 0.0, "exactitude_reference": 0.0}
    with pytest.raises(RuntimeError):
        modele.sauvegarder(scores, dossier=str(tmp_path / "modeles"))

def test_sauvegarder_cree_fichiers_et_metadonnees(modele_entraine, tmp_path):
    modele, scores = modele_entraine
    dossier = tmp_path / "modeles"
    modele.sauvegarder(scores, dossier=str(dossier))

    for nom in ("knn.joblib", "scaler.joblib",
                "label_encoder.joblib", "metadonnees.json"):
        assert os.path.exists(dossier / nom)

    with open(dossier / "metadonnees.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["cible"] == "Delay"
    assert meta["features"] == ["Airline_Encoded", "DayOfWeek", "Time",
                                 "Length", "Length_Carre"]
    assert meta["airlines_possibles"] == modele.donnees.liste_airlines  # ← liste_airlines
    assert meta["intervalle_dayofweek"] == [1, 7]
    assert meta["intervalle_time"] == [0, 1439]
    assert meta["intervalle_length"] == [0, 655]
    assert meta["exactitude_knn"] == pytest.approx(scores["exactitude_knn"])
    assert meta["exactitude_reference"] == pytest.approx(
        scores["exactitude_reference"])
