import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajoute src/ au path Python
sys.path.append(str(Path(__file__).parent.parent))

from src.model import ModeleVol, DonneeInvalideError, taux_retard_lisse

def fabriquer_vols(nb: int = 60) -> pd.DataFrame:
    """
    Prend un argument n correspondant au nombre de faux vols créé pour ensuite pouvoir faire les tests
    """
    lignes = []
    for i in range(nb):
        length = 60 + ((i * 53) % 540)          
        lignes.append({
            "Airline": ["AA", "DL", "UA"][i % 3],
            "Origin": ["JFK", "LAX", "ORD", "ATL"][i % 4],
            "DayOfWeek": (i % 7) + 1,          
            "Time": (i * 47) % 1440,           
            "Length": length,
            "Delay": 1 if length > 300 else 0,
        })
    return pd.DataFrame(lignes)

@pytest.fixture()
def chemin_csv(tmp_path) -> str:
    """enregistre les vols fabriqué pour le test"""
    chemin = tmp_path / "Airlines.csv"
    fabriquer_vols().to_csv(chemin, index=False)
    return str(chemin)

@pytest.fixture(scope="module")
def modele_entraine(tmp_path_factory):
    """Un modèle entraîné une seule fois pour tous les tests de prédiction, avec le corpus de fabriquer_vols"""
    dossier = tmp_path_factory.mktemp("donnees")
    chemin = dossier / "Airlines.csv"
    fabriquer_vols().to_csv(chemin, index=False)
    modele = ModeleVol(str(chemin))
    scores = modele.entrainer()
    return modele, scores


def test_charger_donnees_fichier_introuvable():
    """1er test, lancer le modèle sur un fichier qui n'existe pas pour vérifier si une erreur est bien généré par model.py """
    modele = ModeleVol("donnees/airlines2.csv")
    with pytest.raises(FileNotFoundError):
        modele.donnees.charger_donnees("donnees/airlines2.csv")

def test_charger_donnees_ok(chemin_csv):
    """2ième test, lancer le modèle cette fois ci avec le bon chemin d'acces pour tester s'il est capable d'importer ce fichier sans erreur"""
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.donnees.data 
    assert len(df) == 60
    assert {"Airline", "Origin", "DayOfWeek", "Time", "Length", "Delay"} <= set(df.columns)


def test_preparer_supprime_les_lignes_incompletes(chemin_csv):
    """3ième test, verification que l'ajout d'une ligne vide n'empeche pas l'entrainement du modèle"""
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.donnees.data.copy()  # ← Copie pour éviter de modifier l'original
    df.loc[len(df)] = [None] * len(df.columns)   # ligne vide
    df_propre = modele.preparer(df)
    assert len(df_propre) == 60
    assert df_propre.isna().sum().sum() == 0

def test_preparer_cree_encodage_et_length_carre(chemin_csv):
    """4ième test, verification qu'à la suite de modele.preparer, que  Length_Carre a été calculé correctement, que le nom des compagnie on été transformé correctement
    en vecteurs pour qu'ils puissent être interprété par le KNN., liste airlines contiennent bien l'ensemble des compagnies présente dans le dataset"""
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.preparer(modele.donnees.data)  
    assert (df["Length_Carre"] == df["Length"] ** 2).all()
    assert set(df["Airline_Encoded"].unique()) == {0, 1, 2}
    assert modele.donnees.liste_airlines == ["AA", "DL", "UA"]  # liste_airlines

def test_preparer_ne_modifie_pas_le_dataframe_original(chemin_csv):
    """
    5ème test, vérifie si après l'entrainement le dataset chargé ne change pas, manque pas une colonne ou ajout d'une nouvelle colonne
    """
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.donnees.data.copy() 
    modele.preparer(df)
    assert "Airline_Encoded" not in df.columns
    assert "Length_Carre" not in df.columns

def test_entrainer_retourne_deux_scores_valides(modele_entraine):
    """
    6ème test, vérification que la fonction retourne bien deux valeurs exactitude_knn et exactitude_reference
    """
    _, scores = modele_entraine
    assert set(scores) == {"exactitude_knn", "exactitude_reference"}
    for score in scores.values():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

def test_entrainer_knn_fait_mieux_que_la_reference(modele_entraine):
    """
    7ème test, verification que le modèle knn fasse mieux que le modèle de référence.
    """
    _, scores = modele_entraine
   
    assert scores["exactitude_knn"] >= scores["exactitude_reference"]


def test_predire_avant_entrainement_leve_runtime_error(chemin_csv):
    """
    8ème test, verification qu'on ne puisse pas prédire sur un modèle non entrainé
    """
    modele = ModeleVol(chemin_csv) 
    with pytest.raises(RuntimeError):
        modele.predire("AA", "JFK", 3, 10, 30, 200)

def test_predire_compagnie_inconnue(modele_entraine):
    """
    9ème test, qu'on ne puisse pas prédire sur une compagnie inconnue
    """
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("ZZ", "JFK", 3, 10, 30, 200)

@pytest.mark.parametrize("jour", [0, 8])
def test_predire_jour_hors_intervalle(modele_entraine, jour):
    """
    test pour les valeurs 0,8 si la prediction renvoi bien une erreur
    """
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", "JFK", jour, 10, 30, 200)

@pytest.mark.parametrize("heure,minutes", [(24, 0), (10, 60), (-1, 30)])

def test_predire_heure_invalide(modele_entraine, heure, minutes):
    """
    test pour les valeurs (heure, minutes) suivante (24, 0), (10, 60), (-1, 30) qu'une erreur est bien produite
    """
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", "JFK", 3, heure, minutes, 200)

@pytest.mark.parametrize("duree", [0, -10])
def test_predire_duree_non_positive(modele_entraine, duree):
    """
    test qu'une duree de vol non possible renvoi bien une erreur, testé avec les valeur temps nulle et un temps négatif
    """
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", "JFK", 3, 10, 30, duree)


def test_predire_retourne_deux_probabilites(modele_entraine):
    """
    test si la fonction prédire renvoi bien la probabilité que le vol soit a l'heure et que le vol soit en retard et que les deux font bien 100%
    """
    modele, _ = modele_entraine
    probas = modele.predire("AA", "JFK", 3, 10, 30, 200)
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (2,)
    assert all(0.0 <= p <= 1.0 for p in probas)
    
    assert float(probas.sum()) == pytest.approx(1.0)

def test_predire_vol_long_plus_a_risque_que_vol_court(modele_entraine):
    """
    test si un vol long est plus en retard (car hypothèse créé dans le dataset de test avec 1 if length >300)
    """
    modele, _ = modele_entraine
    p_retard_court = modele.predire("AA", "JFK", 3, 10, 30, 80)[1]
    p_retard_long = modele.predire("AA", "JFK", 3, 10, 30, 590)[1]
    assert p_retard_long > p_retard_court

@pytest.mark.parametrize("jour,heure,minutes", [(1, 0, 0), (7, 23, 59)])
def test_predire_cas_limites_acceptes(modele_entraine, jour, heure, minutes):
    """
    verifie que les cas limites renvoi bien les deux prédictions
    """
    modele, _ = modele_entraine
    probas = modele.predire("DL", "JFK", jour, heure, minutes, 120)
    assert probas.shape == (2,)


def test_sauvegarder_avant_entrainement_leve_runtime_error(chemin_csv, tmp_path):
    """
    verifie qu'un modele non entrainné ne peut pas être entrainné
    """
    modele = ModeleVol(chemin_csv) 
    scores = {"exactitude_knn": 0.0, "exactitude_reference": 0.0}
    with pytest.raises(RuntimeError):
        modele.sauvegarder(scores, dossier=str(tmp_path / "modeles"))

def test_sauvegarder_cree_fichiers_et_metadonnees(modele_entraine, tmp_path):
    """
    verifie qu'un modèle entrainné peut être sauvegardé et que cette action est réalisé correctement en sauvegardant toutes les données nécessaires
    """
    modele, scores = modele_entraine
    dossier = tmp_path / "modeles"
    modele.sauvegarder(scores, dossier=str(dossier))

    for nom in ("knn.joblib", "scaler.joblib",
                "label_encoder.joblib", "metadonnees.json"):
        assert os.path.exists(dossier / nom)

    with open(dossier / "metadonnees.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["cible"] == "Delay"
    assert meta["features"] == ["Airline_Encoded", "Origin_Taux", "DayOfWeek", "Time",
                                 "Length", "Length_Carre"]
    assert meta["aeroports_possibles"] == ["ATL", "JFK", "LAX", "ORD"]
    assert meta["airlines_possibles"] == modele.donnees.liste_airlines  # ← liste_airlines
    assert meta["intervalle_dayofweek"] == [1, 7]
    assert meta["intervalle_time"] == [0, 1439]
    assert meta["intervalle_length"] == [0, 655]
    assert meta["exactitude_knn"] == pytest.approx(scores["exactitude_knn"])
    assert meta["exactitude_reference"] == pytest.approx(
        scores["exactitude_reference"])


def test_predire_aeroport_inconnu(modele_entraine):
    """L'aéroport de départ doit faire partie de ceux vus à l'entraînement."""
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", "ZZZ", 3, 10, 30, 200)


def test_taux_retard_lisse_tire_les_petits_aeroports_vers_le_taux_global():
    """Un grand aéroport garde presque son taux ; un petit reste proche du taux global."""
    origines = pd.Series(["GRAND"] * 1000 + ["PETIT"] * 2)
    retards = pd.Series([1] * 800 + [0] * 200 + [1, 1])
    taux = taux_retard_lisse(origines, retards, poids=20)
    taux_global = retards.mean()
    assert taux["GRAND"] == pytest.approx((800 + 20 * taux_global) / 1020)
    assert taux["PETIT"] == pytest.approx((2 + 20 * taux_global) / 22)
    assert abs(taux["PETIT"] - taux_global) < abs(1.0 - taux_global)
