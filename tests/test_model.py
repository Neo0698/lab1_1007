import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajoute src/ au path Python
sys.path.append(str(Path(__file__).parent.parent))

from src.model import ModeleVol, DonneeInvalideError, PoidsVariables, taux_retard_lisse


def fabriquer_vols(nb: int = 60) -> pd.DataFrame:
    """ Prend un argument n correspondant au nombre de faux vols créé pour ensuite pouvoir faire les tests. """
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
    """enregistre les vols fabriqué pour le test."""
    chemin = tmp_path / "Airlines.csv"
    fabriquer_vols().to_csv(chemin, index=False)
    return str(chemin)


@pytest.fixture(scope="module")
def modele_entraine(tmp_path_factory):
    """Entraîne un modèle une seule fois sur les vols pour les tests."""
    dossier = tmp_path_factory.mktemp("donnees")
    chemin = dossier / "Airlines.csv"
    fabriquer_vols().to_csv(chemin, index=False)
    modele = ModeleVol(str(chemin))
    scores = modele.entrainer()
    return modele, scores


def test_charger_donnees_fichier_introuvable():
    """Vérifie qu'une erreur FileNotFoundError est levée si le CSV n'existe pas."""
    modele = ModeleVol("donnees/airlines2.csv")
    with pytest.raises(FileNotFoundError):
        modele.donnees.charger_donnees("donnees/airlines2.csv")


def test_charger_donnees_ok(chemin_csv):
    """Vérifie que le chargement d'un CSV valide récupère bien toutes les lignes et colonnes."""
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.donnees.data
    assert len(df) == 60
    for col in ("Airline", "Origin", "DayOfWeek", "Time", "Length", "Delay"):
        assert col in df.columns


def test_preparer_supprime_les_lignes_incompletes(chemin_csv):
    """Vérifie que l'ajout d'une ligne vide est bien nettoyé par preparer()."""
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.donnees.data.copy()
    df.loc[len(df)] = [None] * len(df.columns)
    df_propre = modele.preparer(df)
    assert len(df_propre) == 60
    assert df_propre.isna().sum().sum() == 0


def test_preparer_cree_encodage_et_length_carre(chemin_csv):
    """Vérifie le calcul de Length_Carre, l'encodage des compagnies et liste_airlines."""
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.preparer(modele.donnees.data)  
    assert (df["Length_Carre"] == df["Length"] ** 2).all()
    assert set(df["Airline_Encoded"].unique()) == {0, 1, 2}
    assert modele.donnees.liste_airlines == ["AA", "DL", "UA"]  # liste_airlines


def test_preparer_ne_modifie_pas_le_dataframe_original(chemin_csv):
    """Vérifie que preparer() travaille sur une copie sans modifier le DataFrame d'origine."""
    modele = ModeleVol(chemin_csv)
    modele.donnees.charger_donnees(chemin_csv)
    df = modele.donnees.data.copy() 
    modele.preparer(df)
    assert "Airline_Encoded" not in df.columns
    assert "Length_Carre" not in df.columns

def test_entrainer_retourne_deux_scores_valides(modele_entraine):
    """Vérifie qu'entrainer() retourne deux exactitudes comprises entre 0 et 1."""
    _, scores = modele_entraine
    assert set(scores) == {"exactitude_knn", "exactitude_reference"}
    for score in scores.values():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

def test_entrainer_knn_fait_mieux_que_la_reference(modele_entraine):
    """Vérifie que le modèle k-NN obtient un meilleur score que la référence."""
    _, scores = modele_entraine
   
    assert scores["exactitude_knn"] >= scores["exactitude_reference"]


def test_predire_avant_entrainement_leve_runtime_error(chemin_csv):
    """Vérifie qu'on ne peut pas prédire avec un modèle non entraîné."""
    modele = ModeleVol(chemin_csv)
    with pytest.raises(RuntimeError):
        modele.predire("AA", "JFK", 3, 10, 30, 200)

def test_predire_compagnie_inconnue(modele_entraine):
    """Vérifie qu'une compagnie absente des données d'entraînement est refusée."""
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("ZZ", "JFK", 3, 10, 30, 200)

@pytest.mark.parametrize("jour", [0, 8])
def test_predire_jour_hors_intervalle(modele_entraine, jour):
    """Vérifie qu'un jour hors de l'intervalle 1 à 7 lève DonneeInvalideError."""
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", "JFK", jour, 10, 30, 200)

@pytest.mark.parametrize("heure,minutes", [(24, 0), (10, 60), (-1, 30)])

def test_predire_heure_invalide(modele_entraine, heure, minutes):
    """Vérifie qu'une heure ou des minutes invalides lèvent DonneeInvalideError."""
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", "JFK", 3, heure, minutes, 200)


@pytest.mark.parametrize("duree", [0, -10])
def test_predire_duree_non_positive(modele_entraine, duree):
    """Vérifie qu'une durée nulle ou négative lève DonneeInvalideError."""
    modele, _ = modele_entraine
    with pytest.raises(DonneeInvalideError):
        modele.predire("AA", "JFK", 3, 10, 30, duree)


def test_predire_retourne_deux_probabilites(modele_entraine):
    """Vérifie que predire() renvoie deux probabilités dont la somme vaut 1."""
    modele, _ = modele_entraine
    probas = modele.predire("AA", "JFK", 3, 10, 30, 200)
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (2,)
    assert all(0.0 <= p <= 1.0 for p in probas)
    
    assert float(probas.sum()) == pytest.approx(1.0)


def test_predire_vol_long_plus_a_risque_que_vol_court(modele_entraine):
    """Vérifie qu'un vol long a une probabilité de retard plus élevée sur le jeu de test."""
    modele, _ = modele_entraine
    p_retard_court = modele.predire("AA", "JFK", 3, 10, 30, 80)[1]
    p_retard_long = modele.predire("AA", "JFK", 3, 10, 30, 590)[1]
    assert p_retard_long > p_retard_court


@pytest.mark.parametrize("jour,heure,minutes", [(1, 0, 0), (7, 23, 59)])
def test_predire_cas_limites_acceptes(modele_entraine, jour, heure, minutes):
    """Vérifie que les bornes exactes (lundi 00h00 et dimanche 23h59) sont acceptées."""
    modele, _ = modele_entraine
    probas = modele.predire("DL", "JFK", jour, heure, minutes, 120)
    assert probas.shape == (2,)


def test_sauvegarder_avant_entrainement_leve_runtime_error(chemin_csv, tmp_path):
    """Vérifie qu'un modèle non entraîné ne peut pas être sauvegardé."""
    modele = ModeleVol(chemin_csv)
    scores = {"exactitude_knn": 0.0, "exactitude_reference": 0.0}
    with pytest.raises(RuntimeError):
        modele.sauvegarder(scores, dossier=str(tmp_path / "modeles"))


def test_sauvegarder_cree_fichiers_et_metadonnees(modele_entraine, tmp_path):
    """Vérifie que sauvegarder() écrit bien les fichiers .joblib et metadonnees.json."""
    modele, scores = modele_entraine
    dossier = tmp_path / "modeles"
    modele.sauvegarder(scores, dossier=str(dossier))

    for nom in ("knn.joblib", "scaler.joblib", "label_encoder.joblib", "metadonnees.json"):
        assert os.path.exists(dossier / nom)

    with open(dossier / "metadonnees.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["cible"] == "Delay"
    assert meta["features"] == ["Airline_Encoded", "Origin_Taux", "DayOfWeek", "Time",
                                 "Length", "Length_Carre"]
    assert meta["aeroports_possibles"] == ["ATL", "JFK", "LAX", "ORD"]
    assert meta["airlines_possibles"] == modele.donnees.liste_airlines
    assert meta["intervalle_dayofweek"] == [1, 7]
    assert meta["intervalle_time"] == [0, 1439]
    durees = fabriquer_vols()["Length"]
    assert meta["intervalle_length"] == [durees.min(), durees.max()]
    assert meta["exactitude_knn"] == pytest.approx(scores["exactitude_knn"])
    assert meta["exactitude_reference"] == pytest.approx(scores["exactitude_reference"])


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


def test_poids_negatif_leve_value_error():
    """Un poids négatif n'a pas de sens : il doit être refusé."""
    with pytest.raises(ValueError):
        PoidsVariables(k_jour=-1)


def test_poids_appliquer_multiplie_chaque_colonne():
    """Vérifie que chaque colonne normalisée est multipliée par le poids correspondant."""
    poids = PoidsVariables(k_duree=2, k_jour=0)
    X = np.ones((3, 3))
    resultat = poids.appliquer(X, ["DayOfWeek", "Length", "Length_Carre"])
    assert (resultat == np.array([[0.0, 2.0, 2.0]] * 3)).all()


def test_poids_nul_sur_la_duree_rend_la_duree_sans_effet(chemin_csv):
    """Avec k_duree=0, un vol court et un vol long ont exactement la même prédiction."""
    modele = ModeleVol(chemin_csv, poids=PoidsVariables(k_duree=0))
    modele.entrainer()
    court = modele.predire("AA", "JFK", 3, 10, 30, 80)
    long = modele.predire("AA", "JFK", 3, 10, 30, 590)
    assert court == pytest.approx(long)


def test_poids_sauvegardes_et_recharges(chemin_csv, tmp_path):
    """Les poids sont écrits dans le JSON et relus par charger() : mêmes prédictions après rechargement."""
    poids = PoidsVariables(k_heure=2, k_jour=0.5)
    modele = ModeleVol(chemin_csv, poids=poids)
    dossier = str(tmp_path / "modeles")
    modele.sauvegarder(modele.entrainer(), dossier=dossier)

    recharge = ModeleVol(chemin_csv)
    recharge.charger(dossier)
    assert recharge.poids.en_dict() == poids.en_dict()
    assert recharge.predire("DL", "LAX", 5, 18, 0, 300) == pytest.approx(modele.predire("DL", "LAX", 5, 18, 0, 300))


def test_data_en_dict_et_restaurer():
    """La partie Data des métadonnées se relit à l'identique : colonnes, compagnies, aéroports."""
    from src.Data import Data
    source = Data()
    source.charger_airlines(["AA", "DL"])
    source.charger_aeroports(["JFK", "LAX"])
    meta = source.en_dict()

    copie = Data()
    copie.restaurer(meta)
    assert copie.features == source.features
    assert copie.liste_airlines == ["AA", "DL"]
    assert copie.liste_aeroports == ["JFK", "LAX"]


def test_charger_retrouve_les_memes_voisins(chemin_csv, tmp_path):
    """Après charger(), l'encodeur n'est pas réappris et les vols d'entraînement sont les mêmes :
    les voisins (et leurs distances) sont identiques à ceux du modèle entraîné."""
    modele = ModeleVol(chemin_csv)
    dossier = str(tmp_path / "modeles")
    modele.sauvegarder(modele.entrainer(), dossier=dossier)

    recharge = ModeleVol(chemin_csv)
    recharge.charger(dossier)
    avant = modele.voisins("UA", "ORD", 2, 9, 15, 250)
    apres = recharge.voisins("UA", "ORD", 2, 9, 15, 250)
    pd.testing.assert_frame_equal(avant, apres)
    assert recharge.donnees.liste_airlines == modele.donnees.liste_airlines
