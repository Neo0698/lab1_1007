import modeles.avion as mod
modele = mod.ModeleVol("donnees/Airlines.csv")
modele.entrainer()
print("modele._liste_airlines",modele._liste_airlines)
print("_________________________")
import os

import gradio as gr
import joblib
import pandas as pd

CHEMIN_MODELE = os.path.join("modeles", "modele_knn.joblib")
CHEMIN_CSV = "airlines.csv"   # sert seulement à lister les compagnies
JOURS = [("1 - Lundi", 1), ("2 - Mardi", 2), ("3 - Mercredi", 3), ("4 - Jeudi", 4),
         ("5 - Vendredi", 5), ("6 - Samedi", 6), ("7 - Dimanche", 7)]

CSS = """
<style>
.resultat { margin-top: 10px; padding: 15px; border-radius: 4px; text-align: center;
            font-weight: bold; font-size: 15px; }
.success { background-color: #e8f5e9; color: #2e7d32; }
.danger  { background-color: #ffebee; color: #c62828; }
.error   { background-color: #fff3e0; color: #ef6c00; }
</style>
"""


class DonneeInvalideError(Exception):
    """Levée quand une saisie de l'utilisateur est vide ou hors intervalle."""


def charger_modele():
    """Charge le modèle sauvegardé ; lève FileNotFoundError s'il est absent."""
    if not os.path.exists(CHEMIN_MODELE):
        raise FileNotFoundError(f"Modèle introuvable : {CHEMIN_MODELE}")
    return joblib.load(CHEMIN_MODELE)


def lister_compagnies() -> list[str]:
    """Renvoie la liste triée des compagnies présentes dans le CSV."""
    return modele._liste_airlines


def valider(compagnie, jour, heure, minutes, duree) -> tuple[str, int, int, int]:
    """Vérifie les saisies et renvoie (compagnie, jour, temps en minutes, durée).

    Lève DonneeInvalideError avec un message clair si une saisie est vide ou
    hors intervalle.
    """
    if not compagnie:
        raise DonneeInvalideError("Veuillez choisir une compagnie aérienne.")
    for nom, valeur in (("jour", jour), ("heure", heure), ("minutes", minutes),
                        ("durée", duree)):
        if valeur is None:
            raise DonneeInvalideError(f"Le champ « {nom} » est vide.")
    if not 0 <= heure <= 23:
        raise DonneeInvalideError("L'heure doit être comprise entre 0 et 23.")
    if not 0 <= minutes <= 59:
        raise DonneeInvalideError("Les minutes doivent être comprises entre 0 et 59.")
    if not 1 <= duree <= 1500:
        raise DonneeInvalideError("La durée doit être comprise entre 1 et 1500 minutes.")
    return compagnie, int(jour), int(heure) * 60 + int(minutes), int(duree)


def calculer_probabilites(modele, compagnie: str, jour: int, temps: int, duree: int
                          ) -> tuple[float, float]:
    """Renvoie (% à l'heure, % retardé) pour un vol, arrondis à 0,1 %."""
    ligne = pd.DataFrame([{"Airline": compagnie, "DayOfWeek": jour,
                           "Time": temps, "Length": duree}])
    probas = modele.predict_proba(ligne)[0]
    p_retard = probas[list(modele.classes_).index(1)] * 100
    return round(100 - p_retard, 1), round(p_retard, 1)


def predire(compagnie, jour, heure, minutes, duree) -> str:
    """Fonction appelée par le bouton : renvoie le bloc HTML du résultat."""
    try:
        saisies = valider(compagnie, jour, heure, minutes, duree)
        a_l_heure, retarde = calculer_probabilites(charger_modele(), *saisies)
    except DonneeInvalideError as e:
        return f'<div class="resultat error">{e}</div>'
    except FileNotFoundError as e:
        return f'<div class="resultat error">{e}</div>'
    classe = "danger" if retarde > 50 else "success"
    return (f'<div class="resultat {classe}">📊 Résultats des probabilités :<br><br>'
            f"✅ À l'heure : {a_l_heure}%<br>🚨 Retardé : {retarde}%</div>")


def construire_interface() -> gr.Blocks:
    """Construit et renvoie l'interface Gradio."""
    compagnies = lister_compagnies()
    with gr.Blocks(title="Prédiction de Retard de Vol (k-NN)") as demo:
        gr.HTML(CSS)
        gr.Markdown("## ✈️ Prédiction de Vol (k-NN)")
        compagnie = gr.Dropdown(compagnies, label="Compagnie aérienne",
                                value=compagnies[0] if compagnies else None)
        jour = gr.Dropdown(JOURS, label="Jour de la semaine (1 à 7)", value=1)
        heure = gr.Number(label="Heure de départ (0-23)", value=14, precision=0,
                          minimum=0, maximum=23)
        minutes = gr.Number(label="Minutes de départ (0-59)", value=0, precision=0,
                            minimum=0, maximum=59)
        duree = gr.Number(label="Durée du vol en minutes (ex: 495)", value=495,
                          precision=0, minimum=1)
        bouton = gr.Button("Calculer le risque de retard", variant="primary")
        sortie = gr.HTML()
        bouton.click(predire, [compagnie, jour, heure, minutes, duree], sortie)
    return demo


if __name__ == "__main__":
    construire_interface().launch()