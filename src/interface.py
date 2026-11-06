"""Interface Gradio : prédiction de retard de vol (k-NN)."""

import gradio as gr
from matplotlib.figure import Figure
import model as mod
import pandas as pd

CHEMIN_CSV = "donnees/Airlines.csv"
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
    pass

class Interface:  # ✅ Classe renommée en "Interface" (convention Python)
    def __init__(self):
        pass

    def valider(self, compagnie, jour, heure, minutes, duree) -> None:
        """Vérifie les saisies ; lève DonneeInvalideError avec un message clair."""
        if not compagnie:
            raise DonneeInvalideError("Veuillez choisir une compagnie aérienne.")
        for nom, valeur in (("jour", jour), ("heure", heure), ("minutes", minutes), ("durée", duree)):
            if valeur is None:
                raise DonneeInvalideError(f"Le champ « {nom} » est vide.")
        if not 0 <= heure <= 23:
            raise DonneeInvalideError("L'heure doit être comprise entre 0 et 23.")
        if not 0 <= minutes <= 59:
            raise DonneeInvalideError("Les minutes doivent être comprises entre 0 et 59.")
        if duree <= 0:
            raise DonneeInvalideError("La durée du vol doit être positive.")

    def creer_graphique_voisins(self, voisins_df: pd.DataFrame, a_l_heure: float, retarde: float) -> Figure:
        """Génère le graphique des k voisins les plus proches."""
        fig = Figure(figsize=(6, 4.2))
        ax = fig.subplots()
        for delay, couleur, nom in ((0, "#2e7d32", "À l'heure"), (1, "#c62828", "Retardé")):
            v = voisins_df[voisins_df["Delay"] == delay]
            ax.scatter(v["Time"] / 60, v["Length"], c=couleur, s=70, alpha=0.7,
                       edgecolors="black", label=f"{nom} ({len(v)})")
        ax.set_title(f"Les {len(voisins_df)} vols les plus proches")
        ax.set_xlabel("Heure de départ (h)")
        ax.set_ylabel("Durée du vol (min)")
        ax.legend()
        fig.tight_layout()
        return fig

    def creer_predicteur(self, modele):
        """
        fonction predire associé avec:
        bouton.click(
                self.creer_predicteur(modele),
                [compagnie, jour, heure, minutes, duree],
                [sortie, graphique, tableau]
            )
        une unique fonction doit être présente pour que la librairie sache quoi appeler lors que le boutton est cliqué
        """
        def predire(compagnie, jour, heure, minutes, duree):
            try:
                self.valider(compagnie, jour, heure, minutes, duree) #verification des données avec pytest

                prob = modele.predire(compagnie, float(jour), float(heure),
                                      float(minutes), float(duree))
               
                voisins = modele.voisins(compagnie, float(jour), float(heure),
                                         float(minutes), float(duree))
            except DonneeInvalideError as e:
                return f'<div class="resultat error">{e}</div>', None, None
            except (ValueError, KeyError) as e:
                print("Erreur :", e)
                return '<div class="resultat error">Veuillez entrer des valeurs valides.</div>', None, None

            a_l_heure, retarde = round(prob[0] * 100, 2), round(prob[1] * 100, 2)
            classe = "danger" if retarde > 50 else "success"
            html = (f'<div class="resultat {classe}">Résultats des probabilités :<br><br>'
                    f"À l'heure : {a_l_heure}%<br>Retardé : {retarde}%</div>")
            
            return html, self.creer_graphique_voisins(voisins, a_l_heure, retarde), voisins
        return predire

    def construire_interface(self, modele: mod.ModeleVol) -> gr.Blocks:
        """Construit l'interface Gradio autour d'un modèle déjà entraîné."""
        compagnies = list(modele._liste_airlines)
        with gr.Blocks(title="Prédiction de Retard de Vol (k-NN)") as demo:
            gr.HTML(CSS)
            gr.Markdown("## Prédiction de Vol (k-NN)")
            compagnie = gr.Dropdown(compagnies, label="Compagnie aérienne",
                                    value=compagnies[0] if compagnies else None)
            jour = gr.Dropdown(JOURS, label="Jour de la semaine (1 à 7)", value=1)
            heure = gr.Number(label="Heure de départ (0-23)", value=14, precision=0)
            minutes = gr.Number(label="Minutes de départ (0-59)", value=0, precision=0)
            duree = gr.Number(label="Durée du vol en minutes (ex: 495)", value=495, precision=0)
            bouton = gr.Button("Calculer le risque de retard", variant="primary")
            sortie = gr.HTML()
            graphique = gr.Plot(label="Les vols les plus proches")
            tableau = gr.Dataframe(label="Détail des vols utilisés pour la décision")
            bouton.click(
                self.creer_predicteur(modele),
                [compagnie, jour, heure, minutes, duree],
                [sortie, graphique, tableau]
            )
        return demo

def main() -> None:
    """Entraîne le modèle puis lance l'interface."""
    modele = mod.ModeleVol(CHEMIN_CSV)
    try:
        modele.charger()
    except Exception as e:
        print(e)
        modele.entrainer()
    print("_liste_airlines", modele._liste_airlines)
    interface = Interface()  # ✅ Instanciation corrigée
    interface.construire_interface(modele).launch()

if __name__ == "__main__":
    main()