"""Interface Gradio : prédiction de retard de vol (k-NN)."""

import gradio as gr
from matplotlib.figure import Figure
import pandas as pd

from .exceptions import DonneeInvalideError
from .historique import Historique
from .model import ModeleVol
from .variables import variables_depuis_metadonnees

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

class Interface:
    """Application Gradio : contient le modèle, l'historique et les variables d'entrée."""

    def __init__(self, modele: ModeleVol, historique: Historique) -> None:
        self._modele = modele
        self._historique = historique
        self._variables = variables_depuis_metadonnees(modele.metadonnees)

    def valider(self, compagnie, aeroport, jour, heure, minutes, duree) -> list:
        """Valide chaque saisie avec sa variable et renvoie les valeurs converties.

        Raises:
            DonneeInvalideError: si une saisie est vide, non numérique ou hors intervalle.
        """
        saisies = (compagnie, aeroport, jour, heure, minutes, duree)
        return [variable.valider(saisie) for variable, saisie in zip(self._variables, saisies)]

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

    def predire(self, compagnie, aeroport, jour, heure, minutes, duree):
        """Callback du bouton : valide, prédit, enregistre dans l'historique et affiche.

        Returns:
            (html du résultat, figure des voisins, tableau des voisins) ;
            en cas de saisie invalide : (message d'erreur, None, None).
        """
        try:
            compagnie, aeroport, jour, heure, minutes, duree = self.valider(
                compagnie, aeroport, jour, heure, minutes, duree)
            prob = self._modele.predire(compagnie, aeroport, int(jour), int(heure), int(minutes), duree)
            voisins = self._modele.voisins(compagnie, aeroport, int(jour), int(heure), int(minutes), duree)
        except DonneeInvalideError as e:
            return f'<div class="resultat error">{e}</div>', None, None

        a_l_heure, retarde = round(prob[0] * 100, 2), round(prob[1] * 100, 2)
        classe = "danger" if retarde > 50 else "success"
        html = (f'<div class="resultat {classe}">Résultats des probabilités :<br><br>'
                f"À l'heure : {a_l_heure}%<br>Retardé : {retarde}%</div>")

        self._historique.ajouter(compagnie, aeroport, jour, heure, minutes, duree, prob[1])
        return html, self.creer_graphique_voisins(voisins, a_l_heure, retarde), voisins

    def construire_interface(self) -> gr.Blocks:
        """Construit l'interface Gradio à partir des métadonnées du modèle chargé."""
        compagnies = self._variables[0].valeurs
        aeroports = self._variables[1].valeurs
        with gr.Blocks(title="Prédiction de Retard de Vol (k-NN)") as demo:
            gr.HTML(CSS)
            gr.Markdown("## Prédiction de Vol (k-NN)")
            compagnie = gr.Dropdown(compagnies, label="Compagnie aérienne",
                                    value=compagnies[0] if compagnies else None)
            aeroport = gr.Dropdown(aeroports, label="Aéroport de départ (code IATA, ex. JFK)",
                                   value="JFK" if "JFK" in aeroports else (aeroports[0] if aeroports else None),
                                   filterable=True)
            jour = gr.Dropdown(JOURS, label="Jour de la semaine (1 à 7)", value=1)
            heure = gr.Number(label="Heure de départ (0-23)", value=14, precision=0)
            minutes = gr.Number(label="Minutes de départ (0-59)", value=0, precision=0)
            duree = gr.Number(label="Durée du vol en minutes (ex: 495)", value=495, precision=0)
            bouton = gr.Button("Calculer le risque de retard", variant="primary")
            sortie = gr.HTML()
            graphique = gr.Plot(label="Les vols les plus proches")
            tableau = gr.Dataframe(label="Détail des vols utilisés pour la décision")
            bouton.click(
                self.predire,
                [compagnie, aeroport, jour, heure, minutes, duree],
                [sortie, graphique, tableau]
            )
            with gr.Accordion("Historique des prédictions", open=False):
                tableau_historique = gr.Dataframe(value=self._historique.lire)
                gr.Button("Rafraîchir l'historique").click(self._historique.lire, None, tableau_historique)
        return demo

def main() -> None:
    """Charge le modèle sauvegardé (ou l'entraîne et le sauvegarde une fois), puis lance l'interface."""
    modele = ModeleVol(CHEMIN_CSV)
    try:
        modele.charger()
    except FileNotFoundError as e:
        print(f"{e} : entraînement du modèle…")
        modele.sauvegarder(modele.entrainer())
        modele.charger()
    Interface(modele, Historique()).construire_interface().launch()

if __name__ == "__main__":
    main()