# Projet 2 INF1007 - Prédiction de retards de vol

**Noms :** Hassan Ali Mohamed, Raphael Scott

## Description

Application locale (Gradio) qui estime la probabilité qu'un vol arrive en retard (15 minutes ou plus) selon la compagnie aérienne, l'aéroport de départ, le jour de la semaine, l'heure de départ et la durée prévue.

La prédiction utilise un modèle des *k* plus proches voisins (`k = 31`). L'interface affiche les probabilités, un graphique des 31 vols passés les plus proches avec leur détail, et garde un historique des prédictions en CSV.

## Jeu de données

Source : [`Donny-Guo/airline-delay`](https://huggingface.co/datasets/Donny-Guo/airline-delay) sur Hugging Face (données publiques du *Bureau of Transportation Statistics* américain, libres pour un usage éducatif). On l'a choisi plutôt que celui de Kaggle car il contient plus de compagnies aériennes.

Le script create_dataset.py parcourt les 5 premiers millions de lignes en flux, retire les vols annulés, déroutés ou incomplets, et extrait un échantillon équilibré (50 % à l'heure / 50 % en retard) réparti entre les compagnies.

### Nettoyage
`create_dataset.py` retire les vols annulés ou déroutés, car leur retard à l'arrivée n'est pas mesurable et la règle des 15 minutes ne peut pas s'appliquer. Il retire aussi les lignes sans retard, sans heure de départ ou sans aéroport, ainsi que les durées nulles ou négatives présentes dans le jeu d'origine. Il convertit l'heure `HHMM` en minutes depuis minuit, la date en jour de la semaine, et le retard en `Delay` = 1 s'il atteint 15 minutes.

## Installation et lancement

```bash
pip install -r requirements.txt
python -m src.create_dataset   # optionnel : régénère donnees/Airlines.csv
python -m src.model            # optionnel : entraîne le modèle et crée exploration.png
python -m src.main             # lance l'interface sur http://127.0.0.1:7860
```

Comme `modeles/knn.joblib` est trop lourd pour GitHub, `python -m src.main` entraîne et sauvegarde le modèle automatiquement au premier lancement s'il n'est pas présent.

## Structure du code

```text
donnees/              # Airlines.csv et graphiques générés
modeles/              # modèle sauvegardé (.joblib) et metadonnees.json
src/
  main.py             # point d'entrée de l'application
  create_dataset.py   # extraction du CSV depuis Hugging Face
  dataset.py          # classe abstraite Dataset
  Data.py             # chargement, découpage 80/20 et exploration
  model.py            # ModeleVol (préparation, entraînement, prédiction)
  variables.py        # validation des saisies (numériques et catégorielles)
  interface.py        # application Gradio
  historique.py       # historique des prédictions (CSV)
  exceptions.py       # DonneeInvalideError
  outils.py           # décorateur @chronometrer
  choix_modele.py     # comparaison selon k et la taille des données
tests/                # tests unitaires pytest
```

Le décorateur `@chronometrer` (dans src/outils.py) sert à mesurer le temps d'entraînement du modèle.

## Résultats et limites

- Sur le jeu de test (80/20, `random_state=42`), le k-NN obtient **64,5 % d'exactitude** (avec pondération des variables), contre **50,0 %** pour la référence (`DummyClassifier`).

- L'heure de départ et la compagnie jouent beaucoup (les retards s'accumulent en fin de journée), alors que le jour et la durée influencent peu le résultat. Le modèle se trompe quand même environ une fois sur trois : il manque dans les données les causes principales des retards (météo, trafic aérien, retard de l'avion précédent), ce qui limite le projet.

- Seul le jour de la semaine est utilisé : le modèle ne connaît ni le mois, ni les jours fériés. Nos données ne couvrent de plus qu'une partie d'une année (les 5 premiers millions de vols du jeu, vers 2009), ce qui ne permettrait pas d'apprendre l'effet des saisons.

- Le jeu de données ne contient que des compagnies américaines, dont certaines ont disparu depuis.

- Encodages imparfaits : les compagnies sont numérotées par ordre alphabétique, donc deux compagnies paraissent « proches » sans raison réelle ; de même, dimanche (7) paraît éloigné de lundi (1).

## Tests et documentation

- Tests : pytest tests
- Couverture (coverage) : **58 %** sur "src/" ( coverage run --source=src -m pytest tests && coverage report). 

Ce taux s'explique surtout par le fait que les tests couvrent plutôt la logique du modèle et la validation des données (model.py, Data.py, variables.py), alors que les scripts d'expérimentation (create_dataset.py, choix_modele.py), le point d'entrée (main.py) et la construction visuelle Gradio ne sont pas testés en unitaire.

Documentation consultée :
- pandas : https://pandas.pydata.org/docs/user_guide/10min.html
- scikit-learn : https://scikit-learn.org/stable/modules/neighbors.html
- Gradio : https://www.gradio.app/docs/gradio/blocks