# LAB2 INF1007 - Prédiction de retards de vol
**nom** Hassan Ali Mohamed, Raphael Scott


## Prédiction du retard d'un vol en utilisant un KNN

### Structure du code
```
donnees/
	Airlines.csv          # jeu de données
	exploration.png       # graphique d'exploration
modeles/
src/
	create_dataset.py     # extraction du CSV depuis Hugging Face
	dataset.py            # classe abstraite Dataset
	model.py              # Data (données) et ModeleVol (préparation, entraînement, prédiction, sauvegarde)
	variables.py          # Variable → VariableNumerique / VariableCategorielle 
	exceptions.py         # DonneeInvalideError
	historique.py         # historique des prédictions (CSV)
	outils.py             # décorateur @chronometrer
	interface.py          # application Gradio
	choix_modele.py       # recherche du meilleur k (score_vs_k.png)

tests/
	test_model.py
```

### Description
L'application estime si l'avion via arriver en retard( on estime qu'un vol est en retard ) selon la compagnie aérienne, le jour de la semaine, l'heure de départ prévue et la durée prévue du vol. Un modèle des k plus proches voisins (k-NN) fait la prédiction, et l'interface (Gradio) affiche les probabilités ainsi que les vols passés les plus proches du vol saisi.
L’application retourne :
-La probabilité que le vol soit a l'heure ou en retard
-un graphe représentant les exemples de vol similaire à l'heure et en retard
-une liste de ces vols avec la distance

### Le jeu de données
Le jeu de données provient de Hugging Face : Donny-Guo/airline-delay. Nous l’avons préféré à celui de Kaggle, car il contient un peu plus de compagnies aériennes.

Le programme create_dataset.py en extrait le nombre de vols souhaité pour entraîner le modèle. Il parcourt les MAX_LIGNES_LUES premières lignes du jeu Hugging Face (5 millions) afin de récupérer le plus de compagnies possible. Il garde ensuite autant de vols en retard que de vols à l’heure (50 % / 50 %), pour que l’échantillon soit équilibré.

### test effectué
En testant de multiples algorithme d'IA nous avons determiné qu'un modèle KNN avec k=31 etait une bonne combinaison pour trouver des résultats correcte. Le modèle commence a comprendre des patterns tel qu'un vol court courrier avait plus tendance à être en retard en fin qu'en debut de journée.


### Analyse des résultats
Le kNN obtient 63,4 % d’exactitude contre 50,0 % pour la référence (classe majoritaire), soit 12,8 points de mieux. Le modèle a donc appris quelque chose. Il se trompe pourtant encore sur plus d’un vol sur trois.

L’heure de départ et la compagnie sont les variables les plus utiles (de 22 % à 60 % de retards selon l’heure, de 8 % à 74 % selon la compagnie). Le jour de la semaine influence peu, et la durée du vol presque pas. Il manque les causes principales des retards : la météo, l’aéroport, le trafic, les retards en cascade.

### Lancement du programme
pip install -r requirements.txt
python -m src.create_dataset *pour creer le csv en fonction du modèle huggingface*
python -m src.model *entraine le modèle*
python -m src.interface *lance l'interface graphique*