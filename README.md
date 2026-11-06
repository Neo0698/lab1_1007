# LAB2 INF1007 - Prédiction de retards de vol
**nom** Hassan Ali Mohamed, Raphael Scott


## Prédiction du retard d'un vol en utilisant un KNN

### Description
L'application estime si l'avion via arriver en retard( on estime qu'un vol est en retard ) selon la compagnie aérienne, le jour de la semaine, l'heure de départ prévue et la durée prévue du vol. Un modèle des k plus proches voisins (k-NN) fait la prédiction, et l'interface (Gradio) affiche les probabilités ainsi que les vols passés les plus proches du vol saisi.
L’application retourne :
-La probabilité que le vol soit a l'heure ou en retard
-un graphe représentant les exemples de vol similaire à l'heure et en retard
-une liste de ces vols avec la distance

### Le jeu de données
Le jeu de données vien de hugging face, celui possède légèrement plus de compagnie que celui disponible sur Kaggle
https://huggingface.co/datasets/Donny-Guo/airline-delay
On utilise le programme create_dataset.py pour extraire 50 000 lines avec 50% de vol en retard et 50% de vol a l'heure pour que celui-ci soit équilibré. 

### test effectué
En testant de multiples algorithme d'IA nous avons determiné qu'un modèle KNN avec k=31 etait une bonne combinaison pour trouver des résultats correcte. Le modèle commence a comprendre des patterns tel qu'un vol court courrier avait plus tendance à être en retard en fin qu'en debut de journée.

