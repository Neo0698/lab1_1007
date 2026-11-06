
# Prédiction du retard d'un vol en utilisant un KNN

## Description
L'application estime si l'avion via arriver en retard( on estime qu'un vol est en retard ) selon la compagnie aérienne, le jour de la semaine, l'heure de départ prévue et la durée prévue du vol. Un modèle des k plus proches voisins (k-NN) fait la prédiction, et l'interface (Gradio) affiche les probabilités ainsi que les vols passés les plus proches du vol saisi.

## Le jeu de données
Le jeu de données vien de hugging face, celui possède légèrement plus de compagnie que celui disponible sur Kaggle
https://huggingface.co/datasets/Donny-Guo/airline-delay
On utilise le programme create_dataset.py pour extraire 50 000 lines avec 50% de vol en retard et 50% de vol a l'heure pour que celui-ci soit équilibré. 