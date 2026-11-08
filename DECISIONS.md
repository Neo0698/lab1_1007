### choix du dataset

Nous avons décider de prendre le dataset sur huggingface suivant: https://huggingface.co/datasets/Donny-Guo/airline-delay comparé au dataset https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024 car il contient un plus grand nombre de compagnie aerienne, nous avons du ensuite créer un programme python pour lire est extraire les 5 crithères que nous souhaitons pour entrainer le modèle, mais aussi extraire dans les mêmes proportions le nombres de vol en retard et a l'heure, le programme s'occupe d'extraire 50 000 de ces lignes parmis les 60 millions disponible dans le dataset de huggingface. 

### Justification du choix du modèle k-NN (k=31)
Nous avons essayé de multiples modèles, les deux meilleurs était le KNN et la forêt aleatoire, en plus d'être legerement plus lent la foret aleatoire été aussi legerement moins exacte.

### Choix des testes
- 1er test, lancer le modèle sur un fichier qui n'existe pas pour vérifier si une erreur est bien généré par model.py 
- 2ième test, lancer le modèle cette fois ci avec le bon chemin d'acces pour tester s'il est capable d'importer ce fichier sans erreur
- 3ième test, verification que l'ajout d'une ligne vide n'empeche pas l'entrainement du modèle
- 4ième test, verification qu'à la suite de modele.preparer, que  Length_Carre a été calculé correctement, que le nom des compagnie on été transformé correctement en vecteurs pour qu'ils puissent être interprété par le KNN., liste airlines contiennent bien l'ensemble des compagnies présente dans le dataset
- 5ème test, vérifie si après l'entrainement le dataset chargé ne change pas, manque pas une colonne ou ajout d'une nouvelle colonne
- 6ème test, vérification que la fonction retourne bien deux valeurs exactitude_knn et exactitude_reference
- 7ème test, verification que le modèle knn fasse mieux que le modèle de référence.
- 8ème test, verification qu'on ne puisse pas prédire sur un modèle non entrainé
- 9ème test, qu'on ne puisse pas prédire sur une compagnie inconnue
- 10 ème test, test pour les valeurs: 0 et 8 si la prediction renvoi bien une erreur
- 11ième test, test pour les valeurs (heure, minutes) suivante (24, 0), (10, 60), (-1, 30) qu'une erreur est bien produite
- 12ième test, test qu'une duree de vol non possible renvoi bien une erreur, testé avec les valeur temps nulle et un temps négatif
- 13ième test, test si la fonction prédire renvoi bien la probabilité que le vol soit a l'heure et que le vol soit en retard et que les deux font bien 100%
- 14ième test, test si un vol long est plus en retard (car hypothèse créé dans le dataset de test avec 1 if length >300)
- 15 ième test, verifie que les cas limites renvoi bien les deux prédictions
- 16ième test, verifie qu'un modele non entrainné ne peut pas être entrainné
- 17ième test, verifie qu'un modèle entrainné peut être sauvegardé et que cette action est réalisé correctement en sauvegardant toutes les données nécessaires