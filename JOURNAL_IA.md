# Journal d'utilisation de l'IA

## Méthode générale

Nous avons utilisé Claude Code pour écrire le code, faire les commits et les pushs, et décrire explicitement les modifications apportées à chaque étape.

## Résolution de problèmes

### Problèmes d'importation

Certains imports fonctionnaient en local sur notre machine, mais échouaient dans les tests automatiques GitHub. L'IA nous a aidés à identifier le problème et à réécrire les imports pour que le code fonctionne dans les deux environnements.

### Push refusé à cause de la taille du modèle

Le push était bloqué parce que le fichier `knn.joblib` pèse 174 Mo. Le message d'erreur de Git n'était pas clair au départ. Nous avons dû comprendre la limite de taille imposée par GitHub pour résoudre le problème.

## Apprentissages

### Interface graphique

Nous avons appris à utiliser la bibliothèque Gradio pour construire l'interface graphique du projet.

### Tests avec pytest

L'IA nous a permis de comprendre les mécanismes suivants dans `tests/test_model.py` :

- **`@pytest.mark.parametrize`** : exécuter un même test avec plusieurs jeux de valeurs. Par exemple, `@pytest.mark.parametrize("jour", [0, 8])` exécute le test deux fois, avec `jour=0` puis `jour=8`.
- **Fixtures** : `modele_entraine` est une fixture de portée `module`. Le modèle est entraîné une seule fois et partagé entre tous les tests du fichier.
- **`pytest.raises`** : vérifier qu'une erreur précise est bien levée.

Nous avons aussi relevé des incohérences sur la docstring ou aussi pour pointer les fonction n'ayant pas de docstring.