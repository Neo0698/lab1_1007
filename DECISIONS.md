### choix du dataset

Nous avons décider de prendre le dataset sur huggingface suivant: https://huggingface.co/datasets/Donny-Guo/airline-delay comparé au dataset https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024 car il contient un plus grand nombre de compagnie aerienne, nous avons du ensuite créer un programme python pour lire est extraire les 5 crithères que nous souhaitons pour entrainer le modèle, mais aussi extraire dans les mêmes proportions le nombres de vol en retard et a l'heure, le programme s'occupe d'extraire 50 000 de ces lignes parmis les 60 millions disponible dans le dataset de huggingface. 

### Justification du choix du modèle k-NN (k=31)
