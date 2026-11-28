"""Décorateur utilitaire pour mesurer le temps d'exécution."""

import time
from functools import wraps
from typing import Any, Callable


def chronometrer(fonction: Callable) -> Callable:
    """
    Decorateur @chronometrer permet de chronometrer le temps d'execution d'une fonction.
    prend en argument:
        la fonction à chronométrer
    retourne: 
        Une nouvelle fonction qui appelle `fonction`, affiche sa durée et renvoie
        son résultat tel quel.
    erreur:
        ne lève aucune erreur
    """
    @wraps(fonction)
    def enveloppe(*args: Any, **kwargs: Any) -> Any:
        """Appelle fonction avec les mêmes arguments, affiche sa durée et renvoie son résultat."""
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        duree = time.perf_counter() - debut
        print(f"{fonction.__name__} : {duree:.2f} s")
        return resultat

    return enveloppe
