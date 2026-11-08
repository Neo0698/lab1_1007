
import time
from functools import wraps
from typing import Callable


def chronometrer(fonction: Callable) -> Callable:
   
    @wraps(fonction)
    def enveloppe(*args, **kwargs):
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        duree = time.perf_counter() - debut
        print(f"{fonction.__name__} : {duree:.2f} s")
        return resultat
    return enveloppe