from abc import ABC, abstractmethod
from typing import Optional 
import pandas as pd

class Dataset(ABC):
    @abstractmethod
    def charger_donnees(self, path: str) -> pd.DataFrame:
        """Charge les données depuis un fichier."""
        pass

    @property
    @abstractmethod
    def data(self) -> Optional[pd.DataFrame]:  
        """Retourne le DataFrame des données."""
        pass

    @property
    @abstractmethod
    def liste_airlines(self) -> list[str]:
        """Retourne la liste des compagnies."""
        pass