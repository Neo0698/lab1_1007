"""Classes de validation des variables d'entrée du formulaire."""

from abc import ABC, abstractmethod
from typing import Any, Union

from .exceptions import DonneeInvalideError


class Variable(ABC):
    """Classe abstraite représentant une variable saisie par l'utilisateur."""

    def __init__(self, nom: str, libelle: str) -> None:
        """Garde le nom de la variable et le texte affiché à l'utilisateur."""
        self._nom = nom
        self._libelle = libelle

    def _verifier_non_vide(self, valeur: Any) -> None:
        """Lève DonneeInvalideError si la valeur est vide."""
        if valeur is None or (isinstance(valeur, str) and valeur.strip() == ""):
            raise DonneeInvalideError(f"Le champ '{self._libelle}' est vide.")

    @abstractmethod
    def valider(self, valeur: Any) -> Union[float, str]:
        """Valide la saisie et la renvoie convertie.

        Raises:
            DonneeInvalideError: Si la saisie est invalide.
        """

    def __repr__(self) -> str:
        """Retourne une représentation texte de la variable."""
        return f"{self.__class__.__name__}(nom='{self._nom}')"


class VariableNumerique(Variable):
    """Variable numérique devant rester dans un intervalle [minimum, maximum]."""

    def __init__(self, nom: str, libelle: str, minimum: float, maximum: float) -> None:
        """Garde le nom, le libellé et l'intervalle [minimum, maximum] accepté."""
        super().__init__(nom, libelle)
        self._minimum = minimum
        self._maximum = maximum

    def valider(self, valeur: Any) -> float:
        """Renvoie la valeur en float si c'est un nombre dans l'intervalle.

        Raises:
            DonneeInvalideError: si la valeur est vide, non numérique ou hors intervalle.
        """
        self._verifier_non_vide(valeur)
        try:
            nombre = float(valeur)
        except (TypeError, ValueError):
            raise DonneeInvalideError(f"'{self._libelle}' doit être un nombre.")
        if not (self._minimum <= nombre <= self._maximum):
            raise DonneeInvalideError(
                f"'{self._libelle}' doit être entre {self._minimum} et {self._maximum}."
            )
        return nombre


class VariableCategorielle(Variable):
    """Variable dont la valeur doit appartenir à une liste fixe."""

    def __init__(self, nom: str, libelle: str, valeurs: list[str]) -> None:
        """Garde le nom, le libellé et une copie de la liste des valeurs acceptées."""
        super().__init__(nom, libelle)
        self._valeurs = list(valeurs)

    @property
    def valeurs(self) -> list[str]:
        """Retourne une copie des valeurs possibles."""
        return list(self._valeurs)

    def valider(self, valeur: Any) -> str:
        """Vérifie que la valeur choisie fait partie de la liste autorisée.

        Raises:
            DonneeInvalideError: si la valeur est vide ou inconnue.
        """
        self._verifier_non_vide(valeur)
        if valeur not in self._valeurs:
            raise DonneeInvalideError(
                f"'{valeur}' n'est pas une valeur valide pour '{self._libelle}'."
            )
        return valeur


def variables_depuis_metadonnees(meta: dict) -> list[Variable]:
    """Crée la liste des 6 variables du formulaire à partir de metadonnees.json.

    Args:
        meta: Dictionnaire chargé depuis metadonnees.json.

    Returns:
        [compagnie, aéroport, jour, heure, minutes, durée].

    Raises:
        KeyError: si une clé attendue manque dans les métadonnées.
    """
    jour_min, jour_max = meta["intervalle_dayofweek"]
    duree_min, duree_max = meta["intervalle_length"]
    return [
        VariableCategorielle("Airline", "Compagnie aérienne", meta["airlines_possibles"]),
        VariableCategorielle("Origin", "Aéroport de départ", meta["aeroports_possibles"]),
        VariableNumerique("DayOfWeek", "Jour de la semaine", jour_min, jour_max),
        VariableNumerique("Heure", "Heure de départ", 0, 23),
        VariableNumerique("Minutes", "Minutes de départ", 0, 59),
        VariableNumerique("Length", "Durée du vol (min)", duree_min, duree_max),
    ]
