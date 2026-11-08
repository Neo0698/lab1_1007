"""Variables d'entrée du modèle : chacune valide sa saisie à sa façon."""

from abc import ABC, abstractmethod

from .exceptions import DonneeInvalideError


class Variable(ABC):
    """Variable d'entrée abstraite.

    Args:
        nom: Nom de la variable.
        libelle: Texte affiché à l'utilisateur.
    """

    def __init__(self, nom: str, libelle: str) -> None:
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        """Nom de la variable."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Texte affiché à l'utilisateur."""
        return self._libelle

    def _verifier_non_vide(self, valeur) -> None:
        """Lève DonneeInvalideError si la valeur est vide (None ou chaîne vide)."""
        if valeur is None or (isinstance(valeur, str) and valeur.strip() == ""):
            raise DonneeInvalideError(f"Le champ « {self._libelle} » est vide.")

    @abstractmethod
    def valider(self, valeur):
        """Valide la saisie et la renvoie convertie.

        Raises:
            DonneeInvalideError: si la saisie est invalide.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nom={self._nom!r})"


class VariableNumerique(Variable):
    """Variable numérique bornée par [minimum, maximum]."""

    def __init__(self, nom: str, libelle: str, minimum: float, maximum: float) -> None:
        super().__init__(nom, libelle)
        self._minimum = minimum
        self._maximum = maximum

    def valider(self, valeur) -> float:
        """Renvoie la valeur en float si c'est un nombre dans l'intervalle.

        Raises:
            DonneeInvalideError: si la valeur est vide, non numérique ou hors intervalle.
        """
        self._verifier_non_vide(valeur)
        try:
            nombre = float(valeur)
        except (TypeError, ValueError):
            raise DonneeInvalideError(f"« {self._libelle} » doit être un nombre.") from None
        if not self._minimum <= nombre <= self._maximum:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être entre {self._minimum} et {self._maximum}.")
        return nombre

    def __str__(self) -> str:
        return f"{self._libelle} ({self._minimum} à {self._maximum})"


class VariableCategorielle(Variable):
    """Variable dont la valeur doit appartenir à une liste fixe."""

    def __init__(self, nom: str, libelle: str, valeurs: list[str]) -> None:
        super().__init__(nom, libelle)
        self._valeurs = list(valeurs)

    @property
    def valeurs(self) -> list[str]:
        """Copie de la liste des valeurs possibles."""
        return list(self._valeurs)

    def valider(self, valeur) -> str:
        """Renvoie la valeur si elle fait partie des valeurs possibles.

        Raises:
            DonneeInvalideError: si la valeur est vide ou inconnue.
        """
        self._verifier_non_vide(valeur)
        if valeur not in self._valeurs:
            raise DonneeInvalideError(
                f"« {valeur} » n'est pas une valeur possible de « {self._libelle} ».")
        return valeur

    def __str__(self) -> str:
        return f"{self._libelle} ({len(self._valeurs)} choix)"


def variables_depuis_metadonnees(meta: dict) -> list[Variable]:
    """Construit les variables d'entrée, dans l'ordre du formulaire, à partir du JSON.

    Args:
        meta: Métadonnées du modèle (contenu de metadonnees.json).

    Returns:
        [compagnie, aéroport, jour, heure, minutes, durée].

    Raises:
        KeyError: si une clé attendue manque dans les métadonnées.
    """
    jour_min, jour_max = meta["intervalle_dayofweek"]
    return [
        VariableCategorielle("Airline", "Compagnie aérienne", meta["airlines_possibles"]),
        VariableCategorielle("Origin", "Aéroport de départ", meta["aeroports_possibles"]),
        VariableNumerique("DayOfWeek", "Jour de la semaine", jour_min, jour_max),
        VariableNumerique("Heure", "Heure de départ", 0, 23),
        VariableNumerique("Minutes", "Minutes de départ", 0, 59),
        VariableNumerique("Length", "Durée du vol (min)", 1, meta["intervalle_length"][1]),
    ]
