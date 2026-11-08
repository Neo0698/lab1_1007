"""Exceptions personnalisées du projet."""


class DonneeInvalideError(Exception):
    """Levée quand une saisie est vide, non numérique ou hors intervalle."""
