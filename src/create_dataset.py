"""Génère airlines.csv (Airline,DayOfWeek,Time,Length,Delay) à partir du jeu
Hugging Face Donny-Guo/airline-delay (sous-ensemble flight_detail).

L'échantillon final est :
  - équilibré : 50 % de vols en retard (Delay = 1) et 50 % à l'heure (Delay = 0) ;
  - le plus varié possible en compagnies : toutes les compagnies présentes dans
    le jeu sont conservées, et les places sont réparties équitablement entre
    elles (une petite compagnie garde tous ses vols, les grandes se partagent
    le reste).

Le jeu entier (61,6 M de lignes) est parcouru une fois en flux, par lots, sans
tout charger en mémoire. Prévoir plusieurs minutes.

Installation :  pip install datasets pandas numpy
Lancement   :  python creer_csv.py
"""
import numpy as np
import pandas as pd

NOM_DATASET = "Donny-Guo/airline-delay"
CONFIG = "flight_detail"
NB_LIGNES = 2_000_000         # total final (moitié en retard, moitié à l'heure)
SEUIL_RETARD_MIN = 15       # retard à l'arrivée (min) à partir duquel Delay = 1
GRAINE = 42                 # reproductibilité
SORTIE = "airlines.csv"
MAX_PAR_GROUPE = 40_000     # vols gardés au maximum par (compagnie, classe)
LOTS_PAR_FUSION = 20        # fusion de l'échantillon tous les N lots (plus rapide)
TAILLE_LOT = 50_000         # lignes lues à la fois
MAX_LIGNES_LUES = 5_000_000      # ex. 5_000_000 pour un essai rapide (None = tout lire)

COLONNES = ["fl_date", "op_carrier", "crs_dep_time", "crs_elapsed_time",
            "arr_delay", "cancelled", "diverted"]


def preparer_lot(lot: pd.DataFrame, seuil: int, rng: np.random.Generator) -> pd.DataFrame:
    """Filtre les vols inutilisables et calcule les colonnes finales d'un lot.

    Reçoit un lot brut, le seuil de retard et un générateur aléatoire ; renvoie
    un DataFrame Airline, DayOfWeek, Time, Length, Delay + une clé aléatoire
    « cle » servant à tirer des échantillons uniformes.
    """
    valide = (~lot["cancelled"].astype(bool) & ~lot["diverted"].astype(bool)
              & lot["arr_delay"].notna() & lot["crs_dep_time"].notna()
              & (lot["crs_elapsed_time"] > 0))   # le jeu contient des durées négatives
    v = lot[valide]
    hhmm = v["crs_dep_time"].astype(int)
    return pd.DataFrame({
        "Airline": v["op_carrier"].to_numpy(),
        "DayOfWeek": pd.to_datetime(v["fl_date"]).dt.dayofweek.to_numpy() + 1,  # 1 = lundi
        "Time": (((hhmm // 100) * 60 + hhmm % 100) % 1440).to_numpy(),          # min depuis minuit
        "Length": v["crs_elapsed_time"].astype(int).to_numpy(),
        "Delay": (v["arr_delay"] >= seuil).astype(int).to_numpy(),
        "cle": rng.random(len(v)),
    })


def garder_echantillon(garde: pd.DataFrame, lot: pd.DataFrame, k: int) -> pd.DataFrame:
    """Fusionne un lot avec les vols déjà gardés, en gardant au plus k vols par
    (compagnie, classe), ceux de plus petite clé aléatoire (tirage uniforme).

    Renvoie le nouveau DataFrame des vols gardés.
    """
    tout = lot if garde is None else pd.concat([garde, lot], ignore_index=True)
    return (tout.sort_values("cle")
                .groupby(["Airline", "Delay"], sort=False)
                .head(k)
                .reset_index(drop=True))


def repartir(disponibles: dict[str, int], total: int) -> dict[str, int]:
    """Répartit `total` places entre les compagnies le plus équitablement possible.

    Reçoit le nombre de vols disponibles par compagnie ; une compagnie n'obtient
    jamais plus que ce qu'elle a, et le surplus est redistribué aux autres.
    Renvoie le quota par compagnie.
    """
    quotas: dict[str, int] = {}
    restant = total
    petites_d_abord = sorted(disponibles.items(), key=lambda paire: paire[1])
    for i, (compagnie, n) in enumerate(petites_d_abord):
        part = restant // (len(petites_d_abord) - i)
        quotas[compagnie] = min(n, part)
        restant -= quotas[compagnie]
    return quotas


def equilibrer(garde: pd.DataFrame, nb_total: int) -> pd.DataFrame:
    """Construit l'échantillon final 50 % retard / 50 % à l'heure, réparti
    équitablement entre les compagnies.

    Reçoit les vols gardés et le nombre total de lignes voulu ; renvoie le
    DataFrame mélangé (colonnes Airline, DayOfWeek, Time, Length, Delay).
    """
    par_classe = {d: garde[garde["Delay"] == d] for d in (0, 1)}
    moitie = min(nb_total // 2, len(par_classe[0]), len(par_classe[1]))
    parts = []
    for groupe in par_classe.values():
        quotas = repartir(groupe.groupby("Airline").size().to_dict(), moitie)
        for compagnie, quota in quotas.items():
            parts.append(groupe[groupe["Airline"] == compagnie].head(quota))
    final = pd.concat(parts).drop(columns="cle")
    return final.sample(frac=1, random_state=GRAINE).reset_index(drop=True)


def main() -> None:
    """Parcourt le jeu en flux, équilibre l'échantillon et écrit le CSV."""
    from datasets import load_dataset

    rng = np.random.default_rng(GRAINE)
    flux = load_dataset(NOM_DATASET, CONFIG, split="full", streaming=True)
    flux = flux.select_columns(COLONNES)

    garde, en_attente, lus = None, [], 0
    for numero, lot in enumerate(flux.iter(batch_size=TAILLE_LOT), start=1):
        en_attente.append(preparer_lot(pd.DataFrame(lot), SEUIL_RETARD_MIN, rng))
        lus += TAILLE_LOT
        if numero % LOTS_PAR_FUSION == 0:
            garde = garder_echantillon(
                garde, pd.concat(en_attente, ignore_index=True), MAX_PAR_GROUPE)
            en_attente = []
            print(f"{lus:,} lignes lues, {garde['Airline'].nunique()} compagnies")
        if MAX_LIGNES_LUES is not None and lus >= MAX_LIGNES_LUES:
            break
    if en_attente:
        garde = garder_echantillon(
            garde, pd.concat(en_attente, ignore_index=True), MAX_PAR_GROUPE)

    df = equilibrer(garde, NB_LIGNES)
    df.to_csv(SORTIE, index=False)
    if len(df) < NB_LIGNES:
        print(f"Attention : {len(df)} lignes au lieu de {NB_LIGNES}. "
              "Augmentez MAX_PAR_GROUPE ou lisez plus de lignes (MAX_LIGNES_LUES = None).")
    print(f"{len(df)} lignes écrites dans {SORTIE}")
    print(f"Vols en retard : {df['Delay'].mean():.1%}")
    print(f"Compagnies : {df['Airline'].nunique()}")
    print(df.groupby("Airline")["Delay"].agg(["size", "mean"]).round(2))


if __name__ == "__main__":
    main()