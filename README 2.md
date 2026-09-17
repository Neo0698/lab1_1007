# Projet 1 - INF1007 Automne 2026

## Directives
:alarm_clock: Date de remise : **Dimanche 4 octobre 23:59**

:mailbox_with_mail: À remettre sur **Moodle** sous la forme d'un fichier zip.

## Introduction

Dans ce projet, vous aurez comme tâche de compléter une version du jeu **Doodle Jump** 🦘 avec la bibliothèque Python `pygame`.

L'objectif du jeu est de faire monter le personnage, appelé le **Doodle**, le plus haut possible en rebondissant automatiquement de plateforme en plateforme. Le joueur contrôle uniquement les déplacements horizontaux du Doodle. Lorsque celui-ci atteint une certaine hauteur dans la fenêtre, la caméra défile : les plateformes descendent à l'écran et le score augmente selon la distance parcourue.

https://github.com/user-attachments/assets/0f303204-aaac-4647-a9ab-126cf3d4a84d

Le jeu contient quatre types de plateformes :

- **Plateforme verte** : plateforme normale et fixe ;
- **Plateforme bleue** : plateforme qui se déplace horizontalement ;
- **Plateforme marron** : plateforme fragile qui se brise après le rebond ;
- **Plateforme à ressort** : plateforme qui donne au Doodle une impulsion plus forte vers le haut.

Le joueur dispose d'une vie. La partie se termine lorsque le Doodle tombe sous le bas de l'écran. Il est alors possible d'appuyer sur la touche `R` pour recommencer.

Afin de simplifier votre travail, plusieurs éléments sont déjà fournis : l'affichage graphique, le chargement des images, l'écran de fin de partie, le redémarrage du jeu et certaines fonctions utilitaires. Votre travail portera principalement sur la manipulation de dictionnaires, les conditions, les boucles, la génération aléatoire, la physique simple et la détection de collisions.

**Pour lancer le jeu, vous devez exécuter le fichier `main.py`.**

## Installations requises

Ce projet nécessite l'utilisation de la bibliothèque [`pygame`](https://www.pygame.org/wiki/about).

Avant de commencer, assurez-vous que l'environnement conda `INF1007` est activé dans VS Code :

```bash
conda activate INF1007
```

Ensuite, installez Pygame :

```bash
pip install -U pygame==2.6.0
```

## Informations sur le projet

### Structure du projet

Le projet est organisé de la manière suivante :

```plaintext
2026A-PR01/
├── assets/
│   ├── background.png
│   ├── doodle_left.png
│   ├── doodle_right.png
│   ├── platform_green.png
│   ├── platform_blue.png
│   ├── platform_brown.png
│   └── platform_spring.png
├── config.py
├── doodle.py
├── platforms.py
├── window.py
├── game.py
├── main.py
└── README.md
```

### Détails sur les fichiers

- Le dossier `assets/` contient toutes les images utilisées dans le jeu.

- Le fichier `config.py` contient les constantes et variables globales du jeu, notamment :
  - `SCREEN_WIDTH` et `SCREEN_HEIGHT` : dimensions de la fenêtre ;
  - `DOODLE_WIDTH`, `DOODLE_HEIGHT` et `DOODLE_SIZE` : dimensions du Doodle ;
  - `DOODLE_START_X` et `DOODLE_START_Y` : position de départ du Doodle ;
  - `PLATFORM_WIDTH`, `PLATFORM_HEIGHT` et `PLATFORM_SIZE` : dimensions des plateformes ;
  - `MIN_PLATFORM_GAP` et `MAX_PLATFORM_GAP` : distances verticales minimale et maximale entre deux plateformes ;
  - `GRAVITY` : accélération verticale appliquée au Doodle ;
  - `JUMP_VELOCITY` : vitesse verticale appliquée lors d'un rebond normal ;
  - `SPRING_JUMP_VELOCITY` : vitesse verticale appliquée lors d'un rebond sur un ressort ;
  - `DOODLE_SPEED` : vitesse horizontale du Doodle ;
  - `MOVING_PLATFORM_SPEED` : vitesse horizontale des plateformes bleues ;
  - `CAMERA_SCROLL_THRESHOLD` : hauteur à partir de laquelle la caméra commence à défiler ;
  - `PLATFORMS` : liste globale contenant les dictionnaires des plateformes ;
  - `doodle_dict` : dictionnaire global contenant l'état du Doodle.

- Le fichier `doodle.py` charge les images du Doodle et initialise son dictionnaire.

- Le fichier `platforms.py` charge les images des quatre types de plateformes et contient les fonctions `create_platform()` et `choose_platform_type()`, utilisées pour représenter les plateformes et choisir leur type aléatoirement.

- Le fichier `window.py` crée la fenêtre, génère la disposition initiale des plateformes et gère l'affichage du jeu.

- Le fichier `game.py` contient la logique principale du jeu : mouvements, gravité, collisions, rebonds, plateformes mobiles, défilement de caméra et génération de nouvelles plateformes.

- Le fichier `main.py` contient la boucle principale du jeu. **Vous ne devez pas modifier ce fichier.**

### Repère de coordonnées

Dans Pygame, le point `(0, 0)` se trouve dans le coin **supérieur gauche** de la fenêtre :

```plaintext
(0, 0) ───────────────► x
   │
   │
   │        fenêtre de jeu
   │
   ▼
   y
```

Ainsi :

- augmenter `x` déplace un objet vers la droite ;
- diminuer `x` déplace un objet vers la gauche ;
- augmenter `y` déplace un objet vers le bas ;
- diminuer `y` déplace un objet vers le haut.

# Travail à réaliser

Vous devez compléter les sections identifiées par `TODO` dans les fichiers `doodle.py`, `platforms.py`, `window.py` et `game.py`.

> [!IMPORTANT]
> Plusieurs fonctionnalités sont déjà implémentées pour vous. Prenez le temps de lire le code fourni et de comprendre les dictionnaires `doodle_dict` et `PLATFORMS` avant de commencer. Les parties sont conçues pour être réalisées dans l'ordre.

## PARTIE 1 : Le Doodle 🦘

### 1.1 : Position initiale du Doodle

Dans le fichier `doodle.py`, le dictionnaire `doodle_dict` contient toutes les informations nécessaires pour représenter le personnage :

```python
doodle_dict.update({
    "x": 1000,
    "y": 1000,
    "vel_y": 0.0,
    "direction": "right",
    "score": 0,
    "high_score": 0,
    "lives": LIVES,
    "image": doodle_right_img
})
```

Les valeurs `1000` utilisées pour `x` et `y` sont volontairement incorrectes.

Votre première tâche consiste à remplacer ces deux valeurs afin que le Doodle apparaisse à sa position de départ prévue par le jeu.

**Contraintes à respecter :**

- utilisez les variables `DOODLE_START_X` et `DOODLE_START_Y` déjà définies dans `config.py` ;
- ne remplacez pas ces variables par des nombres écrits directement dans le dictionnaire.

À la fin de cette partie, le Doodle doit apparaître au-dessus de la plateforme verte de départ lorsque vous exécutez `main.py`.

### 1.2 : Déplacement horizontal et changement de direction

Dans le fichier `game.py`, complétez la fonction `move_doodle()`.

Le joueur doit pouvoir déplacer le Doodle horizontalement avec :

- `K_LEFT` ou `K_a` : déplacement vers la gauche ;
- `K_RIGHT` ou `K_d` : déplacement vers la droite.

Pour connaître l'état des touches du clavier, utilisez :

```python
keys = pygame.key.get_pressed()
```

**Détails à respecter :**

- la position `x` doit être modifiée de `DOODLE_SPEED` pixels ;
- lorsque le Doodle se déplace vers la gauche :
  - `direction` doit devenir `"left"` ;
  - `image` doit devenir `doodle_left_img` ;
- lorsqu'il se déplace vers la droite :
  - `direction` doit devenir `"right"` ;
  - `image` doit devenir `doodle_right_img`.

#### Passage d'un bord à l'autre de l'écran

Contrairement à Frogger, le Doodle ne doit pas être bloqué aux limites gauche et droite. Il doit réapparaître de l'autre côté de l'écran lorsqu'il dépasse un bord.

Complétez donc également la partie **Screen Wrap** de `move_doodle()` :

- si le Doodle dépasse suffisamment le bord gauche, il doit réapparaître à droite ;
- s'il dépasse suffisamment le bord droit, il doit réapparaître à gauche.

Utilisez `SCREEN_WIDTH` et `DOODLE_WIDTH` pour effectuer les calculs plutôt que des valeurs numériques fixes.

## PARTIE 2 : Les plateformes 🟩🟦

### 2.1 : Création du dictionnaire d'une plateforme

Dans `platforms.py`, la fonction `create_platform(x, y, platform_type)` doit retourner un dictionnaire décrivant une plateforme.

Une version temporaire du dictionnaire est fournie et permet de représenter une plateforme verte. Vous devez la **généraliser** afin que l'argument `platform_type` détermine réellement les propriétés de la plateforme.

Le dictionnaire retourné doit contenir les clés suivantes :

```python
{
    "x": ...,
    "y": ...,
    "type": ...,
    "image": ...,
    "vx": ...,
    "active": ...,
    "width": ...,
    "height": ...
}
```

**Contraintes à respecter :**

- `x` et `y` doivent être obtenus à partir des paramètres reçus par la fonction ;
- `type` doit correspondre à `platform_type` ;
- l'image doit être récupérée dans le dictionnaire `platform_images` ;
- une plateforme bleue possède une vitesse horizontale égale à `MOVING_PLATFORM_SPEED` ;
- les autres plateformes ont une vitesse horizontale nulle ;
- toutes les plateformes sont actives lors de leur création ;
- la largeur normale correspond à `PLATFORM_SIZE[0]` ;
- une plateforme `spring` est 10 pixels plus haute que les autres.

N'écrivez pas directement la valeur de la vitesse des plateformes bleues : utilisez la constante déjà définie dans `config.py`.

### 2.2 : Choix aléatoire et génération initiale

Cette partie comporte deux étapes liées.

#### Fonction `choose_platform_type()`

Dans `platforms.py`, complétez la fonction :

```python
choose_platform_type(green_probability, blue_probability, spring_probability)
```

Elle doit retourner l'une des chaînes `"green"`, `"blue"`, `"spring"` ou `"brown"`.

Les trois paramètres représentent les probabilités des trois premiers types. La probabilité restante correspond automatiquement aux plateformes marron.

Par exemple, si les probabilités reçues sont `0.65`, `0.17` et `0.10`, alors la répartition attendue est :

- verte : 65 % ;
- bleue : 17 % ;
- ressort : 10 % ;
- marron : 8 %.

Utilisez `random.random()`. Attention : pour distinguer les quatre intervalles, les seuils doivent être **cumulatifs**. Vous devez déterminer vous-même ces seuils à partir des probabilités reçues en paramètres.

#### Fonction `generate_initial_platforms()`

Dans `window.py`, une première plateforme verte est déjà ajoutée sous le Doodle. Vous devez compléter la génération du reste des plateformes jusqu'au haut de l'écran.

Pour chaque nouvelle plateforme :

- sa coordonnée `x` doit être choisie aléatoirement tout en gardant la plateforme dans la fenêtre ;
- sa coordonnée `y` dépend de la plateforme précédente et d'un espacement aléatoire compris entre `MIN_PLATFORM_GAP` et `MAX_PLATFORM_GAP` ;
- son type doit être obtenu en appelant `choose_platform_type()` avec les probabilités **65 % / 17 % / 10 % / 8 %** ;
- la plateforme doit être créée avec `create_platform()` puis ajoutée à `PLATFORMS`.

La génération doit s'arrêter lorsque la partie supérieure de la fenêtre est suffisamment remplie. La variable `current_y`, déjà calculée pour vous, doit servir à contrôler cette progression.

> [!TIP]
> Ne recopiez pas la logique de sélection des types dans `window.py`. L'objectif de `choose_platform_type()` est précisément de centraliser cette logique pour pouvoir la réutiliser plus tard.

### 2.3 : Déplacement des plateformes bleues

Dans `game.py`, complétez la fonction `move_platforms()`.

Les plateformes bleues actives doivent se déplacer horizontalement selon leur clé `vx`. Lorsqu'une plateforme atteint un bord de la fenêtre, son mouvement doit changer de direction afin qu'elle reste visible.

Les plateformes vertes, marron et à ressort ne doivent pas être déplacées par cette fonction.

Vous devez déterminer les conditions permettant de détecter le bord gauche et le bord droit à partir de `x`, `width` et `SCREEN_WIDTH`.

## PARTIE 3 : Physique, rebonds et progression 🚀

### 3.1 : Application de la gravité

Dans `game.py`, complétez la fonction `apply_gravity()`.

La clé `vel_y` représente la vitesse verticale du Doodle. À chaque image du jeu, la gravité modifie d'abord cette vitesse, puis la vitesse obtenue est utilisée pour modifier la position verticale.

Rappelez-vous que, dans le repère Pygame, une vitesse verticale positive correspond à un déplacement vers le bas et une vitesse négative à un déplacement vers le haut.

### 3.2 : Détection des collisions avec les plateformes

Complétez `check_platform_collisions()` dans `game.py`.

Cette partie est l'une des principales difficultés du projet. Un chevauchement entre le rectangle du Doodle et celui d'une plateforme **ne suffit pas** à conclure à un atterrissage : le Doodle peut traverser une plateforme pendant qu'il monte.

Votre fonction doit donc vérifier simultanément que :

- le Doodle est en phase de descente ;
- la plateforme considérée est active ;
- le rectangle du Doodle chevauche celui de la plateforme ;
- le Doodle arrive sur le **dessus** de cette plateforme, et non par-dessous ou par le côté.

La fonction `rects_collide(r1, r2)` fournie à la fin de `game.py` permet de tester le chevauchement de deux rectangles représentés par :

```python
(x, y, largeur, hauteur)
```

Pour distinguer un véritable atterrissage d'un simple chevauchement, comparez la position actuelle des pieds du Doodle à leur position approximative lors de l'image précédente. Vous pouvez retrouver cette position précédente grâce à `vel_y`. Une tolérance de **14 pixels** est acceptée afin de rendre la collision plus robuste, mais c'est à vous de construire la condition correspondante.

Lorsqu'un atterrissage est détecté :

- une plateforme verte ou bleue applique `JUMP_VELOCITY` ;
- une plateforme `spring` applique `SPRING_JUMP_VELOCITY` ;
- une plateforme `brown` applique `JUMP_VELOCITY`, puis devient inactive.

Un seul rebond doit être traité par appel de la fonction.

### 3.3 : Défilement de la caméra et score

Complétez `scroll_camera()` dans `game.py`.

Lorsque le Doodle monte au-dessus de `CAMERA_SCROLL_THRESHOLD`, il doit rester visuellement à cette hauteur. Pour donner l'impression qu'il continue son ascension, c'est alors l'ensemble des plateformes qui se déplace vers le bas.

Vous devez déterminer la distance de défilement nécessaire et l'utiliser pour :

- repositionner le Doodle au seuil de caméra ;
- déplacer toutes les plateformes de la même distance ;
- augmenter le score selon la distance verticale parcourue ;
- mettre à jour `high_score` lorsque nécessaire ;
- retirer les plateformes ayant entièrement quitté la zone utile sous l'écran ;
- demander la génération de nouvelles plateformes au-dessus de l'écran.

> [!IMPORTANT]
> `PLATFORMS` est une liste globale également utilisée dans d'autres fichiers. Veillez à modifier son contenu sans casser les références existantes vers cette liste.

### 3.4 : Génération continue de nouvelles plateformes

Complétez `generate_new_platforms()` dans `game.py`.

Lorsque la caméra défile, des plateformes disparaissent progressivement sous la fenêtre. Cette fonction doit donc prolonger le niveau vers le haut.

Votre algorithme doit :

- gérer correctement le cas où `PLATFORMS` est vide ;
- repérer la plateforme actuellement la plus haute ;
- ajouter autant de nouvelles plateformes que nécessaire au-dessus de celle-ci ;
- conserver les espacements aléatoires compris entre `MIN_PLATFORM_GAP` et `MAX_PLATFORM_GAP` ;
- choisir une position horizontale valide ;
- utiliser `choose_platform_type()` au lieu de recopier l'algorithme de sélection aléatoire.

Pour cette génération en cours de partie, utilisez les probabilités suivantes :

- verte : 55 % ;
- bleue : 20 % ;
- ressort : 13 % ;
- marron : 12 %.

Comparez cette fonction avec `generate_initial_platforms()`. Les deux problèmes se ressemblent, mais ne partent pas du même état du jeu. Vous devez identifier quelles informations déjà présentes dans `PLATFORMS` peuvent servir de point de départ.

## Fonctionnalités déjà fournies

Vous n'avez pas à programmer les éléments suivants :

- le chargement et le redimensionnement des images ;
- la création de la fenêtre Pygame ;
- l'affichage du fond, du Doodle, des plateformes et du score ;
- la détection générale du chevauchement de deux rectangles (`rects_collide`) ;
- la détection de la chute sous l'écran (`check_game_over`) ;
- l'écran `GAME OVER` ;
- la touche `R` permettant de recommencer ;
- la boucle principale contenue dans `main.py`.

# Directives pour la remise

Pour remettre votre travail, créez un fichier ZIP nommé `NOM1_PRENOM1_NOM2_PRENOM2-PR01.zip`, où `NOM1` est votre nom de famille et `PRENOM1` votre prénom, et `NOM2` `PRENOM2` ceux de votre binôme.

Le fichier ZIP doit contenir le dossier `2026A-PR01` complet avec les fichiers Python et le dossier `assets/`.

Ne modifiez pas les noms des fichiers ni la structure du projet.

# Barème de correction

Le barème proposé est le suivant :

| **Partie** | **Tâche** | **Points** |
|---|---|---:|
| **PARTIE 1 : Le Doodle 🦘** |  | **/4** |
| 1.1 | Position initiale correcte avec les constantes demandées | 1 |
| 1.2 | Déplacement gauche/droite avec les touches demandées | 1.5 |
| 1.2 | Mise à jour cohérente de la direction et de l'image | 0.5 |
| 1.2 | Passage correct d'un bord de l'écran à l'autre | 1 |
| **PARTIE 2 : Les plateformes 🟩🟦** |  | **/6** |
| 2.1 | Dictionnaire retourné cohérent avec `platform_type` | 1 |
| 2.1 | Utilisation correcte de `MOVING_PLATFORM_SPEED`, de l'image et de la hauteur selon le type | 0.5 |
| 2.2 | `choose_platform_type()` : sélection aléatoire et seuils cumulatifs corrects | 1 |
| 2.2 | Génération initiale : boucle, positions horizontales et espacements valides | 1.25 |
| 2.2 | Génération initiale : création, ajout et probabilités des plateformes | 0.75 |
| 2.3 | Déplacement des plateformes bleues actives | 0.75 |
| 2.3 | Détection des deux bords et inversion correcte du mouvement | 0.75 |
| **PARTIE 3 : Physique, rebonds et progression 🚀** |  | **/10** |
| 3.1 | Mise à jour correcte de la vitesse verticale et de la position | 0.5 |
| 3.2 | Collision traitée uniquement pendant la descente et sur les plateformes actives | 0.75 |
| 3.2 | Construction et vérification correctes des rectangles | 0.75 |
| 3.2 | Détection correcte d'un atterrissage par le dessus | 1.25 |
| 3.2 | Rebonds corrects selon les quatre types de plateformes | 0.75 |
| 3.3 | Calcul du défilement et repositionnement cohérent du Doodle | 1 |
| 3.3 | Mise à jour du score et du meilleur score | 0.75 |
| 3.3 | Déplacement et suppression correcte des plateformes | 1.25 |
| 3.4 | Identification correcte du point de départ de la génération | 0.75 |
| 3.4 | Génération continue avec positions et espacements valides | 1 |
| 3.4 | Réutilisation de `choose_platform_type()` avec les probabilités demandées | 1.25 |
| **Total** |  | **/20** |
