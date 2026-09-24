
# ======================== game.py ========================

import pygame
import random
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, GRAVITY, JUMP_VELOCITY, SPRING_JUMP_VELOCITY,
    DOODLE_SPEED, DOODLE_WIDTH, DOODLE_HEIGHT, PLATFORM_WIDTH,
    MIN_PLATFORM_GAP, MAX_PLATFORM_GAP, CAMERA_SCROLL_THRESHOLD,
    PLATFORMS, doodle_dict, DOODLE_START_X, DOODLE_START_Y, LIVES
)
from platforms import create_platform, choose_platform_type
from doodle import doodle_left_img, doodle_right_img
from window import generate_initial_platforms


# ======================== PARTIE 3.1 ========================
def apply_gravity():
    """
    Applique la gravité au Doodle en augmentant progressivement sa vitesse verticale (vel_y).
    Met à jour la position verticale (y) du Doodle.
    """
    # TODO : Mettez à jour la vitesse verticale puis la position verticale
    # du Doodle à partir de GRAVITY.
    doodle_dict["vel_y"]=check_platform_collisions()
    
    doodle_dict["y"]+=doodle_dict["vel_y"]

    return

# ===========================================================


# ======================== PARTIE 1.2 ========================
def move_doodle():
    """
    Gère le déplacement horizontal du Doodle selon les touches pressées (Flèches ou A/D).
    Implémente le passage fluide d'un côté de l'écran à l'autre (Screen Wrap).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        print("key")
        doodle_dict["x"]-=DOODLE_SPEED


    if keys[pygame.K_RIGHT]:
        print("key")
        doodle_dict["x"]+=DOODLE_SPEED
    

    if(doodle_dict["x"]<0):
        doodle_dict["x"]=SCREEN_WIDTH
    if(doodle_dict["x"]>SCREEN_WIDTH):
        doodle_dict["x"]=0




    # TODO : Gérez les déplacements gauche/droite et mettez à jour
    # simultanément la direction et l'image du Doodle.



    # TODO : Implémentez le Screen Wrap pour qu'une partie du Doodle puisse
    # sortir d'un côté avant de réapparaître de l'autre.
    # N'utilisez pas de dimensions numériques écrites directement.



    return

# ===========================================================


# ======================== PARTIE 2.3 ========================
def move_platforms():
    """
    Déplace horizontalement les plateformes mobiles ("blue").
    Fait rebondir les plateformes lorsqu'elles atteignent les bords de la fenêtre.
    """
    # TODO : Parcourez les plateformes et gérez le déplacement des plateformes
    # bleues encore actives. Elles doivent rester dans la fenêtre en inversant
    # leur vitesse lorsqu'elles atteignent un bord.
    for el in PLATFORMS:
        if(el["type"]=="blue"):
            
            el["x"]+=el["vx"]
            if(el["x"]>SCREEN_WIDTH-el["width"]):
                el["x"] = SCREEN_WIDTH - el["width"]
                el["vx"]*=-1
            if(el["x"]<0):
                el["x"] =0
                el["vx"]*=-1
        
    return

# ===========================================================


# ======================== PARTIE 3.2 ========================
def check_platform_collisions():
    """
    Détecte si le Doodle atterrit sur une plateforme.
    Le rebond ne se produit QUE lorsque le Doodle descend (vel_y > 0)
    et qu'il arrive sur le dessus d'une plateforme.
    """
    # TODO : Implémentez la détection d'un atterrissage.
    #
    # Contraintes :
    # - aucun rebond pendant la montée ;
    # - ignorer les plateformes inactives ;
    # - utiliser rects_collide(...) pour le chevauchement des rectangles ;
    # - un simple chevauchement ne suffit pas : le Doodle doit arriver par
    #   le dessus de la plateforme. Pour le vérifier, comparez la position
    #   actuelle de ses pieds à leur position approximative à l'image
    #   précédente à l'aide de vel_y. Une tolérance de 14 pixels est permise ;
    # - spring : SPRING_JUMP_VELOCITY ;
    # - brown : JUMP_VELOCITY puis désactivation de la plateforme ;
    # - green/blue : JUMP_VELOCITY.
    
    velo=doodle_dict["vel_y"]+GRAVITY
    del_i=-1
    if(doodle_dict["vel_y"] > 0):
        i=0
        for el in PLATFORMS:
            if(rects_collide([doodle_dict["x"],doodle_dict["y"],DOODLE_WIDTH,DOODLE_HEIGHT],[el["x"],el["y"],el["width"],el["height"]])):
                if(doodle_dict["y"] + DOODLE_HEIGHT<=el["y"] + 14):
                    
                    if(el["type"]=="green" or el["type"]=="blue"):
                        velo=JUMP_VELOCITY
                    elif(el["type"]=="spring"):
                        velo=SPRING_JUMP_VELOCITY
                    elif(el["type"]=="brown"):
                        velo=JUMP_VELOCITY
                        del_i=i
            i+=1
    if(del_i!=-1):
        del PLATFORMS[del_i]
        
       
    return velo

# ===========================================================


# ======================== PARTIE 3.3 ========================
def scroll_camera():
    """
    Fait défiler le monde lorsque le Doodle dépasse CAMERA_SCROLL_THRESHOLD.
    Met à jour le score et maintient les plateformes visibles.
    """
    # TODO : Lorsque le Doodle dépasse le seuil de caméra, il doit rester
    # visuellement au seuil pendant que les plateformes sont déplacées vers
    # le bas de la même distance.
    #
    # Le score doit représenter la distance verticale ainsi parcourue et le
    # meilleur score doit être mis à jour. Les plateformes sorties sous
    # l'écran doivent être retirées, puis de nouvelles plateformes générées.
   
    if(doodle_dict["y"]<CAMERA_SCROLL_THRESHOLD):

        doodle_dict["score"]+=abs(CAMERA_SCROLL_THRESHOLD-doodle_dict["y"])
        if(doodle_dict["score"]>doodle_dict["high_score"]):
            doodle_dict["high_score"]=doodle_dict["score"]
        scroll=abs(CAMERA_SCROLL_THRESHOLD-doodle_dict["y"])
        doodle_dict["y"] = CAMERA_SCROLL_THRESHOLD
        print(PLATFORMS[0])
        for i in range(len(PLATFORMS)):
            
            PLATFORMS[i]["y"]+=scroll


    return

# ===========================================================


# ======================== PARTIE 3.4 ========================
def generate_new_platforms():
    """
    Génère de nouvelles plateformes au-dessus du haut de l'écran pour maintenir
    un flux continu lorsque la caméra défile.
    """
    # TODO : Complétez cette fonction en vous inspirant de la logique de
    # génération initiale, sans la recopier inutilement.
    #
    # Vous devrez partir de la plateforme actuellement la plus haute et
    # continuer à ajouter des plateformes tant que nécessaire. Utilisez
    # choose_platform_type(...) avec les probabilités indiquées dans le README.
    
    if PLATFORMS:
        position = min(p["y"] for p in PLATFORMS)
    else:
        position = SCREEN_HEIGHT

    while position > -MAX_PLATFORM_GAP:
        position -= random.randint(MIN_PLATFORM_GAP, MAX_PLATFORM_GAP)
        current_x = random.randint(0, SCREEN_WIDTH - PLATFORM_WIDTH)
        
        PLATFORMS.append(create_platform(current_x, position))
       
    return

# ===========================================================


def check_game_over():
    """
    Vérifie si le Doodle tombe sous le bas de l'écran.
    Si oui, réduit les vies.
    Retourne True si la partie est terminée.
    """
    if doodle_dict["y"] > SCREEN_HEIGHT:
        doodle_dict["lives"] -= 1
        return True
    return False


def restart_game():
    """
    Réinitialise la partie : position du Doodle, vitesse, score et plateformes.
    """
    doodle_dict["x"] = DOODLE_START_X
    doodle_dict["y"] = DOODLE_START_Y
    doodle_dict["vel_y"] = 0.0
    doodle_dict["direction"] = "right"
    doodle_dict["image"] = doodle_right_img
    doodle_dict["score"] = 0
    doodle_dict["lives"] = LIVES

    generate_initial_platforms()


def rects_collide(r1, r2):
    """
    Vérifie si deux rectangles (x, y, largeur, hauteur) se chevauchent.
    Cette fonction est fournie et ne doit pas être modifiée.
    """
    return not (
        r1[0] + r1[2] <= r2[0] or r1[0] >= r2[0] + r2[2] or
        r1[1] + r1[3] <= r2[1] or r1[1] >= r2[1] + r2[3]
    )
