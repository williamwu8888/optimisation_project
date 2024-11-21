import math
import numpy as np

import matplotlib.pyplot as plt

### Données initiales

V_s = 790           # Tension nominale (V)
R_s = 33e-3         # Résistance interne (ohms)
rho_lac = 131e-6    # Résistance linéique LAC (ohms/m)
rho_rail = 18e-6    # Résistance linéique rail (ohms/m)
M = 70_000          # Masse du train (kg)
P_bord = 35_000     # Consommation électrique à bord (W)
rendement = 0.8     # Rendement global

### Les fonctions de calcul

def calcul_resistance(distance, rho):
    '''Calcul de la résistance entre sous-station et train.'''
    return rho * distance

def puissance_mecanique(vitesse, acceleration, alpha):
    '''Calcule de la puissance mécanique nécessaire.'''
    # Coefficients de résistance
    A = 780
    A_t = 6.4
    B = 0
    B_t = 0.14
    C = 0.3634
    C_t = 0

    F_resistance = (A + A_t * M / 1_000) + (B + B_t * M / 1_000) * vitesse + (C + C_t * M / 1_000) * vitesse**2
    F_gravite = M * 9.81 * np.sin(np.radians(alpha))
    F_motrice = F_resistance + F_gravite + (M * acceleration)
    return F_motrice * vitesse

def puissance_electrique(p_mec):
    '''Calcul de la puissance électrique à partir de la mécanique.'''
    return p_mec * rendement

def tension_train(P_train, R_eq):
    '''Calcul de Tension aux bornes du train.
    Résolution de l'équation : V^2 - V*V_s + R_eq*P_train = 0.'''
    delta = V_s**2 - 4 * R_eq * P_train
    # Pour un delta négatif pas de solution
    if delta < 0:
        return None
    vitesse_1 = (V_s + np.sqrt(delta)) / 2
    vitesse_2 = (V_s - np.sqrt(delta)) / 2
    return max(vitesse_1, vitesse_2)

### Train en déplacement

def charger_donnees_train(fichier):
    """Charge les données de déplacement du train à partir d'un fichier texte.
    Arguments: 
        fichier (str): Chemin vers le fichier `marche.txt`.
    Renvoie:
        tuple (2 tableaux numpy): les temps et les positions."""
    try:
        data = np.loadtxt(fichier, delimiter='\t', dtype=float)
        temps = data[:, 0]  # Première colonne : temps
        positions = data[:, 1]  # Deuxième colonne : positions
        return temps, positions
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None, None

def calcul_resistances(position, longueur_ligne):
    '''Calcul des résistances lorsque le train est en déplacement.'''
    distance_1 = position
    distance_2 = longueur_ligne - position
    R1 = calcul_resistance(distance_1, rho_lac + rho_rail) + R_s
    R2 = calcul_resistance(distance_2, rho_lac + rho_rail) + R_s
    return R1, R2

### Simulation sans batterie

def simulation_sans_batterie(temps, positions, longueur_ligne):
    """
    Fait la simulation du système sans batterie.
    
    Arguments :
        temps (array): Tableau des temps.
        positions (array): Tableau des positions.
        longueur_ligne (float): Longueur totale de la ligne.
    
    Retourne :
        tuple: Listes des tensions et des courants.
    """
    tensions = []
    courants = []
    puissances = []
    vitesses = []
    accelerations = []
    resistances_eq = []

    for i in range(len(temps) - 2):  # On boucle sur les indices des pas de temps
        # Résistances
        R1, R2 = calcul_resistances(positions[i], longueur_ligne)
        R_eq = (R1*R2) / (R1+R2)
        resistances_eq.append(R_eq)

        # Calcul de la vitesse (différence entre deux positions successives)
        dt1 = temps[i + 1] - temps[i]
        dt2 = temps[i + 2] - temps[i + 1]
        
        vitesse_1 = (positions[i+1] - positions[i]) / dt1 if dt1 != 0 else 0
        vitesse_2 = (positions[i+2] - positions[i+1]) / dt2 if dt2 != 0 else 0

        acceleration = (vitesse_2-vitesse_1) / (temps[i + 2] - temps[i + 1]) if i > 0 else 0

        # Puissance mécanique et électrique
        P_mec = puissance_mecanique(vitesse_1, acceleration, alpha = 0)
        P_train = puissance_electrique(P_mec) + P_bord

        # Tension aux bornes du train
        V_train = tension_train(P_train, R_eq)
        if V_train is None:
            raise ValueError(f"Pas de solution pour la tension au temps {temps[i]}")

        # Stocker les résultats
        vitesses.append(vitesse_1*3.6)          # Conversion en km/h
        accelerations.append(acceleration)
        tensions.append(V_train)
        puissances.append(P_train/(10**6))      # Conversion en MW
        courants.append(P_train / V_train)

    return resistances_eq, vitesses, accelerations, tensions, puissances, courants

### Simulation avec batterie

def simulation_avec_batterie(temps, positions, longueur_ligne, capacite_batterie):
    """
    Simule le système avec batterie.

    Arguments :
        temps (array): Tableau des temps.
        positions (array): Tableau des positions.
        longueur_ligne (float): Longueur totale de la ligne.
        capacite_batterie (float): Capacité maximale de la batterie (en joules).

    Retourne :
        tuple: Listes des tensions, courants, vitesses, accélérations, puissances et résistances.
    """
    tensions = []
    courants = []
    puissances = []
    vitesses = []
    accelerations = []
    resistances_eq = []

    energie_batterie = capacite_batterie  # État initial de la batterie (plein)
    
    for i in range(len(temps) - 2):  # On boucle sur les indices des pas de temps
        # Résistances
        R1, R2 = calcul_resistances(positions[i], longueur_ligne)
        R_eq = (R1 * R2) / (R1 + R2)
        resistances_eq.append(R_eq)

        # Calcul de la vitesse (différence entre deux positions successives)
        dt1 = temps[i + 1] - temps[i]
        dt2 = temps[i + 2] - temps[i + 1]
        
        vitesse_1 = (positions[i+1] - positions[i]) / dt1 if dt1 != 0 else 0
        vitesse_2 = (positions[i+2] - positions[i+1]) / dt2 if dt2 != 0 else 0

        # Calcul de l'accélération
        acceleration = (vitesse_2 - vitesse_1) / (temps[i + 2] - temps[i + 1]) if dt2 != 0 else 0

        # Puissance mécanique et électrique
        P_mec = puissance_mecanique(vitesse_1, acceleration, alpha=0)
        P_train = puissance_electrique(P_mec) + P_bord

        # Gestion de la batterie
        if P_train > 0:  # Le train consomme de la puissance
            if energie_batterie > 0:  # Si la batterie n'est pas vide
                # Fournir de l'énergie par la batterie
                P_batt = min(energie_batterie / dt1, P_train)  # Puissance max que la batterie peut fournir
                P_train -= P_batt  # Réduction de la puissance demandée à la LAC
                energie_batterie -= P_batt * dt1  # Mise à jour de l'énergie de la batterie
        elif P_train < 0:  # Le train freine (P_train négative)
            if energie_batterie < capacite_batterie:  # Si la batterie n'est pas pleine
                # Stocker l'énergie de freinage
                P_batt = min(-P_train, (capacite_batterie - energie_batterie) / dt1)
                energie_batterie += P_batt * dt1  # Mise à jour de l'énergie stockée
                P_train += P_batt  # Réduction de l'énergie dissipée dans le rhéostat

        # Tension aux bornes du train
        V_train = tension_train(P_train, R_eq)
        if V_train is None:
            raise ValueError(f"Pas de solution pour la tension au temps {temps[i]}")

        # Stocker les résultats
        vitesses.append(vitesse_1 * 3.6)  # Conversion en km/h
        accelerations.append(acceleration)
        tensions.append(V_train)
        puissances.append(P_train / (10**6))  # Conversion en MW
        courants.append(P_train / V_train)

    return resistances_eq, vitesses, accelerations, tensions, puissances, courants

### Affichage des résultats

def tracer_position(temps, positions, label):
    temps = temps[:len(positions)]
    plt.plot(temps, positions, label=label)
    plt.xlabel("Temps (s)")
    plt.ylabel("Position (m)")
    plt.title("Évolution de la position du train au cours du temps")
    plt.legend()
    plt.grid()

def tracer_resistance(temps, resistances, label):
    temps = temps[:len(resistances)]  
    plt.plot(temps, resistances, label=label)
    plt.xlabel("Temps (s)")
    plt.ylabel("Résistance équivalente (Ohms)")
    plt.title("Évolution de la résistance équivalente au cours du temps")
    plt.legend()
    plt.grid()

def tracer_vitesse(temps, vitesses, label):
    temps = temps[:len(vitesses)]  
    plt.plot(temps, vitesses, label=label)
    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse (km/h)")
    plt.title("Évolution de la vitesse au cours du temps")
    plt.legend()
    plt.grid()

def tracer_acceleration(temps, accelerations, label):
    temps = temps[:len(accelerations)]  
    plt.plot(temps, accelerations, label=label)
    plt.xlabel("Temps (s)")
    plt.ylabel("Accélération (m/s²)")
    plt.title("Évolution de l'accélération au cours du temps")
    plt.legend()
    plt.grid()

def tracer_tension(temps, tensions, label):
    temps = temps[:len(tensions)]  
    plt.plot(temps, tensions, label=label)
    plt.xlabel("Temps (s)")
    plt.ylabel("Tension aux bornes du train (V)")
    plt.title("Évolution de la tension au cours du temps")
    plt.legend()
    plt.grid()

def tracer_puissance(temps, puissances, label):
    temps = temps[:len(puissances)]  
    plt.plot(temps, puissances, label=label)
    plt.xlabel("Temps (s)")
    plt.ylabel("Puissance (MW)")
    plt.title("Évolution de la puissance au cours du temps")
    plt.legend()
    plt.grid()

def tracer_courant(temps, courants, label):
    temps = temps[:len(courants)]  
    plt.plot(temps, courants, label=label)
    plt.xlabel("Temps (s)")
    plt.ylabel("Courant (A)")
    plt.title("Évolution du courant au cours du temps")
    plt.legend()
    plt.grid()

def tout_tracer(temps, positions, vitesses, accelerations, puissances, tensions, courants, resistances_eq):
    plt.figure(figsize=(10, 12))
    plt.subplots_adjust(hspace=0.7)
    plt.subplot(4, 1, 1)
    tracer_position(temps, positions, label="Position")
    plt.subplot(4, 1, 2)
    tracer_vitesse(temps, vitesses, label="Vitesse")
    plt.subplot(4, 1, 3)
    tracer_acceleration(temps, accelerations, label="Accélération")
    plt.subplot(4, 1, 4)
    tracer_puissance(temps, puissances, label="Puissance mécanique")

    plt.figure(figsize=(10, 12))
    plt.subplots_adjust(hspace=0.7)
    plt.subplot(3, 1, 1)
    tracer_tension(temps, tensions, label="Tension")
    plt.subplot(3, 1, 2)
    tracer_courant(temps, courants, label="Courant")
    plt.subplot(3, 1, 3)
    tracer_resistance(temps, resistances_eq, label="Résistance")
    plt.show()

### Programme main

# Importer les fonctions définies précédemment (dans d'autres fichiers si nécessaire)
# Par exemple, si elles sont dans des fichiers séparés :
# from simulation import simulation_sans_batterie, simulation_avec_batterie
# from utils import charger_donnees_train, tracer_resultats

def main():
    # 1. Chargement des données
    fichier_donnees = "marche.txt"
    temps, positions = charger_donnees_train(fichier_donnees)
    
    if temps is None or positions is None:
        print("Erreur : Impossible de charger les données du fichier.")
        return

    # Limiter l'échantillonnage à 150 points
    temps = temps[:150]
    positions = positions[:150]

    # 2. Définir les paramètres de la ligne de tramway
    longueur_ligne = max(positions)  # Longueur totale de la ligne (en mètres)
    capacite_batterie = 5e6  # Capacité de la batterie (en joules, ici 5 MJ)

    print(f"Longueur de la ligne : {longueur_ligne} m")
    print(f"Simulation avec une capacité de batterie de {capacite_batterie / 1e6} MJ")

    # 3. Simulation sans batterie
    print("Simulation sans batterie...")
    try:
        resistances_eq_sb, vitesses_sb, accelerations_sb, tensions_sb, puissances_sb, courants_sb = simulation_sans_batterie(temps, positions, longueur_ligne)
    except Exception as e:
        print(f"Erreur pendant la simulation sans batterie : {e}")
        return
    
    print("Affichage des résultats...")
    tout_tracer(temps, positions, vitesses_sb, accelerations_sb, puissances_sb, tensions_sb, courants_sb, resistances_eq_sb)

    # 4. Simulation avec batterie
    print("Simulation avec batterie...")
    try:
        resistances_eq_ab, vitesses_ab, accelerations_ab, tensions_ab, puissances_ab, courants_ab = simulation_avec_batterie(temps, positions, longueur_ligne, capacite_batterie)
    except Exception as e:
        print(f"Erreur pendant la simulation sans batterie : {e}")
        return
    
    print("Affichage des résultats...")
    tout_tracer(temps, positions, vitesses_ab, accelerations_ab, puissances_ab, tensions_ab, courants_ab, resistances_eq_ab)

if __name__ == "__main__":
    main()
