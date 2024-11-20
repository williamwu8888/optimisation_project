import math
import numpy as np
import matplotlib.pyplot as plt

### Données initiales

V_s = 790           # Tension nominale (V)
R_s = 33e-3         # Résistance interne (ohms)
rho_LAC = 131e-6    # Résistance linéique LAC (ohms/m)
rho_rail = 18e-6    # Résistance linéique rail (ohms/m)
M = 70_000          # Masse du train (kg)
P_bord = 35_000     # Consommation électrique à bord (W)
rendement = 0.8     # Rendement global

### Les fonctions de calcul

def calcul_resistance(distance, rho):
    '''Calcul de la résistance entre sous-station et train.'''
    return rho * distance

def puissance_mecanique(vitesse, pente=0):
    '''Calcule de la puissance mécanique nécessaire.'''
    # Coefficients de résistance
    A = 780
    A_t = 6.4
    B = 0
    B_t = 0.14
    C = 0.3634
    C_t = 0

    F_resistance = (A + A_t * M / 1_000) + (B + B_t * M / 1_000) * vitesse + (C + C_t * M / 1_000) * vitesse**2
    F_gravite = M * 9.81 * np.sin(np.radians(pente))
    F = F_resistance + F_gravite
    return F * vitesse

def puissance_electrique(p_mec):
    '''Calcul de la puissance électrique à partir de la mécanique.'''
    return p_mec / rendement

def tension_train(P_train, R_eq):
    '''Calcul de Tension aux bornes du train.
    Résolution de l'équation : V^2 - V*V_s + R_eq*P_train = 0.'''
    delta = V_s**2 - 4 * R_eq * P_train
    # Pour un delta négatif pas de solution
    if delta < 0:
        return None
    V1 = (V_s + np.sqrt(delta)) / 2
    V2 = (V_s - np.sqrt(delta)) / 2
    return max(V1, V2)

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
    R1 = calcul_resistance(distance_1, rho_LAC + rho_rail)
    R2 = calcul_resistance(distance_2, rho_LAC + rho_rail)
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

    for i in range(len(temps) - 1):  # On boucle sur les indices des pas de temps
        # Résistances
        R1, R2 = calcul_resistances(positions[i], longueur_ligne)
        R_eq = R1 + R_s + R2

        # Calcul de la vitesse (différence entre deux positions successives)
        dt = temps[i + 1] - temps[i]
        if dt == 0:
            vitesse = 0
        else:
            vitesse = (positions[i + 1] - positions[i]) / dt

        # Puissance mécanique et électrique
        P_mec = puissance_mecanique(vitesse)
        P_train = puissance_electrique(P_mec) + P_bord

        # Tension aux bornes du train
        V_train = tension_train(P_train, R_eq)
        if V_train is None:
            raise ValueError(f"Pas de solution pour la tension au temps {temps[i]}")

        # Stocker les résultats
        tensions.append(V_train)
        courants.append(P_train / V_train)

    return tensions, courants

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
        tuple: Listes des tensions et des courants.
    """
    tensions = []
    courants = []
    energie_batterie = capacite_batterie  # État initial de la batterie (plein)

    for i in range(len(temps) - 1):  # On boucle sur les pas de temps
        # Résistances
        R1, R2 = calcul_resistances(positions[i], longueur_ligne)
        R_eq = R1 + R_s + R2

        # Calcul de la vitesse
        dt = temps[i + 1] - temps[i]
        if dt == 0:
            vitesse = 0
        else:
            vitesse = (positions[i + 1] - positions[i]) / dt

        # Puissance mécanique et électrique
        P_mec = puissance_mecanique(vitesse)
        P_train = puissance_electrique(P_mec) + P_bord

        # Gestion de la batterie
        if P_train > 0:  # Le train consomme de la puissance
            if energie_batterie > 0:  # Si la batterie n'est pas vide
                # Fournir de l'énergie par la batterie
                P_batt = min(energie_batterie / dt, P_train)  # Puissance max que la batterie peut fournir
                P_train -= P_batt  # Réduction de la puissance demandée à la LAC
                energie_batterie -= P_batt * dt  # Mise à jour de l'énergie de la batterie

        elif P_train < 0:  # Le train freine (P_train négative)
            if energie_batterie < capacite_batterie:  # Si la batterie n'est pas pleine
                # Stocker l'énergie de freinage
                P_batt = min(-P_train, (capacite_batterie - energie_batterie) / dt)
                energie_batterie += P_batt * dt  # Mise à jour de l'énergie stockée
                P_train += P_batt  # Réduction de l'énergie dissipée dans le rhéostat

        # Tension aux bornes du train
        V_train = tension_train(P_train, R_eq)
        if V_train is None:
            raise ValueError(f"Pas de solution pour la tension au temps {temps[i]}")

        # Stocker les résultats
        tensions.append(V_train)
        courants.append(P_train / V_train)

    return tensions, courants

### Affichage des résultats

def tracer_resultats(temps, tensions, label):
    # Aligner les dimensions : supprimer le dernier point de `temps`
    temps = temps[:len(tensions)]  # Troncature de `temps`
    plt.plot(temps, tensions, label=label)
    plt.xlabel("Temps (s)")
    plt.ylabel("Tension aux bornes du train (V)")
    plt.title("Évolution de la tension au cours du temps")
    plt.legend()
    plt.grid()

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

    # 2. Définir les paramètres de la ligne de tramway
    longueur_ligne = max(positions)  # Longueur totale de la ligne (en mètres)
    capacite_batterie = 5e6  # Capacité de la batterie (en joules, ici 5 MJ)

    print(f"Longueur de la ligne : {longueur_ligne} m")
    print(f"Simulation avec une capacité de batterie de {capacite_batterie / 1e6} MJ")

    # 3. Simulation sans batterie
    print("Simulation sans batterie...")
    try:
        tensions_sans_batterie, courants_sans_batterie = simulation_sans_batterie(temps, positions, longueur_ligne)
    except Exception as e:
        print(f"Erreur pendant la simulation sans batterie : {e}")
        return

    # 4. Simulation avec batterie
    print("Simulation avec batterie...")
    try:
        tensions_avec_batterie, courants_avec_batterie = simulation_avec_batterie(temps, positions, longueur_ligne, capacite_batterie)
    except Exception as e:
        print(f"Erreur pendant la simulation avec batterie : {e}")
        return

    # 5. Affichage des résultats
    print("Affichage des résultats...")
    plt.figure(figsize=(10, 6))
    tracer_resultats(temps, tensions_sans_batterie, label="Sans Batterie")
    tracer_resultats(temps, tensions_avec_batterie, label="Avec Batterie")
    plt.show()

if __name__ == "__main__":
    main()

