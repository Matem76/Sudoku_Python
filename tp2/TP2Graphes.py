# Perrot Killiann



#%% Exercice 1

m = [[0,0,0],[0,0,0],[0,0,0]]

def roy_warshall(graphe):
    n = len(graphe)
    AC = [[False for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):

            AC[i][j] = (int)(graphe[i][j] or (i == j))

    for k in range(n):
        for i in range(n):
            for j in range(n):
                AC[i][j] = AC[i][j] or (AC[i][k] and AC[k][j])

    return AC

print(roy_warshall(m))
#%% Exercice 2

m = [[1,0,0],[1,1,0],[0,0,0]]
import numpy as np

def roy_warshall(graphe):
    n = len(graphe)
    p = np.array(m)
    AC = [[False for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):

            AC[i][j] = (int)(graphe[i][j] or (i == j))

    for k in range(n):
        for i in range(n):
            for j in range(n):
                AC[i][j] = AC[i][j] or (AC[i][k] and AC[k][j])

    return AC,np.transpose(p)

print(roy_warshall(m))

#%% Exercice 3

def sans_circuits(graphe):
    n = len(graphe)
    AC = [[False for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            AC[i][j] = graphe[i][j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                AC[i][j] = AC[i][j] or (AC[i][k] and AC[k][j])

    return all(not AC[i][i] for i in range(n))


#%% Exercice 4

def est_connexe(graphe):
    n = len(graphe)
    AC = [[False for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            AC[i][j] = graphe[i][j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                AC[i][j] = AC[i][j] or (AC[i][k] and AC[k][j])

    return all(all(AC[i][j] or AC[j][i] for j in range(n)) for i in range(n))



#%% Exercice 5
def noyau(graphe):
    def dfs_visite(s, graphe, visite, pile):
        visite.add(s)
        for voisin in graphe[s]:
            if voisin not in visite:
                dfs_visite(voisin, graphe, visite, pile)
        pile.append(s)

    def dfs_marquer(s, graphe, composante):
        visite.add(s)
        composante.append(s)
        for voisin in graphe[s]:
            if voisin not in visite:
                dfs_marquer(voisin, graphe, composante)

    n = len(graphe)
    visite = set()
    pile = []

    for sommet in graphe:
        if sommet not in visite:
            dfs_visite(sommet, graphe, visite, pile)

    graphe_transpose = {sommet: [] for sommet in graphe}
    for sommet in graphe:
        for voisin in graphe[sommet]:
            graphe_transpose[voisin].append(sommet)

    visite = set()
    noyau = []

    while pile:
        sommet = pile.pop()
        if sommet not in visite:
            composante = []
            dfs_marquer(sommet, graphe_transpose, composante)
            noyau.append(composante)

    return noyau

graphe = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A', 'D'],
    'D': ['E'],
    'E': ['F'],
    'F': []
}

resultat = noyau(graphe)
print(resultat)

#%% Exercice 6

graphe_allumettes = {
    (3, 3): [(2, 3), (3, 2), (3, 1)],
    (2, 3): [(1, 3), (2, 2), (2, 1)],
    (3, 2): [(2, 2), (3, 1), (3, 0)],
    (3, 1): [(2, 1), (3, 0), (3, -1)],
    (3, 0): [(2, 0), (3, -1), (3, -2)],
    (2, 2): [(1, 2), (2, 1), (2, 0)],
    (2, 1): [(1, 1), (2, 0), (2, -1)],
    (2, 0): [(1, 0), (2, -1), (2, -2)],
    (1, 3): [(1, 2), (1, 1), (1, 0)],
    (1, 2): [(1, 1), (1, 0), (1, -1)],
    (1, 1): [(1, 0), (1, -1), (1, -2)],
    (1, 0): [(1, -1), (1, -2), (1, -3)],
    (1, -1): [],
    (1, -2): [],
    (1, -3): [],
    (2, -1): [],
    (2, -2): [],
    (3, -1): [],
    (3, -2): []
}



strategie_gagnante = noyau(graphe_allumettes)

print("Noyau du jeu des allumettes:")
print(strategie_gagnante)

def jeu_allumettes():
    etat_actuel = (3, 3)
    joueur_courant = 1

    while True:
        print(f'État actuel : {etat_actuel}')
        if etat_actuel not in graphe_allumettes:
            if joueur_courant == 1:
                print('Vous avez gagné !')
            else:
                print('L\'ordinateur a gagné !')
            break

        if joueur_courant == 1:
            mouvements_possibles = graphe_allumettes[etat_actuel]
            print(f'Mouvements possibles : {mouvements_possibles}')
            choix = input('Choisissez un mouvement (au format (x, y)) : ')
            choix = eval(choix)  # Convertir l'entrée en tuple
        else:
            for composante in strategie_gagnante:
                if etat_actuel in composante:
                    choix = composante[0]  # L'ordinateur choisit le premier état du noyau
                    break

        etat_actuel = choix
        joueur_courant = 3 - joueur_courant  # Changer de joueur (1 -> 2, 2 -> 1)

# Exécuter le jeu
jeu_allumettes()

