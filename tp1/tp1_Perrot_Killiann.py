import numpy as np


#%%exercice 1

import numpy as np

M = np.array([[0, 0, 0, 1, 1], [1, 0, 0, 0, 1], [
             1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
print(M)

"""
def successeurs(matrice: np.ndarray, sommet: int):
    return np.nonzero(matrice[sommet])[0]

print(successeurs(M, 2))



def predecesseurs(matrice: np.ndarray, sommet: int):
    matrice_transposee = np.transpose(matrice)
    return np.nonzero(matrice_transposee[sommet])[0]


 print(predecesseurs(M,0))
"""


def successeurs(matrice: np.ndarray, sommet: int):
    if sommet <= 0:
        raise ValueError()
    succ = np.nonzero(matrice[sommet - 1])[0]
    return [s + 1 for s in succ]


def predecesseurs(matrice: np.ndarray, sommet: int):
    if sommet <= 0:
        raise ValueError()
    matrice_transposee = np.transpose(matrice)
    pred = np.nonzero(matrice_transposee[sommet - 1])[0]
    return [p + 1 for p in pred]


print(successeurs(M, 5))
print(predecesseurs(M, 1))


def estchemin(matrice: np.ndarray, arg_list: list):
    for i in range (len(arg_list)-1):
        if arg_list[i+1] not in successeurs(matrice,arg_list[i]):
            return False
    return True

print(estchemin(M,[2,1,5,3,1]))
print(estchemin(M,[4,3,5,1,4]))

def versionNO( matrice : np.ndarray):
    mat =  np.logical_or(matrice, np.transpose(matrice))
    return mat.astype(int)

print(versionNO(M))

#%%exercice2

graphe = {
    1: [2, 3, 5],
    2: [1, 5, 9],
    3: [1, 4, 5, 7],
    4: [3, 7],
    5: [1, 2, 3, 6, 8, 9],
    6: [5, 8, 9],
    7: [3, 4, 8],
    8: [5, 6, 7, 9],
    9: [2, 5, 6, 8]
}

def voisins(dico: dict, sommet : int):
    if (sommet <= 0):
        raise ValueError()
    return graphe[sommet]

print(voisins(graphe,1))

def degre(dico : dict , sommet : int):
    if(sommet <= 0):
        raise ValueError()
    return len(graphe[sommet])

print(degre(graphe,1))

def deMaTL ( matrice : np.ndarray):
    dico = {}
    for i,ligne in enumerate(matrice):
        dico[i+1] = []
        for j in range(len(ligne)):
            if(ligne[j] == 1):
                dico[i+1].append(j+1)
    return dico

print(deMaTL(M))

import numpy as np

def deTLaM(dico : dict):
    nb_noeuds = len(dico)

    M = np.zeros((nb_noeuds, nb_noeuds))

    for noeud, voisins in dico.items():
        for voisin in voisins:
            M[noeud-1, voisin-1] = 1
            M[voisin-1, noeud-1] = 1

    return M

print(deTLaM(graphe))


#%%exercice 3


def graphematrix(nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        contenu = fichier.readlines()

    contenu = [ligne.strip().split() for ligne in contenu]

    matrice = np.array([[bool(int(element)) for element in ligne] for ligne                         in contenu])

    return matrice

print(graphematrix('/home/killiann/algo_des_graphes/matrice.txt'))

def grapheajd (nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        dico = {}
        contenu = fichier.readlines()
        contenu = [ligne.strip().split(':') for ligne in contenu]
        for i,ligne in enumerate(contenu):
            a = ligne[0]
            if( 1 < len(ligne)):
                dico[a] = ligne[1]
        return dico

print(grapheajd('/home/killiann/algo_des_graphes/matrice2.txt'))

#%%exercice4

import matplotlib.pyplot as plt
import numpy as np

def draw_non_oriented_graph(matrix):
    n = len(matrix)
    G = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if matrix[i, j] == 1:
                G[i, j] = G[j, i] = 1

    x = [1, 3, 4]  # Coordonnées x des points
    y = [3, 1, 5]  # Coordonnées y des points

    plt.figure(figsize=(5, 5))

    for i in range(n):
        for j in range(i+1, n):
            if G[i, j] == 1:
                plt.plot([x[i], x[j]], [y[i], y[j]], color='r', linestyle='-', marker='o')

    plt.axis([-1, 5, -1, 6])
    plt.show()


M = np.array([[0, 0, 0, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0]])

draw_non_oriented_graph(M)
