"Perrot killiann"



#%% Exercice 1 :

def est_valide(tab, ligne, colonne):
    for i in range(ligne):
        if tab[i] == colonne or abs(tab[i] - colonne) == abs(i - ligne):
            return False
    return True

def HuitReines(tab, ligne):
    if ligne == 8:
        for i in range(8):
            print(tab[i]+1,end = " ")
        print()
        print("------------------------------")
    else:
        for colonne in range (8):
            if est_valide(tab,ligne,colonne):
                tab[ligne] = colonne
                HuitReines(tab,ligne+1)

def resolvant():
    tab = [-1] * 8
    HuitReines(tab,0)

resolvant()

#%% Exercice 2 :


def est_valide(plateau, x, y):
    """
    Vérifie si la position (x, y) est valide sur le plateau.
    """
    n = len(plateau)
    return 0 <= x < n and 0 <= y < n and plateau[x][y] == -1

def parcours_cavalier(n):
    """
    Résout le problème du parcours du cavalier sur un échiquier de taille n x n.
    Retourne la liste des positions initiales possibles.
    """
    def backtracking(plateau, x, y, mouvement):
        nonlocal trouve
        if trouve:
            return

        if mouvement == n**2:
            solutions.append([row[:] for row in plateau])
            trouve = True
            return

        for i in range(8):
            new_x = x + mouvements_x[i]
            new_y = y + mouvements_y[i]
            if est_valide(plateau, new_x, new_y):
                plateau[new_x][new_y] = mouvement
                backtracking(plateau, new_x, new_y, mouvement + 1)
                plateau[new_x][new_y] = -1

    mouvements_x = [2, 1, -1, -2, -2, -1, 1, 2]
    mouvements_y = [1, 2, 2, 1, -1, -2, -2, -1]
    solutions = []
    trouve = False

    for i in range(n):
        for j in range(n):
            plateau = [[-1 for _ in range(n)] for _ in range(n)]
            plateau[i][j] = 0
            backtracking(plateau, i, j, 1)

    return solutions

resultats = parcours_cavalier(5)
for solution in resultats:
    for row in solution:
        print(row)
    print()
