import math
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# -----------------------------------------------------------------------------
# Graph tools
# -----------------------------------------------------------------------------


def matrice_adjacence_to_liste_adjacence(matrice):
    liste_adjacence = {}
    n = len(matrice)
    for i in range(n):
        voisins = [j for j in range(n) if matrice[i][j] == 1]
        liste_adjacence[i] = voisins
    return liste_adjacence


def liste_adjacence_to_matrice_adjacence(liste_adjacence):
    # Trouver le nombre de sommets dans le graphe
    nb_sommets = (
        max(
            max(liste_adjacence.keys(), default=-1),
            max(
                [max(v) if isinstance(v, list) else -
                 1 for v in liste_adjacence.keys()],
                default=-1,
            ),
        )
        + 1
    )

    matrice_adjacence = [[0] * nb_sommets for _ in range(nb_sommets)]

    for sommet, voisins in liste_adjacence.items():
        for voisin in voisins:
            matrice_adjacence[sommet][voisin] = 1

    return matrice_adjacence


def visualiser_graphe(graph, coloration):
    # create an graph
    G = nx.Graph(graph)

    # draw the graph with coloration
    pos = nx.spring_layout(G)

    # get list color for each vertex
    colors = [coloration[node] for node in G.nodes()]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=1000,
        font_size=10,
        font_color="black",
        font_weight="bold",
    )
    plt.show()


# Exemple of adjacence list
graph_matrice_adjacence = [
    [0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0],
]

# Convert the matrice to an adjacence list
graph_liste_adjacence = matrice_adjacence_to_liste_adjacence(
    graph_matrice_adjacence)

# -----------------------------------------------------------------------------
# Python3 program to implement greedy
# algorithm for graph coloring
# -----------------------------------------------------------------------------


def addEdge(adj, v, w):
    adj[v].append(w)
    adj[w].append(v)
    return adj


def greedyColoring(adj, V):
    result = [-1] * V
    result[0] = 0
    available = [False] * V

    for u in range(1, V):
        for i in adj[u]:
            if result[i] != -1:
                available[result[i]] = True

        cr = 0
        while cr < V:
            if available[cr] == False:
                break
            cr += 1

        result[u] = cr

        for i in adj[u]:
            if result[i] != -1:
                available[result[i]] = False

    return result


"""
# using greedy algorithm to colorize the graph
coloration_glouton = greedyColoring(graph_liste_adjacence, len(graph_liste_adjacence))

# Visualize the graph 
visualiser_graphe(graph_liste_adjacence, coloration_glouton)
"""

# -----------------------------------------------------------------------------
# Python3 program to implement welsh-Powell
# algorithm for graph coloring
# -----------------------------------------------------------------------------


def MAdjacence(G):
    n = len(G)
    Ma = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j in G[i]:
                Ma[i][j] = 1
    return Ma


def Sommets(G):
    return list(G.keys())


def Voisinage(G, sommet):
    return G[sommet]


def WelshPowell(G):
    # colors list
    couleurs = [0, 1, 2, 3, 4, 5]
    Ma = MAdjacence(G)
    sommets = Sommets(G)
    degres, result = [], []
    nb = 0

    for sommet in sommets:
        degres.append(len(Voisinage(G, sommet)))
        result.append(0)

    degres, sommets = zip(*sorted(zip(degres, sommets), reverse=True))

    for i in range(len(degres)):
        if result[i] == 0:
            nb += 1
            result[i] = couleurs[nb]
            for j in range(len(degres)):
                if Ma[sommets[i]][sommets[j]] == 0 and result[j] == 0:
                    for k in range(len(degres)):
                        passe = 0
                        if Ma[sommets[j]][sommets[k]] == 1 and result[k] == result[i]:
                            passe = 1
                            break
                    if passe == 0:
                        result[j] = result[i]

    d, i = {}, 0
    for sommet in sommets:
        d[sommet] = result[i]
        i += 1

    return d


"""
coloration_welsh = WelshPowell(graph_liste_adjacence)
visualiser_graphe(graph_liste_adjacence, coloration_welsh)
"""

# -----------------------------------------------------------------------------
# Python3 program to implement backtracking
# algorithm for graph coloring
# -----------------------------------------------------------------------------


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    # A utility function to check
    # if the current color assignment
    # is safe for vertex v
    def isSafe(self, v, colour, c):
        for i in range(self.V):
            if (self.graph[v][i] == 1 and colour[i] == c) or (
                self.graph[i][v] == 1 and colour[i] == c
            ):
                return False
        return True

    def add_edge(self, u, v):
        self.graph[u][v] = 1
        self.graph[v][u] = 1

    # A recursive utility function to solve m
    # coloring problem
    def graphColourUtil(self, m, colour, v):
        if v == self.V:
            return True

        for c in range(1, m + 1):
            if self.isSafe(v, colour, c) == True:
                colour[v] = c
                if self.graphColourUtil(m, colour, v + 1) == True:
                    return True
                colour[v] = 0

    def graphColouring(self):
        m = 1
        colour = [0] * self.V
        while not self.graphColourUtil(m, colour, 0):
            m += 1
            colour = [0] * self.V
        return colour


class Graph2:
    def __init__(self, vertices, adjacency_list, initial_coloring):
        self.V = vertices
        self.adjacency_list = adjacency_list
        self.initial_coloring = initial_coloring

    def isSafe(self, v, colour, c):
        for neighbor in self.adjacency_list[v]:
            if colour[neighbor] == c:
                return False
        return True

    def add_edge(self, u, v):
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)

    def graphColourUtil(self, colour, v):
        if v == self.V:
            return True

        if colour[v] != 0:
            return self.graphColourUtil(colour, v + 1)

        for c in range(1, max(colour) + 1):
            if self.isSafe(v, colour, c):
                colour[v] = c
                if self.graphColourUtil(colour, v + 1):
                    return True
                colour[v] = 0

    def graphColouring(self):
        colour = self.initial_coloring
        self.graphColourUtil(colour, 0)
        return colour

"""
g = Graph(len(graph_matrice_adjacence))
g.graph = graph_matrice_adjacence
visualiser_graphe(matrice_adjacence_to_liste_adjacence(g.graph),g.graphColouring())
"""
# -----------------------------------------------------------------------------
# To design and implement in Python a coloring algorithm for the case
# of a dynamic graph where the number of vertices and edges evolve over time.
# -----------------------------------------------------------------------------


def remove_node(graph_liste_adjacence, noeud):
    if noeud in graph_liste_adjacence:
        voisins = graph_liste_adjacence[noeud]
        del graph_liste_adjacence[noeud]

        for voisin in voisins:
            if voisin in graph_liste_adjacence:
                graph_liste_adjacence[voisin] = [
                    x for x in graph_liste_adjacence[voisin] if x != noeud
                ]

        return graph_liste_adjacence
    else:
        print(f"The node {noeud} does not exist in the adjacency list.")
        return None


def modify_graph_dynamically(liste_adjacence):
    n = len(liste_adjacence)
    operation = random.randint(0, 4)

    if operation == 0:  # add a node
        new_neighbors = random.sample(range(n), random.randint(1, n // 2))
        filtered_neighbors = [
            neighbor for neighbor in new_neighbors if neighbor in liste_adjacence
        ]
        i = random.randint(0, 2 * n - 1)
        while i in liste_adjacence:
            i = random.randint(0, 2 * n - 1)
        if i in filtered_neighbors:
            filtered_neighbors.remove(i)
        print(f"Adding the Node  {i} with his neighbors {filtered_neighbors}.")
        liste_adjacence[i] = filtered_neighbors
        for neighbor in filtered_neighbors:
            liste_adjacence[neighbor].append(i)

        n = len(liste_adjacence)

    elif operation == 1:  # remove a node
        if n > 1:
            node_to_remove = random.randint(
                0, list(liste_adjacence.keys())[-1])
        while node_to_remove not in liste_adjacence:
            node_to_remove = random.randint(
                0, list(liste_adjacence.keys())[-1])
        liste_adjacence = remove_node(liste_adjacence, node_to_remove)
        print(f"Remove the node {node_to_remove} and his links .")

    elif operation == 2:  # add an edge
        i = random.randint(0, list(liste_adjacence.keys())[-1])
        while i not in liste_adjacence:
            i = random.randint(0, list(liste_adjacence.keys())[-1])
        j = random.randint(0, list(liste_adjacence.keys())[-1])
        while j not in liste_adjacence:
            j = random.randint(0, list(liste_adjacence.keys())[-1])
        if i != j:
            if j not in liste_adjacence[i]:
                print(f"Adding link between the node {i} and the node {j}.")
                liste_adjacence[i].append(j)
                liste_adjacence[j].append(i)
            else:
                print(
                    f"cant add the link because those random numbers {i},{j} cant be used\n"
                )
        else:
            print(
                f"cant add the link because those random numbers {i},{j} cant be used\n"
            )

    elif operation == 3:  # remove an edge
        i = random.randint(0, list(liste_adjacence.keys())[-1])
        while i not in liste_adjacence:
            i = random.randint(0, list(liste_adjacence.keys())[-1])
        if liste_adjacence[i]:
            j = random.choice(liste_adjacence[i])
            print(
                f"Removing node link between the node  {i} and the node {j}.")
            liste_adjacence[i].remove(j)
            liste_adjacence[j].remove(i)

    else:  # modify an edge
        i = random.randint(0, list(liste_adjacence.keys())[-1])
        while i not in liste_adjacence:
            i = random.randint(0, list(liste_adjacence.keys())[-1])
        if liste_adjacence[i]:
            j = random.choice(liste_adjacence[i])
            if j != i:
                if j not in liste_adjacence[i]:
                    print(
                        f"Adding link between the node {i} and the node {j}.")
                    liste_adjacence[i].append(j)
                    liste_adjacence[j].append(i)

                else:
                    print(
                        f"Removing node link between the node {i} and the node {j}.")
                    liste_adjacence[i].remove(j)
                    liste_adjacence[j].remove(i)

    return liste_adjacence


def observe_graph_evolution(initial_graph, num_iterations):
    current_graph = initial_graph.copy()
    graph_matrice_adjacence = liste_adjacence_to_matrice_adjacence(
        current_graph)
    g = Graph(len(graph_matrice_adjacence))
    g.graph = graph_matrice_adjacence
    visualiser_graphe(current_graph, g.graphColouring())
    for _ in range(num_iterations):
        current_graph = modify_graph_dynamically(current_graph)
        if current_graph == None:
            break
        graph_matrice_adjacence = liste_adjacence_to_matrice_adjacence(
            current_graph)
        g = Graph(len(graph_matrice_adjacence))
        g.graph = graph_matrice_adjacence
        visualiser_graphe(current_graph, g.graphColouring())


"""
observe_graph_evolution(graph_liste_adjacence, 4)
"""
# -----------------------------------------------------------------------------
# To use the implemented algorithms to develop an application that
# allows coloring a user-provided graph and solving a Sudoku grid using
# a coloring algorithm."
# ----------------------------------------------------------------------------


def get_sudoku_grid():
    print("Please enter the 9x9 Sudoku grid (use '0' for empty cells):")
    grid = []
    for _ in range(9):
        row = list(map(int, input().split()))
        grid.append(row)
    return grid

def display_9x9_sudoku(grid):
    print("+-------+-------+-------+")
    for i in range(9):
        for j in range(9):
            if j % 3 == 0:
                print("|", end=" ")
            if grid[i][j] == 0:
                print(".", end=" ")
            else:
                print(grid[i][j], end=" ")
            if j == 8:
                print("|")
        if (i + 1) % 3 == 0:
            print("+-------+-------+-------+")


def tableau_vers_matrice(tableau):
    taille = int(math.sqrt(len(tableau)))

    matrice = [[0 for _ in range(taille)] for _ in range(taille)]

    # Remplissage de la matrice en utilisant les valeurs du tableau
    for i in range(taille):
        for j in range(taille):
            matrice[i][j] = tableau[i * taille + j]

    return matrice

sudoku9 = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

def sudoku_to_adjacency_list(sudoku):
    n = len(sudoku)
    adjacency_list = {i: set() for i in range(n * n)}

    for i in range(n):
        for j in range(n):
            # Same row
            for k in range(n):
                if k != j:
                    adjacency_list[n * i + j].add(n * i + k)

            # Same column
            for k in range(n):
                if k != i:
                    adjacency_list[n * i + j].add(n * k + j)

            # Same 3x3 sub-grid
            start_row, start_col = 3 * (i // 3), 3 * (j // 3)
            for x in range(3):
                for y in range(3):
                    if start_row + x != i or start_col + y != j:
                        adjacency_list[n * i + j].add(
                            n * (start_row + x) + start_col + y
                        )

    # Convert sets to lists for consistency with the original function
    adjacency_list = {key: list(value)
                      for key, value in adjacency_list.items()}

    return adjacency_list


def sudoku_to_value_list(sudoku):
    n = len(sudoku)
    value_list = [0] * (n * n)

    for i in range(n):
        for j in range(n):
            value_list[n * i + j] = sudoku[i][j]

    return value_list


value_list = sudoku_to_value_list(sudoku9)

print("#----------------------#")
print("Original Sudoku:")
display_9x9_sudoku(sudoku9)

k =sudoku_to_adjacency_list(sudoku9)
g = Graph2(len(k), k, value_list)
print("#----------------------#")
print("Solved Sudoku:")
display_9x9_sudoku(tableau_vers_matrice(g.graphColouring()))
