import networkx as nx
import matplotlib.pyplot as plt

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

def visualiser_graphe(graph, coloration):
    # create an graph 
    G = nx.Graph(graph)

    # draw the graph with coloration 
    pos = nx.spring_layout(G)  # Layout pour la disposition des noeuds

    # get list color for each vertex
    colors = [coloration[node] for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1000, font_size=10, font_color='black', font_weight='bold')
    plt.show()


# Exemple of adjacence list
graph_matrice_adjacence = [
    [0, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 0]
]

# Convert the matrice to an adjacence list 
graph_liste_adjacence = matrice_adjacence_to_liste_adjacence(graph_matrice_adjacence)


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

# To write the coloration 

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
    # Liste d'indices de couleurs
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


# To write the coloration 

"""
coloration_welsh = WelshPowell(graph_liste_adjacence)
visualiser_graphe(graph_liste_adjacence, coloration_welsh)
"""

# -----------------------------------------------------------------------------
# Python3 program to implement backtracking
# algorithm for graph coloring 
# -----------------------------------------------------------------------------

def is_safe(vertex, color, graph, color_assignment):
    for i in range(len(graph)):
        if graph[vertex][i] == 1 and color_assignment[i] == color:
            return False
    return True

def graph_coloring_backtracking_util(graph, num_colors, color_assignment, vertex):
    if vertex == len(graph):
        return True

    for color in range(1, num_colors+1):
        if is_safe(vertex, color, graph, color_assignment):
            color_assignment[vertex] = color
            if graph_coloring_backtracking_util(graph, num_colors, color_assignment, vertex + 1):
                return True
            color_assignment[vertex] = 0

def graph_coloring_backtracking(graph, num_colors):
    color_assignment = [0] * len(graph)
    if not graph_coloring_backtracking_util(graph, num_colors, color_assignment, 0):
        return None
    return color_assignment

"""
num_colors = 3
coloration_backtracking = graph_coloring_backtracking(graph_matrice_adjacence, num_colors)

visualiser_graphe(graph_liste_adjacence, coloration_backtracking)
"""