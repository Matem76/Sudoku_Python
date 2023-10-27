import networkx as nx
import matplotlib.pyplot as plt
import random 


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
    pos = nx.spring_layout(G)  

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

print(graph_liste_adjacence)
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

def is_safe(vertex, color, liste_adjacence, color_assignment):
    neighbors = liste_adjacence.get(vertex, [])
    for neighbor in neighbors:
        if color_assignment.get(neighbor, None) == color:
            return False
    return True

def graph_coloring_backtracking(liste_adjacence, num_colors):
    if liste_adjacence is None:
        return None

    color_assignment = {vertex: 0 for vertex in liste_adjacence.keys()}
    vertices = list(liste_adjacence.keys())
    vertices.sort(key=lambda x: len(liste_adjacence[x]), reverse=True)
    def graph_coloring_backtracking_util(vertex):
        if vertex == len(vertices):
            return True

        for color in range(1, num_colors+1):
            if is_safe(vertices[vertex], color, liste_adjacence, color_assignment):
                color_assignment[vertices[vertex]] = color
                if graph_coloring_backtracking_util(vertex + 1):
                    return True
                color_assignment[vertices[vertex]] = 0

        return False

    if not graph_coloring_backtracking_util(0):
        return None
    return color_assignment


"""
num_colors = 3
coloration_backtracking = graph_coloring_backtracking(graph_liste_adjacence, num_colors)

visualiser_graphe(graph_liste_adjacence, coloration_backtracking)
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
                graph_liste_adjacence[voisin] = [x for x in graph_liste_adjacence[voisin] if x != noeud]

        return graph_liste_adjacence
    else:
        print(f"The node {noeud} does not exist in the adjacency list.")
        return None


def modify_graph_dynamically(liste_adjacence):
    n = len(liste_adjacence)
    operation = random.randint(1, 1)

    if operation == 0:  # add a node
        new_neighbors = random.sample(range(n), random.randint(1, n//2))
        liste_adjacence[n] = new_neighbors
        for neighbor in new_neighbors:
            liste_adjacence[neighbor].append(n)
        print(f"Adding the Node  {n} with his neighbors {new_neighbors}.")
        n = len(liste_adjacence)  # Mettez à jour n après l'ajout du nœud

    elif operation == 1:  # remove a node
        if n > 1:
            node_to_remove = random.randint(0, n-1)
            while node_to_remove not in liste_adjacence:
                node_to_remove = random.randint(0, n-1)
            liste_adjacence = remove_node(liste_adjacence, node_to_remove)
            print(f"Remove the node {node_to_remove} and his links .")
    
    elif operation == 2:  # add an edge
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if j not in liste_adjacence[i]:
            liste_adjacence[i].append(j)
            liste_adjacence[j].append(i)
            print(f"Adding link between the node {i} and the node {j}.")
    
    elif operation == 3:  # remove an edge
        i = random.randint(0, n-1)
        if liste_adjacence[i]:
            j = random.choice(liste_adjacence[i])
            liste_adjacence[i].remove(j)
            liste_adjacence[j].remove(i)
            print(f"Removing node link between the node  {i} and the node {j}.")
    
    else:  # modify an edge
        i = random.randint(0, n-1)
        if liste_adjacence[i]:
            j = random.choice(liste_adjacence[i])
            if j != i:
                if j not in liste_adjacence[i]:
                    liste_adjacence[i].append(j)
                    liste_adjacence[j].append(i)
                    print(f"Adding link between the node {i} and the node {j}.")
                else:
                    liste_adjacence[i].remove(j)
                    liste_adjacence[j].remove(i)
                    print(f"Removing node link between the node {i} and the node {j}.")

    return liste_adjacence



def observe_graph_evolution(initial_graph, num_iterations, num_colors):
    current_graph = initial_graph.copy()
    coloration_backtracking = graph_coloring_backtracking(current_graph, num_colors)
    visualiser_graphe(current_graph, coloration_backtracking)
    for _ in range(num_iterations):
        current_graph = modify_graph_dynamically(current_graph)
        if(current_graph == None):
            break
        coloration_backtracking = graph_coloring_backtracking(current_graph, num_colors)
        visualiser_graphe(current_graph, coloration_backtracking)



num_colors = 7
observe_graph_evolution(graph_liste_adjacence, 4, num_colors)


# -----------------------------------------------------------------------------
#To use the implemented algorithms to develop an application that
# allows coloring a user-provided graph and solving a Sudoku grid using
# a coloring algorithm."
# ----------------------------------------------------------------------------

"""
def get_sudoku_grid():
    print("Please enter the 4x4 Sudoku grid (use '0' for empty cells):")
    grid = []
    for _ in range(4):
        row = list(map(int, input().split()))
        grid.append(row)
    return grid

def display_sudoku(grid):
    print("+-----+-----+")
    for i in range(4):
        for j in range(4):
            if j % 2 == 0:
                print("|", end=" ")
            if grid[i][j] == 0:
                print(".", end=" ")
            else:
                print(grid[i][j], end=" ")
        print("|")
        if (i + 1) % 2 == 0:
            print("+-----+-----+")


sudoku_grid = get_sudoku_grid()

display_sudoku(sudoku_grid)"""

