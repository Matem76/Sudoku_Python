


#%%Exercice 1 

def dijkstra(G, s):
    def init_dijkstra(n):
        d = [float('inf')] * n
        pi = [0] * n
        d[s] = 0
        return d, pi

    def get_shortest_path(pi, v):
        path = []
        while v != s:
            path.insert(0, v)
            v = pi[v]
        path.insert(0, s)
        return path

    d, pi = init_dijkstra(len(G))

    S = set(range(len(G)))

    while S:
        u = min(S, key=lambda v: d[v])
        S.remove(u)
        for v, w in enumerate(G[u]):
            if w > 0 and d[u] + w < d[v]:
                d[v] = d[u] + w
                pi[v] = u

    shortest_paths = [get_shortest_path(pi, v) for v in range(len(G)) if v != s]

    return d, pi, shortest_paths

# Graphe
G = [[0, 10, 0, 5, 0],
     [0, 0, 1, 2, 0],
     [0, 0, 0, 0, 4],
     [0, 3, 9, 0, 2],
     [7, 0, 6, 0, 0]]

s = 0  # Sommet source

d, pi, shortest_paths = dijkstra(G, s)

print("Distances minimales:", d)
print("Prédécesseurs:", pi)

for v, path in enumerate(shortest_paths, start=1): # probleme affichaghe ici pour s > 0 
    print(f"Chemin le plus court de {s} à {v}: {path}")
    
    
 #%%Exercice 2 
 
 def init_bellman_ford(n, source):
    d = [float('inf')] * n
    pi = [0] * n
    d[source] = 0
    return d, pi

def relax(u, v, d, pi, graph):
    if d[u] + graph[u][v] < d[v]:
        d[v] = d[u] + graph[u][v]
        pi[v] = u

def bellman_ford(graph, source):
    n = len(graph)
    d, pi = init_bellman_ford(n, source)

    for k in range(n - 1):
        for u in range(n):
            for v in range(n):
                if graph[u][v] != float('inf'):
                    relax(u, v, d, pi, graph)

    return d, pi

def get_shortest_path(pi, v, source):
    path = []
    while v != source:
        path.insert(0, v)
        v = pi[v]
    path.insert(0, source)
    return path

# Exemple d'utilisation
G = [[0, 10, float('inf'), 5, float('inf')],
     [float('inf'), 0, 1, 2, float('inf')],
     [float('inf'), float('inf'), 0, float('inf'), 4],
     [float('inf'), 3, 9, 0, 2],
     [7, float('inf'), 6, float('inf'), 0]]

s = 0  # Sommet source

d, pi = bellman_ford(G, s)
 

print("Distances minimales:", d)
print("Prédécesseurs:", pi)

for v in range(len(G)):
    if v != s:
        path = get_shortest_path(pi, v, s)
        print(f"Chemin le plus court de {s} à {v}: {path}") 
        
        
#%%Exercice 3 & 4 

def PileVide(P):
    return len(P) == 0

def Empiler(P, x):
    P.append(x)

def Depiler(P):
    if not PileVide(P):
        return P.pop()
    else:
        raise Exception("La pile est vide, impossible de dépiler.")
     
        
"""
P = []  # Création d'une pile vide

print(PileVide(P))  # Output: True

Empiler(P, 1)
Empiler(P, 2)
Empiler(P, 3)

print(PileVide(P))  # Output: False

print(Depiler(P))  # Output: 3
print(Depiler(P))  # Output: 2
print(Depiler(P))  # Output: 1

print(PileVide(P))  # Output: True  """


def CopierColler(P):
    # Crée une copie de la pile P en utilisant Empiler et Depiler
    Q = []
    while not PileVide(P):
        x = Depiler(P)
        Empiler(Q, x)
        Empiler(P, x)  # Restaure P à son état initial
    return Q

def Inverser(P):
    Q = CopierColler(P)
    return Q

def Rotation(P):
    Q = CopierColler(P)
    x = Depiler(Q)
    Empiler(Q, x)
    return Q

def DernierPremier(P):
    Q = CopierColler(P)
    dernier = Depiler(Q)
    premier = Depiler(Q)
    Empiler(Q, dernier)
    Empiler(Q, premier)
    return Q
"""
P = []  # Création d'une pile vide

Empiler(P, 1)
Empiler(P, 2)
Empiler(P, 3)

Q = CopierColler(P)
print(Q)  # Output: [3, 2, 1]

R = Inverser(P)
print(R)  # Output: [1, 2, 3]

S = Rotation(P)
print(S)  # Output: [3, 1, 2]

T = DernierPremier(P)
print(T)  # Output: [2, 1, 3] """

#%% Exercice 5 

def Insertion(T, x):
    # Insère l'élément x dans le tas T
    index = len(T)
    T[index] = x
    
    while index > 1 and T[index] < T[index // 2]:
        T[index], T[index // 2] = T[index // 2], T[index]
        index //= 2

def Suppression(T):
    # Supprime et retourne l'élément le plus petit du tas T
    if len(T) == 1:
        return None
    
    min_value = T[1]
    last_value = T.popitem()[1]
    
    if len(T) > 1:
        T[1] = last_value
        index = 1
        while True:
            left_child = 2 * index
            right_child = 2 * index + 1
            smallest = index
            if left_child < len(T) and T[left_child] < T[smallest]:
                smallest = left_child
            if right_child < len(T) and T[right_child] < T[smallest]:
                smallest = right_child
            if smallest == index:
                break
            T[index], T[smallest] = T[smallest], T[index]
            index = smallest
    
    return min_value
"""
Tas = {1: 5, 2: 8, 3: 3, 4: 9, 5: 1}

Insertion(Tas, 2)
print(Tas)  # Output: {1: 2, 2: 5, 3: 3, 4: 9, 5: 1, 6: 8}

element_supprime = Suppression(Tas)
print(element_supprime)  # Output: 1
print(Tas)  # Output: {1: 2, 2: 5, 3: 3, 4: 9, 5: 8}"""


