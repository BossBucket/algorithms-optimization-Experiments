"""Modulo TSP compartido por los ejercicios 3, 4 y 5."""
import itertools
import math
import random


def random_instance(n: int, rng: random.Random, size: float = 10.0):
    return [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]


def tour_length(points, tour) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]])
               for i in range(len(tour)))


def random_tour(n: int, rng: random.Random):
    t = list(range(n))
    rng.shuffle(t)
    return t


def nearest_neighbor(points, start: int):
    """Vecino mas cercano: desde 'start', ir siempre a la ciudad no visitada mas cercana."""
    unvisited = set(range(len(points)))
    unvisited.remove(start)
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: math.dist(points[last], points[j]))
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


def two_opt(points, tour):
    """2-opt (primera mejora) hasta llegar a un optimo local.

    Revisa pares de aristas; en cuanto encuentra un par que al cruzarse/reconectarse
    acorta el tour, aplica el movimiento (invierte el segmento) y cuenta UNA mejora.
    Termina cuando ningun movimiento acorta el tour.
    Devuelve (tour, numero_de_mejoras).
    """
    tour = list(tour)
    n = len(tour)
    d = lambda a, b: math.dist(points[a], points[b])
    improvements = 0
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                if i == 0 and k == n - 1:
                    continue  # invertir todo el tour no cambia su longitud
                a, b = tour[i - 1], tour[i]
                c, e = tour[k], tour[(k + 1) % n]
                # cambio de longitud al sustituir (a,b),(c,e) por (a,c),(b,e)
                delta = d(a, c) + d(b, e) - d(a, b) - d(c, e)
                if delta < -1e-12:
                    tour[i:k + 1] = tour[i:k + 1][::-1]
                    improvements += 1
                    improved = True
                    break
            if improved:
                break
    return tour, improvements


def mst_prim(points):
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children, root: int = 0):
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def mst_approximation(points):
    return preorder(mst_prim(points))


def brute_force_optimal(points):
    n = len(points)
    return min(((0,) + p for p in itertools.permutations(range(1, n))),
               key=lambda t: tour_length(points, t))
