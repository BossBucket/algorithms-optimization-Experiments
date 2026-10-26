"""Ejercicio 4: recocido simulado (Simulated Annealing) con 2-opt y esquemas de enfriamiento."""
import math
import random


# ---- Esquemas de enfriamiento: devuelven una funcion T(t, N) --------------------
# t = iteracion actual (0..N-1), N = total de iteraciones.

def linear_schedule(t0: float):
    """T baja en linea recta desde t0 hasta 0 al final."""
    return lambda t, N: t0 * (1 - t / N)


def exponential_schedule(t0: float, tf: float):
    """T se multiplica por un factor constante: baja rapido al inicio, lento al final.
    Va de t0 a tf justo en la ultima iteracion."""
    return lambda t, N: t0 * (tf / t0) ** (t / N)


def logarithmic_schedule(t0: float):
    """Enfriamiento logaritmico (teorico): muy lento, T0 / (1 + ln(1 + t))."""
    return lambda t, N: t0 / (1 + math.log(1 + t))


# ---- Recocido simulado ----------------------------------------------------------

def simulated_annealing(points, initial_tour, schedule, iterations, rng, record_every=100):
    """SA para TSP usando 2-opt como movimiento aleatorio.

    En cada iteracion: elige al azar dos posiciones i<k, calcula cuanto cambiaria la
    longitud del tour (delta) si invierte el tramo i..k, y
      - si delta < 0 (mejora): siempre acepta
      - si delta >= 0 (empeora): acepta con probabilidad exp(-delta / T)

    Devuelve (mejor_tour, mejor_costo, historial) con historial = lista de
    (iteracion, temperatura, costo_actual, mejor_costo) muestreada cada record_every.
    """
    n = len(points)
    D = [[math.dist(a, b) for b in points] for a in points]
    tour = list(initial_tour)
    cost = sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))
    best_tour, best_cost = tour[:], cost
    history = []

    for t in range(iterations):
        T = max(schedule(t, iterations), 1e-9)
        if t % record_every == 0:
            history.append((t, T, cost, best_cost))
        i = rng.randrange(n - 1)
        k = rng.randrange(i + 1, n)
        if i == 0 and k == n - 1:
            continue
        a, b = tour[i - 1], tour[i]
        c, e = tour[k], tour[(k + 1) % n]
        delta = D[a][c] + D[b][e] - D[a][b] - D[c][e]
        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:k + 1] = tour[i:k + 1][::-1]
            cost += delta
            if cost < best_cost - 1e-12:
                best_cost, best_tour = cost, tour[:]
    return best_tour, best_cost, history
