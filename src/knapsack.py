"""Ejercicio 2: mochila 0/1 con voraz (valor/peso vs menor peso) y DP exacto."""


def greedy_knapsack(values, weights, capacity, key="ratio"):
    """Mochila 0/1 voraz.

    key = 'ratio'  -> prioriza mayor valor/peso (criterio clasico)
    key = 'weight' -> prioriza menor peso primero (criterio del ejercicio)

    Recorre los objetos en ese orden y mete cada uno si todavia cabe.
    Devuelve (indices elegidos, peso usado, valor total).
    """
    n = len(values)
    if key == "ratio":
        order = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    elif key == "weight":
        order = sorted(range(n), key=lambda i: weights[i])
    else:
        raise ValueError("key debe ser 'ratio' o 'weight'")

    chosen, used, total = [], 0, 0
    for i in order:
        if used + weights[i] <= capacity:
            chosen.append(i)
            used += weights[i]
            total += values[i]
    return sorted(chosen), used, total


def dp_knapsack(values, weights, capacity):
    """Optimo exacto por programacion dinamica (referencia).
    Devuelve (indices elegidos, peso usado, valor total)."""
    n = len(values)
    K = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        v, w = values[i - 1], weights[i - 1]
        for c in range(capacity + 1):
            K[i][c] = K[i - 1][c]
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])
    chosen, c = [], capacity
    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    chosen = sorted(chosen)
    return chosen, sum(weights[i] for i in chosen), K[n][capacity]
