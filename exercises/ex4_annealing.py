"""Ejercicio 4: Recocido simulado (SA) con 2-opt: esquemas de enfriamiento.

Ejecutar desde la raiz del repo:  python -m exercises.ex4_annealing
"""
import csv
import random
import statistics as st
import time

import matplotlib.pyplot as plt

from src.annealing import (exponential_schedule, linear_schedule,
                           logarithmic_schedule, simulated_annealing)
from src.tsp import random_instance, random_tour, tour_length, two_opt
from src.utils import DATA_DIR, save_fig, set_seed

N_CITIES = 50
RUNS = 10                                   # repeticiones por configuracion
BUDGETS = [2_000, 10_000, 50_000, 100_000, 500_000]   # iteraciones
N_CONV = 100_000                            # presupuesto usado para curvas de convergencia
T0, TF = 5.0, 0.01                          # temperatura inicial / final

SCHEDULES = {
    "Lineal": linear_schedule(T0),
    "Exponencial": exponential_schedule(T0, TF),
    "Logaritmico": logarithmic_schedule(T0),
}
COLORS = {"Lineal": "tab:blue", "Exponencial": "tab:red", "Logaritmico": "tab:green"}


def main():
    rng = set_seed()
    pts = random_instance(N_CITIES, rng)    # UNA instancia fija para comparar justo

    # Referencia: 2-opt (busqueda local pura) desde tours aleatorios
    base = [tour_length(pts, two_opt(pts, random_tour(N_CITIES, random.Random(r)))[0])
            for r in range(RUNS)]
    base_mean = st.mean(base)

    results = []          # (esquema, iteraciones, run, tiempo, mejor_costo)
    histories = {name: [] for name in SCHEDULES}
    for name, sched in SCHEDULES.items():
        for N in BUDGETS:
            for r in range(RUNS):
                init = random_tour(N_CITIES, random.Random(r))   # mismo inicio para todos
                t0 = time.perf_counter()
                _, best, hist = simulated_annealing(
                    pts, init, sched, N, random.Random(1000 + r), record_every=max(N // 500, 1))
                dt = time.perf_counter() - t0
                results.append((name, N, r, dt, best))
                if N == N_CONV:
                    histories[name].append(hist)
        print(f"{name} listo")

    with open(DATA_DIR / "ex4_annealing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["esquema", "iteraciones", "run", "tiempo_s", "mejor_costo"])
        w.writerows(results)

    best_known = min(min(r[4] for r in results), min(base))

    # ---- Tabla resumen ----
    print(f"\n{N_CITIES} ciudades, {RUNS} corridas por celda. "
          f"2-opt desde aleatorio: media {base_mean:.2f}\n")
    print(f"{'Esquema':<12}{'Iter':>9}{'Costo medio':>13}{'+-std':>8}{'Mejor':>8}{'Tiempo (s)':>12}{'Brecha':>9}")
    for name in SCHEDULES:
        for N in BUDGETS:
            rows = [r for r in results if r[0] == name and r[1] == N]
            costs = [r[4] for r in rows]
            print(f"{name:<12}{N:>9,}{st.mean(costs):>13.2f}{st.stdev(costs):>8.2f}"
                  f"{min(costs):>8.2f}{st.mean(r[3] for r in rows):>12.3f}"
                  f"{(st.mean(costs) / best_known - 1):>9.1%}")

    # ---- Figura 1: convergencia ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for name in SCHEDULES:
        H = histories[name]
        its = [h[0] for h in H[0]]
        temps = [h[1] for h in H[0]]
        cur = [st.mean(h[i][2] for h in H) for i in range(len(its))]
        axes[0].plot(its, temps, color=COLORS[name], label=name)
        axes[1].plot(its, cur, color=COLORS[name], label=name)
        axes[2].plot(temps, cur, color=COLORS[name], label=name)
    axes[0].set(title="Temperatura vs iteraciones", xlabel="Iteracion", ylabel="Temperatura T")
    axes[1].set(title="Costo vs iteraciones", xlabel="Iteracion", ylabel="Costo actual (media de corridas)")
    axes[2].set(title="Costo vs temperatura", xlabel="Temperatura T (escala log; enfria hacia la derecha)")
    axes[2].set_xscale("log")
    axes[2].invert_xaxis()
    for ax in axes[1:]:
        ax.axhline(base_mean, ls="--", color="gray", label="2-opt (busqueda local)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Convergencia del recocido simulado ({N_CONV:,} iteraciones, {RUNS} corridas, {N_CITIES} ciudades)")
    print("\nFigura:", save_fig("ex4_convergence"))

    # ---- Figura 2: tiempo vs calidad ----
    plt.figure(figsize=(7.5, 5))
    for name in SCHEDULES:
        xs, ys, es = [], [], []
        for N in BUDGETS:
            rows = [r for r in results if r[0] == name and r[1] == N]
            xs.append(st.mean(r[3] for r in rows))
            c = [r[4] for r in rows]
            ys.append(st.mean(c))
            es.append(st.stdev(c))
        plt.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3, color=COLORS[name], label=name)
    plt.axhline(base_mean, ls="--", color="gray", label="2-opt (busqueda local)")
    plt.xscale("log")
    plt.xlabel("Tiempo de ejecucion (s, escala log)  [puntos = 2k, 10k, 50k, 100k, 500k iteraciones]")
    plt.ylabel("Costo final (media +- std)")
    plt.title("Tiempo vs calidad por esquema de enfriamiento")
    plt.grid(alpha=0.3)
    plt.legend()
    print("Figura:", save_fig("ex4_time_quality"))


if __name__ == "__main__":
    main()
