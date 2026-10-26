"""Ejercicio 5: Razon de aproximacion del algoritmo MST para el TSP.

Ejecutar desde la raiz del repo:  python -m exercises.ex5_mst
"""
import csv
import random
import statistics as st

import matplotlib.pyplot as plt

from src.annealing import exponential_schedule, simulated_annealing
from src.tsp import (brute_force_optimal, held_karp_cost, mst_approximation,
                     random_instance, random_tour, tour_length, two_opt)
from src.utils import DATA_DIR, save_fig, set_seed

N_CITIES = 9
N_INSTANCES = 100
SA_ITERS = 20_000
T0, TF = 5.0, 0.01


def main():
    rng = set_seed()
    rows = []
    for inst in range(N_INSTANCES):
        pts = random_instance(N_CITIES, rng)

        opt = held_karp_cost(pts)                       # optimo exacto
        if inst < 3:                                    # verificacion con fuerza bruta (8! = 40,320)
            bf = tour_length(pts, brute_force_optimal(pts))
            assert abs(bf - opt) < 1e-9, "Held-Karp no coincide con fuerza bruta"

        mst_tour = mst_approximation(pts)
        mst_cost = tour_length(pts, mst_tour)
        mst2_cost = tour_length(pts, two_opt(pts, mst_tour)[0])

        ls_cost = tour_length(pts, two_opt(pts, random_tour(N_CITIES, rng))[0])  # busqueda local
        _, sa_cost, _ = simulated_annealing(                                     # recocido simulado
            pts, random_tour(N_CITIES, rng), exponential_schedule(T0, TF), SA_ITERS,
            random.Random(inst), record_every=SA_ITERS)

        rows.append({"instancia": inst, "optimo": opt, "mst": mst_cost,
                     "mst_2opt": mst2_cost, "busqueda_local": ls_cost, "recocido": sa_cost})

    with open(DATA_DIR / "ex5_mst.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    methods = {"MST puro": "mst", "MST + 2-opt": "mst_2opt",
               "2-opt (busq. local)": "busqueda_local", "Recocido simulado": "recocido"}
    ratios = {m: [r[k] / r["optimo"] for r in rows] for m, k in methods.items()}
    mst_r = ratios["MST puro"]
    assert max(mst_r) <= 2 + 1e-9, "Se violo la garantia teorica"

    # ---- Resumen ----
    print(f"{N_INSTANCES} instancias de {N_CITIES} ciudades (razon = costo obtenido / costo optimo)\n")
    print(f"{'Metodo':<22}{'Media':>8}{'Mediana':>9}{'Maxima':>8}{'Optimo':>9}")
    for m, rs in ratios.items():
        hit = sum(x < 1 + 1e-9 for x in rs)
        print(f"{m:<22}{st.mean(rs):>8.3f}{st.median(rs):>9.3f}{max(rs):>8.3f}{hit:>6d}/{N_INSTANCES}")
    print(f"\nMST puro: razon maxima observada {max(mst_r):.3f} (limite teorico 2)")
    gap = [a - b for a, b in zip(ratios['MST puro'], ratios['MST + 2-opt'])]
    print(f"MST + 2-opt mejora al MST puro en {sum(g > 1e-9 for g in gap)}/{N_INSTANCES} instancias; "
          f"reduce la razon media de {st.mean(mst_r):.3f} a {st.mean(ratios['MST + 2-opt']):.3f}")

    # ---- Figura 1: razon de aproximacion del MST ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].hist(mst_r, bins=15, color="tab:blue", edgecolor="white")
    axes[0].axvline(2, color="red", ls="--", label="Limite teorico = 2")
    axes[0].axvline(st.mean(mst_r), color="black", ls=":", label=f"Media = {st.mean(mst_r):.2f}")
    axes[0].set(title="Histograma de la razon de aproximacion (MST puro)",
                xlabel="Costo obtenido / Costo optimo", ylabel="Numero de instancias")
    axes[0].set_xlim(0.95, 2.1)
    axes[1].scatter(range(N_INSTANCES), mst_r, s=14, color="tab:blue")
    axes[1].axhline(2, color="red", ls="--", label="Limite teorico = 2")
    axes[1].axhline(1, color="green", ls="--", label="Optimo = 1")
    axes[1].set(title="Razon por instancia", xlabel="Instancia", ylabel="Costo obtenido / Costo optimo")
    axes[1].set_ylim(0.9, 2.1)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    print("\nFigura:", save_fig("ex5_ratio_mst"))

    # ---- Figura 2: comparativa de metodos ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    names = list(methods)
    axes[0].boxplot([[r[methods[m]] for r in rows] for m in names], showmeans=True)
    axes[0].boxplot([[r["optimo"] for r in rows]], positions=[len(names) + 1], showmeans=True)
    axes[0].set_xticks(range(1, len(names) + 2))
    axes[0].set_xticklabels(names + ["Optimo"], rotation=15, fontsize=8)
    axes[0].set(title="Costo obtenido", ylabel="Longitud del tour")
    axes[1].boxplot([ratios[m] for m in names], showmeans=True)
    axes[1].axhline(1, color="green", ls="--", label="Optimo")
    axes[1].set_xticks(range(1, len(names) + 1))
    axes[1].set_xticklabels(names, rotation=15, fontsize=8)
    axes[1].set(title="Razon costo / optimo", ylabel="Costo obtenido / Costo optimo")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Comparativa de metodos ({N_INSTANCES} instancias de {N_CITIES} ciudades; triangulo = media)")
    print("Figura:", save_fig("ex5_methods"))


if __name__ == "__main__":
    main()
