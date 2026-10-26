"""Ejercicio 3: 2-opt para TSP: inicio aleatorio vs. vecino mas cercano.

Ejecutar desde la raiz del repo:  python -m exercises.ex3_two_opt
"""
import csv
import statistics as st

import matplotlib.pyplot as plt

from src.tsp import (nearest_neighbor, random_instance, random_tour,
                     tour_length, two_opt)
from src.utils import DATA_DIR, save_fig, set_seed

N_CITIES = 30
N_INSTANCES = 100


def main():
    rng = set_seed()
    rows = []
    for inst in range(N_INSTANCES):
        pts = random_instance(N_CITIES, rng)

        t_rand = random_tour(N_CITIES, rng)
        t_nn = nearest_neighbor(pts, rng.randrange(N_CITIES))  # nodo inicial aleatorio

        opt_rand, imp_rand = two_opt(pts, t_rand)
        opt_nn, imp_nn = two_opt(pts, t_nn)

        rows.append({
            "instancia": inst,
            "init_aleatorio": tour_length(pts, t_rand),
            "init_nn": tour_length(pts, t_nn),
            "mejoras_aleatorio": imp_rand, "mejoras_nn": imp_nn,
            "final_aleatorio": tour_length(pts, opt_rand),
            "final_nn": tour_length(pts, opt_nn),
        })

    with open(DATA_DIR / "ex3_two_opt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    col = lambda k: [r[k] for r in rows]
    print(f"{N_INSTANCES} instancias aleatorias de {N_CITIES} ciudades\n")
    print(f"{'':<22}{'Aleatorio':>12}{'Vecino mas cercano':>20}")
    for label, a, b in [
        ("Longitud inicial (media)", col("init_aleatorio"), col("init_nn")),
        ("Mejoras (media)", col("mejoras_aleatorio"), col("mejoras_nn")),
        ("Mejoras (mediana)", col("mejoras_aleatorio"), col("mejoras_nn")),
        ("Longitud final (media)", col("final_aleatorio"), col("final_nn")),
    ]:
        f = st.median if "mediana" in label else st.mean
        print(f"{label:<22}{f(a):>12.2f}{f(b):>20.2f}")

    fewer = sum(r["mejoras_nn"] < r["mejoras_aleatorio"] for r in rows)
    better = sum(r["final_nn"] < r["final_aleatorio"] for r in rows)
    worse = sum(r["final_nn"] > r["final_aleatorio"] for r in rows)
    print(f"\nNN necesita menos mejoras en {fewer}/{N_INSTANCES} instancias")
    print(f"Optimo local de NN mejor en {better}, peor en {worse}, "
          f"empate en {N_INSTANCES - better - worse}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["Aleatorio", "Vecino mas cercano"]
    for ax, (ka, kb, title) in zip(axes, [
        ("mejoras_aleatorio", "mejoras_nn", "Numero de mejoras hasta el optimo local"),
        ("final_aleatorio", "final_nn", "Longitud del tour final (optimo local)"),
    ]):
        ax.boxplot([col(ka), col(kb)], showmeans=True)
        ax.set_xticklabels(labels)  # compatible con versiones viejas y nuevas de matplotlib
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mejoras 2-opt")
    axes[1].set_ylabel("Longitud del tour")
    fig.suptitle(f"2-opt: {N_INSTANCES} instancias de {N_CITIES} ciudades (triangulo = media)")
    print("\nFigura:", save_fig("ex3_two_opt"))


if __name__ == "__main__":
    main()
