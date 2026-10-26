import csv
import multiprocessing as mp
import statistics
import time

import matplotlib.pyplot as plt

from src.parallel_pi import estimate_pi
from src.utils import DATA_DIR, save_fig

PROCESSES = [1, 2, 4, 8]
SEEDS = list(range(8))      # 8 tareas independientes
SAMPLES = 1_000_000         # puntos por tarea
REPEATS = 5                 # repetimos y usamos la mediana (un solo run es ruidoso)


def run_once(n_procs: int) -> float:
    """Tiempo total (pared) de repartir las 8 tareas entre n_procs procesos."""
    t0 = time.perf_counter()
    with mp.Pool(processes=n_procs) as pool:
        pool.starmap(estimate_pi, [(s, SAMPLES) for s in SEEDS])
    return time.perf_counter() - t0


def main():
    print(f"Nucleos detectados: {mp.cpu_count()}")
    run_once(1)  # calentamiento (la primera corrida suele ser mas lenta)

    times = {}
    for p in PROCESSES:
        runs = [run_once(p) for _ in range(REPEATS)]
        times[p] = statistics.median(runs)
        print(f"p={p}: mediana = {times[p]:.3f} s  (corridas: "
              f"{', '.join(f'{r:.3f}' for r in runs)})")

    speedup = {p: times[1] / times[p] for p in PROCESSES}
    efficiency = {p: speedup[p] / p for p in PROCESSES}

    print("\n procesos | tiempo (s) | speedup | eficiencia")
    for p in PROCESSES:
        print(f"{p:9d} | {times[p]:10.3f} | {speedup[p]:7.2f} | {efficiency[p]:9.0%}")

    with open(DATA_DIR / "ex1_speedup.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["procesos", "tiempo_s", "speedup", "eficiencia"])
        for p in PROCESSES:
            w.writerow([p, f"{times[p]:.4f}", f"{speedup[p]:.3f}", f"{efficiency[p]:.3f}"])

    plt.figure(figsize=(6, 4.5))
    plt.plot(PROCESSES, PROCESSES, "--", color="gray", label="Speedup ideal (lineal)")
    plt.plot(PROCESSES, [speedup[p] for p in PROCESSES], "o-", label="Speedup medido")
    for p in PROCESSES:
        plt.annotate(f"{speedup[p]:.2f}x", (p, speedup[p]),
                     textcoords="offset points", xytext=(6, -12))
    plt.xticks(PROCESSES)
    plt.xlabel("Numero de procesos")
    plt.ylabel("Speedup  T(1) / T(p)")
    plt.title(f"Speedup de estimate_pi ({mp.cpu_count()} nucleos)")
    plt.grid(alpha=0.3)
    plt.legend()
    print("\nFigura:", save_fig("ex1_speedup"))


if __name__ == "__main__":
    main()