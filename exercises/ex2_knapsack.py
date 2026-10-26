"""Ejercicio 2 Mochila voraz:python -m exercises.ex2_knapsack
"""
import csv

import matplotlib.pyplot as plt

from src.knapsack import dp_knapsack, greedy_knapsack
from src.utils import DATA_DIR, save_fig

# Instancia disenada: el criterio "menor peso" es mucho peor que valor/peso.
# Hay dos objetos ligeros pero casi sin valor, y uno pesado muy valioso.
DESIGNED = {
    "name": "Instancia disenada (W=10)",
    "items": ["A", "B", "C"],
    "values": [1, 2, 60],
    "weights": [1, 3, 8],
    "capacity": 10,
}

# Instancia de clase: aqui incluso valor/peso falla frente al optimo.
CLASS = {
    "name": "Instancia de clase (W=50)",
    "items": ["A", "B", "C"],
    "values": [60, 100, 120],
    "weights": [10, 20, 30],
    "capacity": 50,
}


def evaluate(inst):
    v, w, W = inst["values"], inst["weights"], inst["capacity"]
    names = inst["items"]
    methods = {
        "Valor/peso": greedy_knapsack(v, w, W, key="ratio"),
        "Menor peso": greedy_knapsack(v, w, W, key="weight"),
        "Optimo (DP)": dp_knapsack(v, w, W),
    }
    rows = []
    for m, (chosen, used, total) in methods.items():
        rows.append({"instancia": inst["name"], "metodo": m,
                     "objetos": "".join(names[i] for i in chosen),
                     "capacidad_usada": used, "capacidad": W, "valor_total": total})
    return rows


def print_table(rows):
    print(f"\n{rows[0]['instancia']}")
    print(f"{'Metodo':<12} | {'Objetos':<8} | {'Capacidad usada':<16} | Valor total")
    for r in rows:
        print(f"{r['metodo']:<12} | {r['objetos']:<8} | "
              f"{r['capacidad_usada']:>3} / {r['capacidad']:<10} | {r['valor_total']}")


def main():
    all_rows = []
    for inst in (DESIGNED, CLASS):
        rows = evaluate(inst)
        print_table(rows)
        all_rows += rows

    with open(DATA_DIR / "ex2_knapsack.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Grafica de barras para la instancia disenada y la de clase
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = {"Valor/peso": "tab:blue", "Menor peso": "tab:orange", "Optimo (DP)": "tab:green"}
    for r_idx, inst in enumerate((DESIGNED, CLASS)):
        rows = evaluate(inst)
        names = [r["metodo"] for r in rows]
        cols = [colors[n] for n in names]
        for c_idx, (field, title) in enumerate([("valor_total", "Valor total"),
                                                ("capacidad_usada", "Capacidad usada")]):
            ax = axes[r_idx][c_idx]
            vals = [r[field] for r in rows]
            bars = ax.bar(names, vals, color=cols)
            for b, r in zip(bars, rows):
                ax.annotate(f"{b.get_height()}\n[{r['objetos']}]",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=9)
            if field == "capacidad_usada":
                ax.axhline(inst["capacity"], ls="--", color="gray")
            ax.set_ylim(0, max(vals + [inst["capacity"] if field == "capacidad_usada" else 0]) * 1.25)
            ax.set_title(f"{inst['name']}\n{title}", fontsize=10)
    print("\nFigura:", save_fig("ex2_knapsack"))


if __name__ == "__main__":
    main()
