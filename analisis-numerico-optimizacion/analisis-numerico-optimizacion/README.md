# Analisis Numerico - Optimizacion de algoritmos

Implementaciones y experimentos de la tarea (paralelismo, mochila voraz, 2-opt, recocido simulado y aproximacion MST para el TSP).

## Estructura

```
src/         codigo reutilizable (pi, mochila, TSP, recocido, utilidades)
exercises/   un script por ejercicio (ex1 ... ex5)
results/     figuras (figures/) y datos (data/) generados
report/      reporte final
```

## Uso

```bash
pip install -r requirements.txt
python -m exercises.ex1_speedup
python -m exercises.ex2_knapsack
python -m exercises.ex3_two_opt
python -m exercises.ex4_annealing
python -m exercises.ex5_mst
```

Todas las corridas usan semillas fijas (`SEED = 42` en `src/utils.py`) para que las graficas sean reproducibles.
