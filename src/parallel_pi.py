"""Ejercicio 1: Monte Carlo para pi y medicion de speedup.

estimate_pi vive aqui, en un modulo limpio, para que los procesos hijo
(modo spawn en Windows/macOS) no re-ejecuten codigo de otros ejercicios.
"""
import random


def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    """Fraccion de puntos aleatorios dentro del cuarto de circulo unitario."""
    rng = random.Random(seed)
    inside = sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples
