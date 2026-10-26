"""Utilidades comunes: semillas, rutas de resultados y estilo de graficas."""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guardar figuras sin abrir ventanas
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
DATA_DIR = ROOT / "results" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def set_seed(seed: int = SEED) -> random.Random:
    """Devuelve un generador propio y reproducible (no toca el global)."""
    return random.Random(seed)


def save_fig(name: str) -> Path:
    """Guarda la figura actual en results/figures/<name>.png"""
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
