"""Наглядные результаты именно проверяемого решения, без вызова эталона."""
import json
from pathlib import Path
import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "outputs/tests/student"
PLOTS = True


def show(title, **values):
    print(f"\n  {title}", flush=True)
    for label, value in values.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                value = np.array2string(np.asarray(value), precision=6, suppress_small=False)
            except ValueError:
                value = str(value)
        print(f"    {label}: {value}", flush=True)


def save_basis_plot(a, p, x, coordinates, new_matrix):
    """Два пути вычисления одного образа в общей системе координат."""
    if not PLOTS:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a, p, x = np.asarray(a), np.asarray(p), np.asarray(x)
    reconstructed = p @ np.asarray(coordinates)
    direct, transformed = a @ x, p @ np.asarray(new_matrix) @ coordinates
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, vectors, title in [
        (axes[0], [(p[:, 0], 'p1'), (p[:, 1], 'p2'), (x, 'x'), (reconstructed, 'P c')], 'Один вектор в разных базисах'),
        (axes[1], [(direct, 'A x'), (transformed, 'P B c')], 'Совпадение результатов вычисления')]:
        for i, (v, label) in enumerate(vectors):
            ax.plot([0, v[0]], [0, v[1]], marker='o', label=label,
                    linestyle='--' if i == len(vectors)-1 else '-')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.3)
        ax.legend()
        ax.set_title(title)
    folder = OUTPUT / '04_operator_basis'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'basis.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Рисунок смены базиса: {path.resolve()}', flush=True)
