from pathlib import Path

import matplotlib.pyplot as plt
import tsplib95


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_PATH = PROJECT_ROOT / "data" / "tsplib" / "eil51.tsp"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "eil51_nodes.png"


def main() -> None:
    if not PROBLEM_PATH.exists():
        raise FileNotFoundError(f"Problem dosyası bulunamadı: {PROBLEM_PATH}")

    problem = tsplib95.load(str(PROBLEM_PATH))
    coordinates = problem.node_coords

    if not coordinates:
        raise ValueError("Problem dosyasında düğüm koordinatları bulunamadı.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    node_ids = sorted(coordinates)
    x_values = [coordinates[node_id][0] for node_id in node_ids]
    y_values = [coordinates[node_id][1] for node_id in node_ids]

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.scatter(
        x_values,
        y_values,
        s=90,
        edgecolors="black",
        linewidths=1,
        zorder=2,
    )

    # Başlangıç düğümünü farklı bir işaretle göster.
    start_x, start_y = coordinates[1]
    ax.scatter(
        [start_x],
        [start_y],
        s=230,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=3,
        label="Başlangıç düğümü: 1",
    )

    for node_id in node_ids:
        x, y = coordinates[node_id]

        ax.annotate(
            str(node_id),
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            zorder=4,
        )

    ax.set_title(
        f"{problem.name} — {problem.dimension} Düğümlü TSPLIB Problemi",
        fontsize=16,
    )
    ax.set_xlabel("X koordinatı")
    ax.set_ylabel("Y koordinatı")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Problem adı  : {problem.name}")
    print(f"Düğüm sayısı : {problem.dimension}")
    print(f"Görsel       : {OUTPUT_PATH}")
    print("Görsel oluşturma başarılı.")


if __name__ == "__main__":
    main()