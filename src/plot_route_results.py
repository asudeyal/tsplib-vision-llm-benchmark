import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import tsplib95

from src.tsp_utils import (
    KNOWN_OPTIMUM,
    calculate_optimality_gap,
    calculate_route_distance,
    load_eil51_problem,
    normalize_route,
    validate_route,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "openrouter_zero_shot_eil51.json"
)

OPTIMAL_TOUR_PATH = (
    PROJECT_ROOT
    / "data"
    / "tsplib"
    / "eil51.opt.tour"
)

MODEL_ROUTE_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "eil51_openrouter_zero_shot_route.png"
)

OPTIMAL_ROUTE_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "eil51_optimal_route.png"
)


def load_model_route() -> list[int]:
    """Zero-shot sonuç dosyasındaki model rotasını okur."""
    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Deney sonuç dosyası bulunamadı: {RESULT_PATH}"
        )

    result = json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )

    route = result.get("parsed_route")

    if not isinstance(route, list):
        raise ValueError(
            "Sonuç dosyasında parsed_route listesi bulunamadı."
        )

    if not all(
        isinstance(node, int) and not isinstance(node, bool)
        for node in route
    ):
        raise ValueError(
            "Model rotasındaki bütün düğümler tam sayı olmalıdır."
        )

    return route


def load_optimal_route() -> list[int]:
    """TSPLIB optimum tur dosyasındaki rotayı okur."""
    if not OPTIMAL_TOUR_PATH.exists():
        raise FileNotFoundError(
            f"Optimum tur dosyası bulunamadı: {OPTIMAL_TOUR_PATH}"
        )

    tour_problem = tsplib95.load(str(OPTIMAL_TOUR_PATH))

    if not tour_problem.tours:
        raise ValueError(
            "Optimum tur dosyasında rota bulunamadı."
        )

    return list(tour_problem.tours[0])


def plot_route(
    problem,
    route: Sequence[int],
    output_path: Path,
    title: str,
) -> None:
    """Verilen eil51 rotasını PNG dosyasına çizer."""
    nodes = list(problem.get_nodes())
    normalized_route = normalize_route(route, nodes)

    if not validate_route(normalized_route, nodes):
        raise ValueError(
            f"Çizilmek istenen rota geçersiz: {output_path.name}"
        )

    coordinates = problem.node_coords

    route_x = [
        coordinates[node_id][0]
        for node_id in normalized_route
    ]
    route_y = [
        coordinates[node_id][1]
        for node_id in normalized_route
    ]

    node_ids = sorted(coordinates)
    node_x = [
        coordinates[node_id][0]
        for node_id in node_ids
    ]
    node_y = [
        coordinates[node_id][1]
        for node_id in node_ids
    ]

    fig, ax = plt.subplots(figsize=(12, 12))

    # Tur sırasına göre kenarları çiz.
    ax.plot(
        route_x,
        route_y,
        linewidth=1.3,
        alpha=0.85,
        zorder=1,
    )

    # Bütün düğümleri çiz.
    ax.scatter(
        node_x,
        node_y,
        s=80,
        edgecolors="black",
        linewidths=0.9,
        zorder=2,
    )

    # Başlangıç düğümünü belirginleştir.
    start_x, start_y = coordinates[1]

    ax.scatter(
        [start_x],
        [start_y],
        s=240,
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
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("X koordinatı")
    ax.set_ylabel("Y koordinatı")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    problem = load_eil51_problem()

    model_route = load_model_route()
    optimal_route = load_optimal_route()

    model_distance = calculate_route_distance(
        problem,
        model_route,
    )
    optimal_distance = calculate_route_distance(
        problem,
        optimal_route,
    )

    model_gap = calculate_optimality_gap(
        model_distance,
        KNOWN_OPTIMUM,
    )

    plot_route(
        problem=problem,
        route=model_route,
        output_path=MODEL_ROUTE_OUTPUT,
        title=(
            "eil51 — OpenRouter Nemotron Zero-Shot Rotası\n"
            f"Mesafe: {model_distance} | Gap: %{model_gap:.2f}"
        ),
    )

    plot_route(
        problem=problem,
        route=optimal_route,
        output_path=OPTIMAL_ROUTE_OUTPUT,
        title=(
            "eil51 — TSPLIB Optimum Rotası\n"
            f"Mesafe: {optimal_distance} | Gap: %0.00"
        ),
    )

    print("Model rotası:")
    print(f"Mesafe : {model_distance}")
    print(f"Gap    : %{model_gap:.2f}")
    print(f"Görsel : {MODEL_ROUTE_OUTPUT}")

    print("\nOptimum rota:")
    print(f"Mesafe : {optimal_distance}")
    print(f"Görsel : {OPTIMAL_ROUTE_OUTPUT}")

    print("\nRota görselleri başarıyla oluşturuldu.")


if __name__ == "__main__":
    main()