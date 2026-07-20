# VERSION: 2.0 - handles invalid model routes without crashing
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import tsplib95

from src.core.tsp_utils import (
    KNOWN_OPTIMUM,
    calculate_optimality_gap,
    calculate_route_distance,
    load_eil51_problem,
    normalize_route,
    validate_route,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "openrouter"
    / "zero_shot"
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
    / "figures"
    / "eil51_openrouter_zero_shot_route.png"
)

OPTIMAL_ROUTE_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "figures"
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


def route_diagnostics(route: Sequence[int]) -> dict:
    """Geçersiz bir rotadaki eksik ve tekrar eden düğümleri bulur."""
    route_list = list(route)
    route_without_return = route_list.copy()

    if (
        len(route_without_return) >= 2
        and route_without_return[0] == route_without_return[-1]
    ):
        route_without_return = route_without_return[:-1]

    expected_nodes = set(range(1, 52))
    counts = Counter(route_without_return)

    return {
        "route_length": len(route_list),
        "expected_route_length": 52,
        "missing_nodes": sorted(
            expected_nodes - set(route_without_return)
        ),
        "duplicate_nodes": sorted(
            node for node, count in counts.items() if count > 1
        ),
        "unexpected_nodes": sorted(
            set(route_without_return) - expected_nodes
        ),
        "starts_at_1": bool(route_list) and route_list[0] == 1,
        "returns_to_1": (
            len(route_list) >= 2 and route_list[-1] == 1
        ),
    }


def plot_route(
    problem,
    route: Sequence[int],
    output_path: Path,
    title: str,
) -> None:
    """Verilen geçerli eil51 rotasını PNG dosyasına çizer."""
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
    node_x = [coordinates[node_id][0] for node_id in node_ids]
    node_y = [coordinates[node_id][1] for node_id in node_ids]

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.plot(
        route_x,
        route_y,
        linewidth=1.3,
        alpha=0.85,
        zorder=1,
    )

    ax.scatter(
        node_x,
        node_y,
        s=80,
        edgecolors="black",
        linewidths=0.9,
        zorder=2,
    )

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    problem = load_eil51_problem()
    model_route = load_model_route()
    optimal_route = load_optimal_route()
    nodes = list(problem.get_nodes())

    model_route_valid = validate_route(model_route, nodes)

    if model_route_valid:
        model_distance = calculate_route_distance(
            problem,
            model_route,
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
                "eil51 — OpenRouter Zero-Shot Rotası\n"
                f"Mesafe: {model_distance} | Gap: %{model_gap:.2f}"
            ),
        )

        print("Model rotası:")
        print("Geçerli : True")
        print(f"Mesafe  : {model_distance}")
        print(f"Gap     : %{model_gap:.2f}")
        print(f"Görsel  : {MODEL_ROUTE_OUTPUT}")

    else:
        diagnostics = route_diagnostics(model_route)

        # Önceki modele ait geçerli rota görselinin yanlışlıkla
        # güncel sonuç sanılmasını önle.
        if MODEL_ROUTE_OUTPUT.exists():
            MODEL_ROUTE_OUTPUT.unlink()

        print("Model rotası:")
        print("Geçerli : False")
        print(
            f"Uzunluk : {diagnostics['route_length']} "
            f"(beklenen {diagnostics['expected_route_length']})"
        )
        print(f"Eksik   : {diagnostics['missing_nodes']}")
        print(f"Tekrar  : {diagnostics['duplicate_nodes']}")
        print(f"Fazladan: {diagnostics['unexpected_nodes']}")
        print("Model rota görseli oluşturulmadı.")

    optimal_distance = calculate_route_distance(
        problem,
        optimal_route,
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

    print("\nOptimum rota:")
    print(f"Mesafe : {optimal_distance}")
    print(f"Görsel : {OPTIMAL_ROUTE_OUTPUT}")
    print("\nRota sonucu kontrolü tamamlandı.")


if __name__ == "__main__":
    main()
