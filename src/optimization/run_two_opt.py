import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.tsp_utils import (
    calculate_optimality_gap,
    calculate_route_distance,
    load_eil51_problem,
    validate_route,
)
from src.visualization.plot_route_results import plot_route


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "groq"
    / "optimize"
    / "groq_optimize_eil51.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "baselines"
    / "two_opt"
    / "eil51_from_groq_route.json"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "output"
    / "figures"
    / "eil51_groq_route_after_two_opt.png"
)

KNOWN_OPTIMUM = 426


def load_input_route() -> list[int]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Groq optimization sonuç dosyası bulunamadı:\n"
            f"{INPUT_PATH}"
        )

    result = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    route = result.get("final", {}).get("route")

    if not isinstance(route, list):
        raise ValueError(
            "Sonuç dosyasında final.route listesi bulunamadı."
        )

    if not all(
        isinstance(node, int) and not isinstance(node, bool)
        for node in route
    ):
        raise ValueError(
            "Rotadaki bütün düğümler tam sayı olmalıdır."
        )

    return route


def two_opt_best_improvement(
    problem: Any,
    route: list[int],
) -> tuple[list[int], list[dict[str, int]], int]:
    """
    Simetrik TSP için deterministik best-improvement 2-opt uygular.

    Başlangıç/bitiş düğümü olan 1 sabit kalır. Her turda en fazla
    mesafe azaltan tek segment ters çevrilir. Hiç iyileştirme
    kalmayınca algoritma durur.
    """
    best_route = list(route)
    moves: list[dict[str, int]] = []
    passes = 0

    while True:
        passes += 1
        best_delta = 0
        best_i: int | None = None
        best_j: int | None = None

        # İlk ve son düğüm 1 olduğu için yalnızca iç bölümü değiştir.
        for i in range(1, len(best_route) - 2):
            a = best_route[i - 1]
            b = best_route[i]

            for j in range(i + 1, len(best_route) - 1):
                c = best_route[j]
                d = best_route[j + 1]

                old_edges = (
                    problem.get_weight(a, b)
                    + problem.get_weight(c, d)
                )
                new_edges = (
                    problem.get_weight(a, c)
                    + problem.get_weight(b, d)
                )
                delta = new_edges - old_edges

                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_i is None or best_j is None:
            break

        before_distance = calculate_route_distance(
            problem,
            best_route,
        )

        best_route[best_i : best_j + 1] = reversed(
            best_route[best_i : best_j + 1]
        )

        after_distance = calculate_route_distance(
            problem,
            best_route,
        )

        moves.append(
            {
                "move": len(moves) + 1,
                "segment_start_index": best_i,
                "segment_end_index": best_j,
                "distance_before": before_distance,
                "distance_after": after_distance,
                "improvement": before_distance - after_distance,
            }
        )

    return best_route, moves, passes


def main() -> None:
    problem = load_eil51_problem()
    expected_nodes = list(problem.get_nodes())
    initial_route = load_input_route()

    if not validate_route(initial_route, expected_nodes):
        raise ValueError(
            "2-opt yalnızca geçerli bir TSP turuna uygulanabilir."
        )

    initial_distance = calculate_route_distance(
        problem,
        initial_route,
    )

    optimized_route, moves, passes = two_opt_best_improvement(
        problem,
        initial_route,
    )

    if not validate_route(optimized_route, expected_nodes):
        raise RuntimeError(
            "2-opt sonucunda rota geçersiz hâle geldi."
        )

    final_distance = calculate_route_distance(
        problem,
        optimized_route,
    )
    final_gap = calculate_optimality_gap(
        final_distance,
        KNOWN_OPTIMUM,
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "deterministic_two_opt",
        "problem": "eil51",
        "input_source": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "known_optimum": KNOWN_OPTIMUM,
        "initial": {
            "valid": True,
            "distance": initial_distance,
            "gap_percent": round(
                calculate_optimality_gap(
                    initial_distance,
                    KNOWN_OPTIMUM,
                ),
                2,
            ),
            "route": initial_route,
        },
        "algorithm": {
            "name": "best_improvement_2_opt",
            "passes": passes,
            "accepted_moves": len(moves),
            "moves": moves,
        },
        "final": {
            "valid": True,
            "distance": final_distance,
            "gap_percent": round(final_gap, 2),
            "distance_improvement": (
                initial_distance - final_distance
            ),
            "improvement_percent": round(
                (
                    (initial_distance - final_distance)
                    / initial_distance
                )
                * 100,
                2,
            ),
            "route": optimized_route,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_route(
        problem=problem,
        route=optimized_route,
        output_path=FIGURE_PATH,
        title=(
            "eil51 — Groq Rotası + Deterministik 2-opt\n"
            f"Mesafe: {final_distance} | Gap: %{final_gap:.2f}"
        ),
    )

    print("=== Deterministik 2-opt tamamlandı ===")
    print(f"Başlangıç mesafe : {initial_distance}")
    print(f"Son mesafe       : {final_distance}")
    print(
        f"Toplam iyileşme  : "
        f"{initial_distance - final_distance}"
    )
    print(f"Gap (%)          : {final_gap:.2f}")
    print(f"Kabul edilen hamle: {len(moves)}")
    print(f"Tarama turu      : {passes}")
    print(f"Sonuç dosyası    : {OUTPUT_PATH}")
    print(f"Rota görseli     : {FIGURE_PATH}")


if __name__ == "__main__":
    main()
