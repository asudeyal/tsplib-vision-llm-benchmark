from pathlib import Path
from typing import Sequence

import tsplib95


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBLEM_PATH = PROJECT_ROOT / "data" / "tsplib" / "eil51.tsp"

KNOWN_OPTIMUM = 426


def load_eil51_problem():
    """eil51 TSPLIB problemini yükler."""
    if not PROBLEM_PATH.exists():
        raise FileNotFoundError(
            f"eil51 problem dosyası bulunamadı: {PROBLEM_PATH}"
        )

    return tsplib95.load(str(PROBLEM_PATH))


def normalize_route(
    route: Sequence[int],
    expected_nodes: Sequence[int],
) -> list[int]:
    """
    LLM rotayı başlangıç düğümünü sonda tekrar etmeden döndürürse kapatır.

    Örnek:
        [1, 2, ..., 51] -> [1, 2, ..., 51, 1]
    """
    normalized_route = list(route)

    if not normalized_route:
        return []

    if len(normalized_route) == len(expected_nodes):
        normalized_route.append(normalized_route[0])

    return normalized_route


def validate_route(
    route: Sequence[int],
    expected_nodes: Sequence[int],
) -> bool:
    """
    Rotanın geçerli bir Hamilton TSP turu olup olmadığını kontrol eder.

    Koşullar:
    - Başlangıç ve bitiş düğümü aynı olmalı.
    - Bütün düğümler tam bir kez ziyaret edilmeli.
    - Fazladan veya eksik düğüm bulunmamalı.
    """
    nodes = list(expected_nodes)
    normalized_route = normalize_route(route, nodes)

    if len(normalized_route) != len(nodes) + 1:
        return False

    if normalized_route[0] != normalized_route[-1]:
        return False

    visited_nodes = normalized_route[:-1]

    if len(visited_nodes) != len(set(visited_nodes)):
        return False

    return set(visited_nodes) == set(nodes)


def calculate_route_distance(
    problem,
    route: Sequence[int],
) -> int:
    """Geçerli rotanın TSPLIB kurallarına göre mesafesini hesaplar."""
    expected_nodes = list(problem.get_nodes())
    normalized_route = normalize_route(route, expected_nodes)

    if not validate_route(normalized_route, expected_nodes):
        raise ValueError("Mesafesi hesaplanmak istenen rota geçersiz.")

    return sum(
        problem.get_weight(start_node, end_node)
        for start_node, end_node in zip(
            normalized_route[:-1],
            normalized_route[1:],
            strict=True,
        )
    )


def calculate_optimality_gap(
    route_distance: float,
    optimum: float = KNOWN_OPTIMUM,
) -> float:
    """Rota mesafesinin bilinen optimumdan yüzde sapmasını hesaplar."""
    if optimum <= 0:
        raise ValueError("Optimum değer sıfırdan büyük olmalıdır.")

    return ((route_distance - optimum) / optimum) * 100


def evaluate_route(
    problem,
    route: Sequence[int],
) -> dict:
    """LLM tarafından döndürülen rotayı değerlendirir."""
    expected_nodes = list(problem.get_nodes())
    normalized_route = normalize_route(route, expected_nodes)
    is_valid = validate_route(normalized_route, expected_nodes)

    if not is_valid:
        return {
            "route": normalized_route,
            "valid": False,
            "distance": None,
            "known_optimum": KNOWN_OPTIMUM,
            "optimality_gap": None,
        }

    distance = calculate_route_distance(problem, normalized_route)
    gap = calculate_optimality_gap(distance)

    return {
        "route": normalized_route,
        "valid": True,
        "distance": distance,
        "known_optimum": KNOWN_OPTIMUM,
        "optimality_gap": gap,
    }