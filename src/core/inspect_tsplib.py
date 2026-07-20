from pathlib import Path

import tsplib95


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tsplib"

PROBLEM_PATH = DATA_DIR / "eil51.tsp"
TOUR_PATH = DATA_DIR / "eil51.opt.tour"

EXPECTED_NAME = "eil51"
EXPECTED_DIMENSION = 51
EXPECTED_EDGE_WEIGHT_TYPE = "EUC_2D"
EXPECTED_OPTIMUM = 426


def calculate_tour_distance(
    problem: tsplib95.models.StandardProblem,
    tour: list[int],
) -> int:
    """Kapalı bir turun TSPLIB kurallarına göre toplam mesafesini hesaplar."""
    if not tour:
        raise ValueError("Tur boş olamaz.")

    closed_tour = tour + [tour[0]]

    return sum(
        problem.get_weight(start_node, end_node)
        for start_node, end_node in zip(
            closed_tour[:-1],
            closed_tour[1:],
            strict=True,
        )
    )


def main() -> None:
    if not PROBLEM_PATH.exists():
        raise FileNotFoundError(f"Problem dosyası bulunamadı: {PROBLEM_PATH}")

    if not TOUR_PATH.exists():
        raise FileNotFoundError(f"Tur dosyası bulunamadı: {TOUR_PATH}")

    problem = tsplib95.load(str(PROBLEM_PATH))
    tour_problem = tsplib95.load(str(TOUR_PATH))

    if not tour_problem.tours:
        raise ValueError("Optimum tur dosyasında tur bulunamadı.")

    optimal_tour = list(tour_problem.tours[0])
    nodes = list(problem.get_nodes())
    distance = calculate_tour_distance(problem, optimal_tour)

    print(f"Problem adı       : {problem.name}")
    print(f"Düğüm sayısı      : {problem.dimension}")
    print(f"Mesafe türü       : {problem.edge_weight_type}")
    print(f"Okunan düğüm      : {len(nodes)}")
    print(f"Tur düğüm sayısı  : {len(optimal_tour)}")
    print(f"Tur mesafesi      : {distance}")
    print(f"Beklenen optimum  : {EXPECTED_OPTIMUM}")

    assert problem.name == EXPECTED_NAME
    assert problem.dimension == EXPECTED_DIMENSION
    assert problem.edge_weight_type == EXPECTED_EDGE_WEIGHT_TYPE
    assert set(optimal_tour) == set(nodes)
    assert len(optimal_tour) == len(set(optimal_tour))
    assert distance == EXPECTED_OPTIMUM

    print("Doğrulama başarılı.")


if __name__ == "__main__":
    main()