from pathlib import Path

import pytest
import tsplib95

from src.core.tsp_utils import (
    KNOWN_OPTIMUM,
    calculate_optimality_gap,
    calculate_route_distance,
    evaluate_route,
    load_eil51_problem,
    normalize_route,
    validate_route,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPTIMAL_TOUR_PATH = (
    PROJECT_ROOT / "data" / "tsplib" / "eil51.opt.tour"
)


@pytest.fixture
def problem():
    return load_eil51_problem()


@pytest.fixture
def optimal_route():
    tour_problem = tsplib95.load(str(OPTIMAL_TOUR_PATH))
    return list(tour_problem.tours[0])


def test_problem_is_eil51(problem):
    assert problem.name == "eil51"
    assert problem.dimension == 51
    assert problem.edge_weight_type == "EUC_2D"


def test_open_route_is_normalized(problem, optimal_route):
    nodes = list(problem.get_nodes())

    normalized = normalize_route(optimal_route, nodes)

    assert len(normalized) == 52
    assert normalized[0] == normalized[-1]


def test_optimal_route_is_valid(problem, optimal_route):
    nodes = list(problem.get_nodes())

    assert validate_route(optimal_route, nodes) is True


def test_optimal_route_distance_is_426(problem, optimal_route):
    distance = calculate_route_distance(problem, optimal_route)

    assert distance == KNOWN_OPTIMUM
    assert distance == 426


def test_optimal_route_gap_is_zero(problem, optimal_route):
    distance = calculate_route_distance(problem, optimal_route)
    gap = calculate_optimality_gap(distance)

    assert gap == pytest.approx(0.0)


def test_duplicate_node_route_is_invalid(problem, optimal_route):
    invalid_route = optimal_route.copy()
    invalid_route[0] = invalid_route[1]

    nodes = list(problem.get_nodes())

    assert validate_route(invalid_route, nodes) is False


def test_missing_node_route_is_invalid(problem, optimal_route):
    invalid_route = optimal_route[:-1]
    nodes = list(problem.get_nodes())

    assert validate_route(invalid_route, nodes) is False


def test_invalid_route_has_no_distance(problem, optimal_route):
    invalid_route = optimal_route[:-1]

    result = evaluate_route(problem, invalid_route)

    assert result["valid"] is False
    assert result["distance"] is None
    assert result["optimality_gap"] is None


def test_evaluate_optimal_route(problem, optimal_route):
    result = evaluate_route(problem, optimal_route)

    assert result["valid"] is True
    assert result["distance"] == 426
    assert result["optimality_gap"] == pytest.approx(0.0)