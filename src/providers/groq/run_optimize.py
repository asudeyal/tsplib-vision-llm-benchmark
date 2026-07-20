import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from src.core.tsp_utils import evaluate_route, load_eil51_problem
from src.providers.openrouter.run_zero_shot import (
    encode_image,
    extract_json_object,
    extract_route,
    get_usage_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_PATH = (
    PROJECT_ROOT / "output" / "figures" / "eil51_nodes.png"
)
REPAIR_RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "groq"
    / "repair"
    / "groq_multi_agent_eil51.json"
)
SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "groq"
    / "optimize"
    / "groq_optimize_eil51.json"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "output"
    / "checkpoints"
    / "groq"
    / "groq_optimize_eil51_checkpoint.json"
)

DEFAULT_MODEL = "qwen/qwen3.6-27b"
KNOWN_OPTIMUM = 426
MAX_BASE64_SIZE = 4 * 1024 * 1024


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Geçerli Groq repair rotasını critic-scorer yöntemiyle "
            "iyileştirmeye çalışır."
        )
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Azami optimizasyon iterasyonu.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="API çağrıları arasındaki bekleme süresi.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help=(
            "Bu kadar ardışık başarısız iyileştirmeden sonra "
            "erken durdur."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Mevcut checkpoint dosyasından devam eder.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def route_diagnostics(
    route: Sequence[int] | None,
    expected_nodes: Sequence[int],
) -> dict[str, Any]:
    nodes = list(expected_nodes)
    expected_set = set(nodes)
    expected_length = len(nodes) + 1

    if route is None:
        return {
            "route_length": 0,
            "expected_route_length": expected_length,
            "closed": False,
            "missing_nodes": nodes,
            "duplicate_nodes": [],
            "out_of_range_nodes": [],
        }

    route_list = list(route)
    closed = (
        len(route_list) >= 2
        and route_list[0] == route_list[-1]
    )
    visited = route_list[:-1] if closed else route_list
    counts = Counter(visited)

    return {
        "route_length": len(route_list),
        "expected_route_length": expected_length,
        "closed": closed,
        "missing_nodes": sorted(expected_set - set(visited)),
        "duplicate_nodes": sorted(
            node
            for node, count in counts.items()
            if node in expected_set and count > 1
        ),
        "out_of_range_nodes": sorted(
            {
                node
                for node in visited
                if node not in expected_set
            }
        ),
    }


def compact_evaluation(
    evaluation: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    gap = evaluation.get("optimality_gap")

    return {
        "valid": bool(evaluation.get("valid")),
        "distance": evaluation.get("distance"),
        "gap_percent": (
            round(float(gap), 2)
            if gap is not None
            else None
        ),
        "route_length": diagnostics["route_length"],
        "expected_route_length": diagnostics[
            "expected_route_length"
        ],
        "closed": diagnostics["closed"],
        "missing_nodes": diagnostics["missing_nodes"],
        "duplicate_nodes": diagnostics["duplicate_nodes"],
        "out_of_range_nodes": diagnostics[
            "out_of_range_nodes"
        ],
    }


def load_repaired_route() -> list[int]:
    if not REPAIR_RESULT_PATH.exists():
        raise FileNotFoundError(
            "Groq repair sonuç dosyası bulunamadı:\n"
            f"{REPAIR_RESULT_PATH}"
        )

    result = json.loads(
        REPAIR_RESULT_PATH.read_text(encoding="utf-8")
    )
    route = result.get("final", {}).get("route")

    if not isinstance(route, list):
        raise ValueError(
            "Repair sonuç dosyasında final.route bulunamadı."
        )

    if not all(
        isinstance(node, int) and not isinstance(node, bool)
        for node in route
    ):
        raise ValueError(
            "Repair rotasındaki bütün düğümler tam sayı olmalı."
        )

    problem = load_eil51_problem()
    evaluation = evaluate_route(problem, route)

    if not evaluation["valid"]:
        raise ValueError(
            "Optimization-only deneyi geçerli bir repair rotasıyla "
            "başlamalı."
        )

    return route


def build_critic_prompt(
    current_route: list[int],
    current_distance: int,
    current_gap: float,
    iteration: int,
) -> str:
    return f"""
You are the critic agent in iteration {iteration} of an eil51 TSP
route-optimization experiment.

The current route is already VALID.

Current route:
{json.dumps(current_route)}

Exact evaluation:
- Current distance: {current_distance}
- Known optimum: {KNOWN_OPTIMUM}
- Current optimality gap: {current_gap}

Your task is NOT to repair the route. Your task is to produce a
DIFFERENT valid route with a strictly smaller distance.

Use the image and apply one or more concrete local-search moves:
- 2-opt edge exchange with segment reversal,
- relocating one or more nodes,
- swapping nodes,
- removing crossing edges,
- replacing long jumps with nearby-node connections.

Hard constraints:
- Start at node 1.
- End at node 1.
- Visit every integer node from 1 through 51 exactly once.
- The route must contain exactly 52 integers.
- Never use nodes below 1 or above 51.
- Preserve validity.
- Do not return the current route unchanged.
- Do not merely rotate or reverse the complete route.
- A candidate is useful only if its exact distance is below
  {current_distance}.
- Return no explanation and no markdown.

Return exactly one JSON object:
{{"route": [1, ..., 1]}}
""".strip()


def build_scorer_prompt(
    current_route: list[int],
    current_distance: int,
    candidate_route: list[int],
    candidate_distance: int,
    iteration: int,
) -> str:
    return f"""
You are the scorer agent in iteration {iteration} of an eil51 TSP
optimization experiment.

Choose the route that should continue.

CURRENT ROUTE:
{json.dumps(current_route)}

CURRENT EXACT DISTANCE:
{current_distance}

CANDIDATE ROUTE:
{json.dumps(candidate_route)}

CANDIDATE EXACT DISTANCE:
{candidate_distance}

Rules:
1. Both routes are valid.
2. Choose candidate only when its exact distance is strictly lower.
3. Otherwise choose current.
4. Do not generate a new route.
5. Return no explanation and no markdown.

Return exactly one JSON object:
{{"choice": "current"}}
or
{{"choice": "candidate"}}
""".strip()


def extract_response_content(
    response: Any,
    agent_name: str,
) -> str:
    choices = getattr(response, "choices", None)

    if not choices:
        details = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else repr(response)
        )
        raise RuntimeError(
            f"{agent_name} boş completion döndürdü: {details}"
        )

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            f"{agent_name} boş metin içeriği döndürdü."
        )

    return content


def call_critic(
    client: OpenAI,
    model: str,
    image_base64: str,
    current_route: list[int],
    current_evaluation: dict[str, Any],
    iteration: int,
) -> tuple[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0.35,
        max_tokens=2000,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "none"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_critic_prompt(
                            current_route=current_route,
                            current_distance=current_evaluation[
                                "distance"
                            ],
                            current_gap=current_evaluation[
                                "optimality_gap"
                            ],
                            iteration=iteration,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                f"{image_base64}"
                            )
                        },
                    },
                ],
            }
        ],
    )

    return (
        extract_response_content(response, "Critic"),
        response,
    )


def call_scorer(
    client: OpenAI,
    model: str,
    current_route: list[int],
    current_distance: int,
    candidate_route: list[int],
    candidate_distance: int,
    iteration: int,
) -> tuple[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=80,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "none"},
        messages=[
            {
                "role": "user",
                "content": build_scorer_prompt(
                    current_route=current_route,
                    current_distance=current_distance,
                    candidate_route=candidate_route,
                    candidate_distance=candidate_distance,
                    iteration=iteration,
                ),
            }
        ],
    )

    return (
        extract_response_content(response, "Scorer"),
        response,
    )


def extract_scorer_choice(raw_response: str) -> str:
    parsed = extract_json_object(raw_response)
    choice = parsed.get("choice")

    if choice not in {"current", "candidate"}:
        raise ValueError(
            "Scorer yanıtında current veya candidate yok."
        )

    return choice


def create_initial_state(
    model: str,
    requested_iterations: int,
    patience: int,
) -> dict[str, Any]:
    problem = load_eil51_problem()
    nodes = list(problem.get_nodes())
    initial_route = load_repaired_route()
    initial_evaluation = evaluate_route(problem, initial_route)
    initial_diagnostics = route_diagnostics(initial_route, nodes)

    return {
        "experiment": "groq_optimization_critic_scorer",
        "problem": "eil51",
        "model_requested": model,
        "requested_iterations": requested_iterations,
        "patience": patience,
        "completed_iterations": 0,
        "timestamp_started_utc": utc_now(),
        "timestamp_updated_utc": utc_now(),
        "api_calls_attempted": 0,
        "api_calls_succeeded": 0,
        "initial_route": initial_route,
        "initial_evaluation": initial_evaluation,
        "initial_diagnostics": initial_diagnostics,
        "current_route": initial_route,
        "current_evaluation": initial_evaluation,
        "current_diagnostics": initial_diagnostics,
        "best_route": initial_route,
        "best_evaluation": initial_evaluation,
        "best_diagnostics": initial_diagnostics,
        "non_improvement_streak": 0,
        "early_stopped": False,
        "stop_reason": None,
        "iterations": [],
        "last_error": None,
    }


def load_or_create_state(
    model: str,
    requested_iterations: int,
    patience: int,
    resume: bool,
) -> dict[str, Any]:
    if resume and CHECKPOINT_PATH.exists():
        state = json.loads(
            CHECKPOINT_PATH.read_text(encoding="utf-8")
        )

        if state.get("model_requested") != model:
            raise ValueError(
                "Checkpoint modeli .env modelinden farklı."
            )

        state["requested_iterations"] = requested_iterations
        state["patience"] = patience
        state["early_stopped"] = False
        state["stop_reason"] = None
        return state

    return create_initial_state(
        model=model,
        requested_iterations=requested_iterations,
        patience=patience,
    )


def build_summary(state: dict[str, Any]) -> dict[str, Any]:
    initial = compact_evaluation(
        state["initial_evaluation"],
        state["initial_diagnostics"],
    )
    final = compact_evaluation(
        state["best_evaluation"],
        state["best_diagnostics"],
    )

    steps: list[dict[str, Any]] = []

    for item in state["iterations"]:
        candidate = compact_evaluation(
            item["critic"]["evaluation"],
            item["critic"]["diagnostics"],
        )

        steps.append(
            {
                "iteration": item["iteration"],
                "candidate": candidate,
                "scorer_choice": item["scorer"][
                    "parsed_choice"
                ],
                "accepted": item["accepted"],
                "selection_source": item["selection_source"],
                "best_distance_after_iteration": item[
                    "best_distance_after_iteration"
                ],
                "non_improvement_streak": item[
                    "non_improvement_streak"
                ],
            }
        )

    initial_distance = initial["distance"]
    final_distance = final["distance"]

    return {
        "experiment": state["experiment"],
        "problem": state["problem"],
        "model": state["model_requested"],
        "iterations": {
            "requested": state["requested_iterations"],
            "completed": state["completed_iterations"],
            "patience": state["patience"],
        },
        "api_calls": {
            "attempted": state["api_calls_attempted"],
            "succeeded": state["api_calls_succeeded"],
        },
        "known_optimum": KNOWN_OPTIMUM,
        "initial": {
            **initial,
            "route": state["initial_route"],
        },
        "steps": steps,
        "final": {
            **final,
            "distance_improvement": (
                initial_distance - final_distance
                if initial_distance is not None
                and final_distance is not None
                else None
            ),
            "route": state["best_route"],
        },
        "early_stopped": state["early_stopped"],
        "stop_reason": state["stop_reason"],
        "last_error": state["last_error"],
        "updated_at_utc": state["timestamp_updated_utc"],
    }


def save_state(state: dict[str, Any]) -> None:
    save_json(CHECKPOINT_PATH, state)
    save_json(SUMMARY_OUTPUT_PATH, build_summary(state))


def main() -> None:
    args = parse_arguments()

    if args.iterations <= 0:
        raise ValueError("İterasyon sayısı sıfırdan büyük olmalı.")

    if args.delay < 0:
        raise ValueError("Bekleme süresi negatif olamaz.")

    if args.patience <= 0:
        raise ValueError("Patience sıfırdan büyük olmalı.")

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip()

    if not api_key:
        raise ValueError("GROQ_API_KEY bulunamadı.")

    state = load_or_create_state(
        model=model,
        requested_iterations=args.iterations,
        patience=args.patience,
        resume=args.resume,
    )

    completed = state["completed_iterations"]

    if completed >= args.iterations:
        save_state(state)
        print("İstenen iterasyon sayısı zaten tamamlanmış.")
        print(f"Sade sonuç : {SUMMARY_OUTPUT_PATH}")
        print(f"Checkpoint : {CHECKPOINT_PATH}")
        return

    image_base64 = encode_image(IMAGE_PATH)

    if len(image_base64.encode("utf-8")) > MAX_BASE64_SIZE:
        raise ValueError(
            "Base64 görüntü Groq 4 MB sınırını aşıyor."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    problem = load_eil51_problem()
    expected_nodes = list(problem.get_nodes())

    print(f"Model             : {model}")
    print("Problem           : eil51")
    print(
        f"Başlangıç mesafe  : "
        f"{state['best_evaluation']['distance']}"
    )
    print(f"Azami iterasyon   : {args.iterations}")
    print(f"Patience          : {args.patience}")
    print(
        "Azami yeni çağrı  : "
        f"{(args.iterations - completed) * 2}"
    )

    for iteration in range(completed + 1, args.iterations + 1):
        print(f"\n=== Optimizasyon {iteration} ===")

        current_route = state["best_route"]
        current_evaluation = state["best_evaluation"]
        current_diagnostics = state["best_diagnostics"]

        print(
            f"Mevcut mesafe     : "
            f"{current_evaluation['distance']}"
        )
        print("Critic çağrısı gönderiliyor...")

        state["api_calls_attempted"] += 1

        try:
            critic_raw, critic_response = call_critic(
                client=client,
                model=model,
                image_base64=image_base64,
                current_route=current_route,
                current_evaluation=current_evaluation,
                iteration=iteration,
            )
            state["api_calls_succeeded"] += 1
        except Exception as exc:
            state["last_error"] = {
                "iteration": iteration,
                "agent": "critic",
                "message": str(exc),
                "timestamp_utc": utc_now(),
            }
            state["timestamp_updated_utc"] = utc_now()
            save_state(state)
            raise RuntimeError(
                "Critic çağrısı başarısız oldu. "
                "--resume ile devam edebilirsin."
            ) from exc

        candidate_route: list[int] | None = None
        candidate_parse_error: str | None = None

        try:
            candidate_route = extract_route(critic_raw)
            candidate_evaluation = evaluate_route(
                problem,
                candidate_route,
            )
        except ValueError as exc:
            candidate_parse_error = str(exc)
            candidate_evaluation = {
                "route": candidate_route,
                "valid": False,
                "distance": None,
                "known_optimum": KNOWN_OPTIMUM,
                "optimality_gap": None,
            }

        candidate_diagnostics = route_diagnostics(
            candidate_route,
            expected_nodes,
        )

        print(
            f"Aday geçerli      : "
            f"{candidate_evaluation['valid']}"
        )
        print(
            f"Aday mesafe       : "
            f"{candidate_evaluation['distance']}"
        )

        scorer_raw: str | None = None
        scorer_response: Any | None = None
        scorer_choice = "skipped"
        scorer_parse_error: str | None = None
        scorer_call_error: str | None = None

        candidate_valid = bool(candidate_evaluation["valid"])

        if candidate_valid:
            if args.delay:
                time.sleep(args.delay)

            print("Scorer çağrısı gönderiliyor...")
            state["api_calls_attempted"] += 1

            try:
                scorer_raw, scorer_response = call_scorer(
                    client=client,
                    model=model,
                    current_route=current_route,
                    current_distance=current_evaluation[
                        "distance"
                    ],
                    candidate_route=candidate_route,
                    candidate_distance=candidate_evaluation[
                        "distance"
                    ],
                    iteration=iteration,
                )
                state["api_calls_succeeded"] += 1

                try:
                    scorer_choice = extract_scorer_choice(
                        scorer_raw
                    )
                except ValueError as exc:
                    scorer_parse_error = str(exc)
                    scorer_choice = (
                        "candidate"
                        if candidate_evaluation["distance"]
                        < current_evaluation["distance"]
                        else "current"
                    )
            except Exception as exc:
                scorer_call_error = str(exc)
                scorer_choice = (
                    "candidate"
                    if candidate_evaluation["distance"]
                    < current_evaluation["distance"]
                    else "current"
                )

        is_strict_improvement = (
            candidate_valid
            and candidate_evaluation["distance"]
            < current_evaluation["distance"]
        )

        if is_strict_improvement:
            accepted = True
            selection_source = (
                "scorer_confirmed"
                if scorer_choice == "candidate"
                else "python_distance_guard"
            )
            state["best_route"] = candidate_route
            state["best_evaluation"] = candidate_evaluation
            state["best_diagnostics"] = candidate_diagnostics
            state["current_route"] = candidate_route
            state["current_evaluation"] = candidate_evaluation
            state["current_diagnostics"] = candidate_diagnostics
            state["non_improvement_streak"] = 0
        else:
            accepted = False
            if not candidate_valid:
                selection_source = "python_validity_guard"
            elif scorer_choice == "candidate":
                selection_source = (
                    "python_rejected_non_improvement"
                )
            else:
                selection_source = "scorer_current"
            state["non_improvement_streak"] += 1

        state["completed_iterations"] = iteration
        state["last_error"] = None
        state["timestamp_updated_utc"] = utc_now()

        state["iterations"].append(
            {
                "iteration": iteration,
                "critic": {
                    "raw_response": critic_raw,
                    "parsed_route": candidate_route,
                    "parse_error": candidate_parse_error,
                    "evaluation": candidate_evaluation,
                    "diagnostics": candidate_diagnostics,
                    "model_returned": getattr(
                        critic_response,
                        "model",
                        None,
                    ),
                    "usage": get_usage_data(critic_response),
                },
                "scorer": {
                    "raw_response": scorer_raw,
                    "parsed_choice": scorer_choice,
                    "parse_error": scorer_parse_error,
                    "call_error": scorer_call_error,
                    "model_returned": (
                        getattr(
                            scorer_response,
                            "model",
                            None,
                        )
                        if scorer_response is not None
                        else None
                    ),
                    "usage": (
                        get_usage_data(scorer_response)
                        if scorer_response is not None
                        else None
                    ),
                },
                "accepted": accepted,
                "selection_source": selection_source,
                "best_distance_after_iteration": state[
                    "best_evaluation"
                ]["distance"],
                "non_improvement_streak": state[
                    "non_improvement_streak"
                ],
            }
        )

        print(f"Scorer seçimi     : {scorer_choice}")
        print(f"Aday kabul edildi : {accepted}")
        print(f"Seçim kaynağı     : {selection_source}")
        print(
            f"En iyi mesafe     : "
            f"{state['best_evaluation']['distance']}"
        )
        print(
            f"Başarısız seri    : "
            f"{state['non_improvement_streak']}"
        )

        if state["non_improvement_streak"] >= args.patience:
            state["early_stopped"] = True
            state["stop_reason"] = (
                f"{args.patience} ardışık iterasyonda "
                "iyileşme sağlanamadı."
            )
            state["timestamp_updated_utc"] = utc_now()
            save_state(state)
            print("\nErken durdurma etkinleşti.")
            break

        save_state(state)

        if iteration < args.iterations and args.delay:
            time.sleep(args.delay)

    save_state(state)

    print("\n=== Optimizasyon tamamlandı ===")
    print(
        "API çağrısı       : "
        f"{state['api_calls_succeeded']}/"
        f"{state['api_calls_attempted']} başarılı"
    )
    print(
        f"Başlangıç mesafe  : "
        f"{state['initial_evaluation']['distance']}"
    )
    print(
        f"En iyi mesafe     : "
        f"{state['best_evaluation']['distance']}"
    )
    print(
        f"Erken durdu       : {state['early_stopped']}"
    )
    print(f"Sade sonuç        : {SUMMARY_OUTPUT_PATH}")
    print(f"Checkpoint        : {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
