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

from src.providers.openrouter.run_zero_shot import (
    encode_image,
    extract_json_object,
    extract_route,
    get_usage_data,
)
from src.core.tsp_utils import evaluate_route, load_eil51_problem


PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_PATH = (
    PROJECT_ROOT / "output" / "figures" / "eil51_nodes.png"
)
ZERO_SHOT_RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "groq"
    / "zero_shot"
    / "groq_zero_shot_eil51.json"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "groq"
    / "repair"
    / "groq_multi_agent_eil51.json"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "output"
    / "checkpoints"
    / "groq"
    / "groq_multi_agent_eil51_checkpoint.json"
)

DEFAULT_MODEL = "qwen/qwen3.6-27b"
KNOWN_OPTIMUM = 426
MAX_BASE64_SIZE = 4 * 1024 * 1024


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Groq üzerinde eil51 critic-scorer multi-agent "
            "deneyi çalıştırır."
        )
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Toplam critic-scorer iterasyon sayısı.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="API çağrıları arasındaki bekleme süresi.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Mevcut checkpoint dosyasından devam eder.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_evaluation(route: list[int] | None = None) -> dict[str, Any]:
    return {
        "route": route,
        "valid": False,
        "distance": None,
        "known_optimum": KNOWN_OPTIMUM,
        "optimality_gap": None,
    }


def route_diagnostics(
    route: Sequence[int] | None,
    expected_nodes: Sequence[int],
) -> dict[str, Any]:
    """Geçersiz rotanın yapısal sorunlarını ayrıntılı biçimde çıkarır."""
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
            "issue_score": expected_length + len(nodes),
        }

    route_list = list(route)
    closed = (
        len(route_list) >= 2
        and route_list[0] == route_list[-1]
    )

    visited = route_list[:-1] if closed else route_list
    counts = Counter(visited)

    missing_nodes = sorted(expected_set - set(visited))
    duplicate_nodes = sorted(
        node
        for node, count in counts.items()
        if node in expected_set and count > 1
    )
    duplicate_excess = sum(
        count - 1
        for node, count in counts.items()
        if node in expected_set and count > 1
    )
    out_of_range_nodes = sorted(
        {
            node
            for node in visited
            if node not in expected_set
        }
    )

    length_error = abs(len(route_list) - expected_length)
    closure_error = 0 if closed else 1

    issue_score = (
        len(missing_nodes)
        + duplicate_excess
        + len(out_of_range_nodes)
        + length_error
        + closure_error
    )

    return {
        "route_length": len(route_list),
        "expected_route_length": expected_length,
        "closed": closed,
        "missing_nodes": missing_nodes,
        "duplicate_nodes": duplicate_nodes,
        "out_of_range_nodes": out_of_range_nodes,
        "issue_score": issue_score,
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
        "issue_score": diagnostics["issue_score"],
    }


def load_zero_shot_route() -> list[int]:
    if not ZERO_SHOT_RESULT_PATH.exists():
        raise FileNotFoundError(
            "Groq zero-shot sonuç dosyası bulunamadı:\n"
            f"{ZERO_SHOT_RESULT_PATH}\n"
            "Önce python -m src.run_groq_zero_shot çalıştır."
        )

    result = json.loads(
        ZERO_SHOT_RESULT_PATH.read_text(encoding="utf-8")
    )
    route = result.get("parsed_route")

    if not isinstance(route, list):
        raise ValueError(
            "Groq zero-shot sonucunda parsed_route listesi yok."
        )

    if not all(
        isinstance(node, int) and not isinstance(node, bool)
        for node in route
    ):
        raise ValueError(
            "Zero-shot rotasındaki bütün düğümler tam sayı olmalı."
        )

    return route


def build_critic_prompt(
    current_route: list[int],
    current_evaluation: dict[str, Any],
    current_diagnostics: dict[str, Any],
    iteration: int,
) -> str:
    repair_instruction = ""

    if not current_evaluation["valid"]:
        repair_instruction = f"""
The current route is INVALID.

Repair information:
- Missing nodes: {current_diagnostics["missing_nodes"]}
- Duplicate nodes: {current_diagnostics["duplicate_nodes"]}
- Out-of-range nodes: {current_diagnostics["out_of_range_nodes"]}
- Current route length: {current_diagnostics["route_length"]}
- Expected route length: {current_diagnostics["expected_route_length"]}
- Closed route: {current_diagnostics["closed"]}

Your first priority is to repair every structural error.
After repairing it, use the image to make the route spatially efficient.
""".strip()

    return f"""
You are the critic agent in iteration {iteration} of an eil51 TSP
optimization experiment.

The image contains exactly 51 numbered nodes, with valid node IDs
from 1 through 51.

Current route:
{json.dumps(current_route)}

Current evaluation:
- Valid: {current_evaluation["valid"]}
- Exact TSPLIB distance: {current_evaluation["distance"]}
- Known optimum: {current_evaluation["known_optimum"]}
- Optimality gap: {current_evaluation["optimality_gap"]}

{repair_instruction}

Produce exactly one candidate route.

Hard constraints:
- Start at node 1.
- End at node 1.
- Visit each integer node from 1 through 51 exactly once.
- The route must contain exactly 52 integers.
- Never use a node below 1 or above 51.
- Do not omit or duplicate any node except the final return to node 1.
- Avoid long jumps and crossing edges.
- Prefer nearby nodes based on the image.
- Do not use simple numerical order.
- Return no explanation and no markdown.

Return exactly one JSON object:
{{"route": [1, ..., 1]}}
""".strip()


def build_scorer_prompt(
    current_route: list[int],
    current_evaluation: dict[str, Any],
    current_diagnostics: dict[str, Any],
    candidate_route: list[int],
    candidate_evaluation: dict[str, Any],
    candidate_diagnostics: dict[str, Any],
    iteration: int,
) -> str:
    return f"""
You are the scorer agent in iteration {iteration} of an eil51 TSP
optimization experiment.

Choose which route should continue.

CURRENT ROUTE:
{json.dumps(current_route)}

CURRENT METRICS:
- Valid: {current_evaluation["valid"]}
- Distance: {current_evaluation["distance"]}
- Missing: {current_diagnostics["missing_nodes"]}
- Duplicates: {current_diagnostics["duplicate_nodes"]}
- Out of range: {current_diagnostics["out_of_range_nodes"]}
- Issue score: {current_diagnostics["issue_score"]}

CANDIDATE ROUTE:
{json.dumps(candidate_route)}

CANDIDATE METRICS:
- Valid: {candidate_evaluation["valid"]}
- Distance: {candidate_evaluation["distance"]}
- Missing: {candidate_diagnostics["missing_nodes"]}
- Duplicates: {candidate_diagnostics["duplicate_nodes"]}
- Out of range: {candidate_diagnostics["out_of_range_nodes"]}
- Issue score: {candidate_diagnostics["issue_score"]}

Selection rules:
1. Prefer a valid route over an invalid route.
2. If both are valid, prefer the lower exact distance.
3. If both are invalid, prefer the lower issue score.
4. Do not create a new route.
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
            f"{agent_name} boş completion döndürdü. Yanıt: {details}"
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
    current_diagnostics: dict[str, Any],
    iteration: int,
) -> tuple[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=1800,
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
                            current_evaluation=current_evaluation,
                            current_diagnostics=current_diagnostics,
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
    current_evaluation: dict[str, Any],
    current_diagnostics: dict[str, Any],
    candidate_route: list[int],
    candidate_evaluation: dict[str, Any],
    candidate_diagnostics: dict[str, Any],
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
                    current_evaluation=current_evaluation,
                    current_diagnostics=current_diagnostics,
                    candidate_route=candidate_route,
                    candidate_evaluation=candidate_evaluation,
                    candidate_diagnostics=candidate_diagnostics,
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


def deterministic_choice(
    current_evaluation: dict[str, Any],
    current_diagnostics: dict[str, Any],
    candidate_evaluation: dict[str, Any],
    candidate_diagnostics: dict[str, Any],
) -> str:
    current_valid = bool(current_evaluation["valid"])
    candidate_valid = bool(candidate_evaluation["valid"])

    if candidate_valid and not current_valid:
        return "candidate"

    if current_valid and not candidate_valid:
        return "current"

    if current_valid and candidate_valid:
        candidate_distance = candidate_evaluation["distance"]
        current_distance = current_evaluation["distance"]

        if candidate_distance < current_distance:
            return "candidate"

        return "current"

    if (
        candidate_diagnostics["issue_score"]
        < current_diagnostics["issue_score"]
    ):
        return "candidate"

    return "current"


def resolve_choice(
    scorer_choice: str,
    deterministic: str,
    current_evaluation: dict[str, Any],
    candidate_evaluation: dict[str, Any],
    current_diagnostics: dict[str, Any],
    candidate_diagnostics: dict[str, Any],
) -> tuple[str, str]:
    """
    Açıkça daha iyi olan rotayı Python korur.
    Eşit kalite durumunda scorer kararı kullanılır.
    """
    current_valid = bool(current_evaluation["valid"])
    candidate_valid = bool(candidate_evaluation["valid"])

    if current_valid != candidate_valid:
        return deterministic, "python_validity_guard"

    if current_valid and candidate_valid:
        current_distance = current_evaluation["distance"]
        candidate_distance = candidate_evaluation["distance"]

        if current_distance != candidate_distance:
            return deterministic, "python_distance_guard"

        return scorer_choice, "scorer_tie_break"

    current_score = current_diagnostics["issue_score"]
    candidate_score = candidate_diagnostics["issue_score"]

    if current_score != candidate_score:
        return deterministic, "python_repair_guard"

    return scorer_choice, "scorer_tie_break"


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_initial_state(
    model: str,
    requested_iterations: int,
) -> dict[str, Any]:
    problem = load_eil51_problem()
    nodes = list(problem.get_nodes())
    initial_route = load_zero_shot_route()
    initial_evaluation = evaluate_route(problem, initial_route)
    initial_diagnostics = route_diagnostics(initial_route, nodes)

    return {
        "experiment": "groq_multi_agent_critic_scorer",
        "problem": "eil51",
        "model_requested": model,
        "requested_iterations": requested_iterations,
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
        "iterations": [],
        "last_error": None,
    }


def load_or_create_state(
    model: str,
    requested_iterations: int,
    resume: bool,
) -> dict[str, Any]:
    if resume and CHECKPOINT_PATH.exists():
        state = json.loads(
            CHECKPOINT_PATH.read_text(encoding="utf-8")
        )

        if state.get("model_requested") != model:
            raise ValueError(
                "Checkpoint modeli .env modelinden farklı.\n"
                f"Checkpoint: {state.get('model_requested')}\n"
                f".env      : {model}"
            )

        state["requested_iterations"] = requested_iterations
        return state

    return create_initial_state(model, requested_iterations)


def build_summary(state: dict[str, Any]) -> dict[str, Any]:
    initial = compact_evaluation(
        state["initial_evaluation"],
        state["initial_diagnostics"],
    )
    final = compact_evaluation(
        state["current_evaluation"],
        state["current_diagnostics"],
    )

    steps: list[dict[str, Any]] = []

    for item in state["iterations"]:
        candidate = compact_evaluation(
            item["critic"]["evaluation"],
            item["critic"]["diagnostics"],
        )
        accepted = compact_evaluation(
            item["accepted_evaluation"],
            item["accepted_diagnostics"],
        )

        steps.append(
            {
                "iteration": item["iteration"],
                "candidate": candidate,
                "scorer_choice": item["scorer"][
                    "parsed_choice"
                ],
                "accepted_choice": item["accepted_choice"],
                "selection_source": item["selection_source"],
                "result": accepted,
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
        },
        "api_calls": {
            "attempted": state["api_calls_attempted"],
            "succeeded": state["api_calls_succeeded"],
        },
        "known_optimum": KNOWN_OPTIMUM,
        "initial": initial,
        "steps": steps,
        "final": {
            **final,
            "distance_improvement": (
                initial_distance - final_distance
                if initial_distance is not None
                and final_distance is not None
                else None
            ),
            "route": state["current_route"],
        },
        "last_error": state.get("last_error"),
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

    load_dotenv(PROJECT_ROOT / ".env")

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip()

    state = load_or_create_state(
        model=model,
        requested_iterations=args.iterations,
        resume=args.resume,
    )

    completed = state["completed_iterations"]

    if completed >= args.iterations:
        save_state(state)
        print("İstenen iterasyon sayısı zaten tamamlanmış.")
        print(f"Sade sonuç : {SUMMARY_OUTPUT_PATH}")
        print(f"Checkpoint : {CHECKPOINT_PATH}")
        return

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise ValueError("GROQ_API_KEY bulunamadı.")

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

    print(f"Model              : {model}")
    print("Problem            : eil51")
    print(f"Hedef iterasyon    : {args.iterations}")
    print(f"Tamamlanan         : {completed}")
    print("Maks. API / iter.  : 2")
    print(
        "Azami yeni çağrı   : "
        f"{(args.iterations - completed) * 2}"
    )

    for iteration in range(completed + 1, args.iterations + 1):
        print(f"\n=== İterasyon {iteration} ===")

        current_route = state["current_route"]
        current_evaluation = state["current_evaluation"]
        current_diagnostics = state["current_diagnostics"]

        print(
            "Critic çağrısı gönderiliyor... "
            f"Geçerli: {current_evaluation['valid']} | "
            f"Mesafe: {current_evaluation['distance']} | "
            f"Eksik: {current_diagnostics['missing_nodes']}"
        )

        state["api_calls_attempted"] += 1

        try:
            critic_raw, critic_response = call_critic(
                client=client,
                model=model,
                image_base64=image_base64,
                current_route=current_route,
                current_evaluation=current_evaluation,
                current_diagnostics=current_diagnostics,
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
                "Aynı iterasyonu --resume ile yeniden deneyebilirsin.\n"
                f"{exc}"
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
            candidate_evaluation = empty_evaluation(
                candidate_route
            )

        candidate_diagnostics = route_diagnostics(
            candidate_route,
            expected_nodes,
        )

        print(
            f"Aday geçerli : {candidate_evaluation['valid']}"
        )
        print(
            f"Aday mesafe  : {candidate_evaluation['distance']}"
        )
        print(
            f"Aday eksik   : "
            f"{candidate_diagnostics['missing_nodes']}"
        )
        print(
            f"Aday sorun   : "
            f"{candidate_diagnostics['issue_score']}"
        )

        scorer_raw: str | None = None
        scorer_response: Any | None = None
        scorer_parse_error: str | None = None
        scorer_call_error: str | None = None

        if candidate_route is None:
            scorer_choice = "current"
            selection_source = "critic_parse_failure"
            accepted_choice = "current"
        else:
            if args.delay:
                time.sleep(args.delay)

            print("Scorer çağrısı gönderiliyor...")
            state["api_calls_attempted"] += 1

            try:
                scorer_raw, scorer_response = call_scorer(
                    client=client,
                    model=model,
                    current_route=current_route,
                    current_evaluation=current_evaluation,
                    current_diagnostics=current_diagnostics,
                    candidate_route=candidate_route,
                    candidate_evaluation=candidate_evaluation,
                    candidate_diagnostics=candidate_diagnostics,
                    iteration=iteration,
                )
                state["api_calls_succeeded"] += 1

                try:
                    scorer_choice = extract_scorer_choice(
                        scorer_raw
                    )
                except ValueError as exc:
                    scorer_parse_error = str(exc)
                    scorer_choice = deterministic_choice(
                        current_evaluation,
                        current_diagnostics,
                        candidate_evaluation,
                        candidate_diagnostics,
                    )
            except Exception as exc:
                scorer_call_error = str(exc)
                scorer_choice = deterministic_choice(
                    current_evaluation,
                    current_diagnostics,
                    candidate_evaluation,
                    candidate_diagnostics,
                )

            deterministic = deterministic_choice(
                current_evaluation,
                current_diagnostics,
                candidate_evaluation,
                candidate_diagnostics,
            )
            accepted_choice, selection_source = resolve_choice(
                scorer_choice=scorer_choice,
                deterministic=deterministic,
                current_evaluation=current_evaluation,
                candidate_evaluation=candidate_evaluation,
                current_diagnostics=current_diagnostics,
                candidate_diagnostics=candidate_diagnostics,
            )

        if accepted_choice == "candidate":
            accepted_route = candidate_route
            accepted_evaluation = candidate_evaluation
            accepted_diagnostics = candidate_diagnostics
        else:
            accepted_route = current_route
            accepted_evaluation = current_evaluation
            accepted_diagnostics = current_diagnostics

        iteration_result = {
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
                    getattr(scorer_response, "model", None)
                    if scorer_response is not None
                    else None
                ),
                "usage": (
                    get_usage_data(scorer_response)
                    if scorer_response is not None
                    else None
                ),
            },
            "accepted_choice": accepted_choice,
            "selection_source": selection_source,
            "accepted_route": accepted_route,
            "accepted_evaluation": accepted_evaluation,
            "accepted_diagnostics": accepted_diagnostics,
        }

        state["iterations"].append(iteration_result)
        state["current_route"] = accepted_route
        state["current_evaluation"] = accepted_evaluation
        state["current_diagnostics"] = accepted_diagnostics
        state["completed_iterations"] = iteration
        state["last_error"] = None
        state["timestamp_updated_utc"] = utc_now()

        save_state(state)

        print(f"Scorer seçimi : {scorer_choice}")
        print(f"Kabul edilen  : {accepted_choice}")
        print(f"Seçim kaynağı : {selection_source}")
        print(
            f"Yeni geçerli  : {accepted_evaluation['valid']}"
        )
        print(
            f"Yeni mesafe   : {accepted_evaluation['distance']}"
        )
        print(
            f"Yeni eksik    : "
            f"{accepted_diagnostics['missing_nodes']}"
        )

        if iteration < args.iterations and args.delay:
            time.sleep(args.delay)

    print("\n=== Deney tamamlandı ===")
    print(
        "API çağrısı        : "
        f"{state['api_calls_succeeded']}/"
        f"{state['api_calls_attempted']} başarılı"
    )
    print(
        f"Son rota geçerli   : "
        f"{state['current_evaluation']['valid']}"
    )
    print(
        f"Son rota mesafesi  : "
        f"{state['current_evaluation']['distance']}"
    )
    print(
        f"Son eksik düğümler : "
        f"{state['current_diagnostics']['missing_nodes']}"
    )
    print(f"Sade sonuç         : {SUMMARY_OUTPUT_PATH}")
    print(f"Checkpoint         : {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
