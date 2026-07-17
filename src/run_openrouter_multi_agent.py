import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.run_openrouter_zero_shot import (
    encode_image,
    extract_json_object,
    extract_route,
    get_usage_data,
)
from src.tsp_utils import evaluate_route, load_eil51_problem


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "output" / "eil51_nodes.png"

ZERO_SHOT_RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "openrouter_zero_shot_eil51.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "openrouter_multi_agent_eil51.json"
)

DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "eil51 üzerinde OpenRouter critic-scorer "
            "multi-agent deneyi çalıştırır."
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
        help="Mevcut sonuç dosyasından devam eder.",
    )

    return parser.parse_args()


def load_zero_shot_route() -> list[int]:
    """Zero-shot deneyinin ürettiği başlangıç rotasını yükler."""
    if not ZERO_SHOT_RESULT_PATH.exists():
        raise FileNotFoundError(
            "Zero-shot sonuç dosyası bulunamadı:\n"
            f"{ZERO_SHOT_RESULT_PATH}\n"
            "Önce zero-shot deneyini çalıştır."
        )

    result = json.loads(
        ZERO_SHOT_RESULT_PATH.read_text(encoding="utf-8")
    )

    route = result.get("parsed_route")

    if not isinstance(route, list):
        raise ValueError(
            "Zero-shot sonuç dosyasında parsed_route bulunamadı."
        )

    if not all(
        isinstance(node, int) and not isinstance(node, bool)
        for node in route
    ):
        raise ValueError(
            "Zero-shot rotasındaki düğümler tam sayı olmalıdır."
        )

    return route


def build_critic_prompt(
    current_route: list[int],
    current_evaluation: dict[str, Any],
    iteration: int,
) -> str:
    """Critic ajanı için görsel rota iyileştirme promptu üretir."""
    return f"""
You are the critic agent in iteration {iteration} of a TSP optimization process.

The image contains the 51 numbered nodes of the TSPLIB eil51 benchmark.

Current route:
{json.dumps(current_route)}

Current route information:
- Valid: {current_evaluation["valid"]}
- Exact TSPLIB distance: {current_evaluation["distance"]}
- Known optimum: {current_evaluation["known_optimum"]}
- Optimality gap: {current_evaluation["optimality_gap"]}

Analyze the spatial positions of the nodes in the image and produce exactly
one improved route.

Requirements:
- Start at node 1.
- Visit every node from 1 through 51 exactly once.
- Return to node 1.
- Avoid long jumps and crossing edges.
- Prefer nearby nodes.
- Do not simply use numerical node order.
- Try to produce a route shorter than the current route.
- Do not include explanations or markdown.

Return exactly one JSON object:
{{"route": [1, ..., 1]}}
""".strip()


def build_scorer_prompt(
    current_route: list[int],
    current_evaluation: dict[str, Any],
    candidate_route: list[int] | None,
    candidate_evaluation: dict[str, Any],
    iteration: int,
) -> str:
    """Scorer ajanı için rota seçim promptu üretir."""
    return f"""
You are the scorer agent in iteration {iteration} of a TSP optimization process.

Choose which route should continue to the next iteration.

Current route:
{json.dumps(current_route)}

Current route evaluation:
- Valid: {current_evaluation["valid"]}
- Exact TSPLIB distance: {current_evaluation["distance"]}
- Optimality gap: {current_evaluation["optimality_gap"]}

Candidate route:
{json.dumps(candidate_route)}

Candidate route evaluation:
- Valid: {candidate_evaluation["valid"]}
- Exact TSPLIB distance: {candidate_evaluation["distance"]}
- Optimality gap: {candidate_evaluation["optimality_gap"]}

Selection rules:
1. A valid route must always be preferred over an invalid route.
2. If both routes are valid, choose the route with the lower exact distance.
3. Do not generate another route.
4. Do not include explanations or markdown.

Return exactly one of these JSON objects:
{{"choice": "current"}}
{{"choice": "candidate"}}
""".strip()


def call_critic(
    client: OpenAI,
    model: str,
    image_base64: str,
    current_route: list[int],
    current_evaluation: dict[str, Any],
    iteration: int,
) -> tuple[str, Any]:
    """Critic ajanına tek API isteği gönderir."""
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_critic_prompt(
                            current_route=current_route,
                            current_evaluation=current_evaluation,
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

    raw_response = extract_response_content(
        response,
        agent_name="Critic",
    )

    return raw_response, response


def call_scorer(
    client: OpenAI,
    model: str,
    current_route: list[int],
    current_evaluation: dict[str, Any],
    candidate_route: list[int] | None,
    candidate_evaluation: dict[str, Any],
    iteration: int,
) -> tuple[str, Any]:
    """Scorer ajanına tek API isteği gönderir."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": build_scorer_prompt(
                    current_route=current_route,
                    current_evaluation=current_evaluation,
                    candidate_route=candidate_route,
                    candidate_evaluation=candidate_evaluation,
                    iteration=iteration,
                ),
            }
        ],
    )

    raw_response = extract_response_content(
        response,
        agent_name="Scorer",
    )

    return raw_response, response


def extract_response_content(
    response: Any,
    agent_name: str,
) -> str:
    """OpenRouter completion metnini güvenli şekilde çıkarır."""
    choices = getattr(response, "choices", None)

    if not choices:
        if hasattr(response, "model_dump"):
            response_details = response.model_dump()
        else:
            response_details = repr(response)

        raise RuntimeError(
            f"{agent_name} boş veya eksik completion döndürdü. "
            f"Yanıt: {response_details}"
        )

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            f"{agent_name} boş metin içeriği döndürdü."
        )

    return content


def extract_scorer_choice(raw_response: str) -> str:
    """Scorer yanıtından current veya candidate seçimini çıkarır."""
    parsed = extract_json_object(raw_response)
    choice = parsed.get("choice")

    if choice not in {"current", "candidate"}:
        raise ValueError(
            "Scorer yanıtında geçerli choice bulunamadı."
        )

    return choice


def save_state(state: dict[str, Any]) -> None:
    """Deney durumunu checkpoint olarak diske yazar."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_initial_state(
    model: str,
    iterations: int,
) -> dict[str, Any]:
    """Zero-shot rotasından yeni deney durumu oluşturur."""
    problem = load_eil51_problem()
    initial_route = load_zero_shot_route()
    initial_evaluation = evaluate_route(problem, initial_route)

    if not initial_evaluation["valid"]:
        raise ValueError(
            "Zero-shot başlangıç rotası geçerli değil."
        )

    return {
        "experiment": "openrouter_multi_agent_critic_scorer",
        "problem": "eil51",
        "model_requested": model,
        "timestamp_started_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "requested_iterations": iterations,
        "completed_iterations": 0,
        "api_calls_per_iteration": 2,
        "total_api_calls": 0,
        "initial_route": initial_route,
        "initial_evaluation": initial_evaluation,
        "current_route": initial_route,
        "current_evaluation": initial_evaluation,
        "iterations": [],
    }


def load_or_create_state(
    model: str,
    iterations: int,
    resume: bool,
) -> dict[str, Any]:
    """Yeni deney başlatır veya checkpoint'ten devam eder."""
    if resume and OUTPUT_PATH.exists():
        state = json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )

        previous_model = state.get("model_requested")

        if previous_model != model:
            raise ValueError(
                "Checkpoint modeli ile seçilen model farklı.\n"
                f"Checkpoint: {previous_model}\n"
                f"Seçilen   : {model}"
            )

        state["requested_iterations"] = iterations
        return state

    return create_initial_state(
        model=model,
        iterations=iterations,
    )


def main() -> None:
    args = parse_arguments()

    if args.iterations <= 0:
        raise ValueError(
            "İterasyon sayısı sıfırdan büyük olmalıdır."
        )

    if args.delay < 0:
        raise ValueError(
            "Bekleme süresi negatif olamaz."
        )

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY bulunamadı."
        )

    image_base64 = encode_image(IMAGE_PATH)
    problem = load_eil51_problem()

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    state = load_or_create_state(
        model=model,
        iterations=args.iterations,
        resume=args.resume,
    )

    completed_iterations = state["completed_iterations"]

    if completed_iterations >= args.iterations:
        print(
            "İstenen iterasyon sayısı zaten tamamlanmış."
        )
        print(f"Sonuç dosyası: {OUTPUT_PATH}")
        return

    print(f"Model              : {model}")
    print("Problem            : eil51")
    print(f"Hedef iterasyon    : {args.iterations}")
    print(f"Tamamlanan         : {completed_iterations}")
    print("Maks. API / iter.  : 2")
    print(
        f"Azami yeni çağrı   : "
        f"{(args.iterations - completed_iterations) * 2}"
    )

    for iteration in range(
        completed_iterations + 1,
        args.iterations + 1,
    ):
        print(f"\n=== İterasyon {iteration} ===")

        current_route = state["current_route"]
        current_evaluation = state["current_evaluation"]

        print(
            "Critic çağrısı gönderiliyor... "
            f"Mevcut mesafe: {current_evaluation['distance']}"
        )

        critic_raw = None
        critic_response = None
        critic_call_error = None
        candidate_route = None
        candidate_parse_error = None

        # API isteği denendiği için çağrı sayısını önceden artır.
        state["total_api_calls"] += 1

        try:
            critic_raw, critic_response = call_critic(
                client=client,
                model=model,
                image_base64=image_base64,
                current_route=current_route,
                current_evaluation=current_evaluation,
                iteration=iteration,
            )

            try:
                candidate_route = extract_route(critic_raw)
                candidate_evaluation = evaluate_route(
                    problem,
                    candidate_route,
                )
            except ValueError as exc:
                candidate_parse_error = str(exc)
                candidate_evaluation = {
                    "route": None,
                    "valid": False,
                    "distance": None,
                    "known_optimum": 426,
                    "optimality_gap": None,
                }

        except RuntimeError as exc:
            # OpenRouter boş/eksik completion döndürürse deneyi çökertme.
            critic_call_error = str(exc)
            candidate_parse_error = critic_call_error
            candidate_evaluation = {
                "route": None,
                "valid": False,
                "distance": None,
                "known_optimum": 426,
                "optimality_gap": None,
            }

        print(f"Aday geçerli : {candidate_evaluation['valid']}")
        print(f"Aday mesafe  : {candidate_evaluation['distance']}")

        if candidate_parse_error:
            print(f"Aday hatası  : {candidate_parse_error}")

        scorer_raw = None
        scorer_response = None
        scorer_parse_error = None
        scorer_skipped = False

        if not candidate_evaluation["valid"]:
            # Geçersiz aday mevcut geçerli rotayı yenemez.
            scorer_skipped = True
            scorer_choice = "current"
            print(
                "Scorer çağrısı atlandı: "
                "critic adayı geçersiz."
            )

        else:
            if args.delay:
                time.sleep(args.delay)

            print("Scorer çağrısı gönderiliyor...")

            # API isteği denendiği için çağrı sayısını önceden artır.
            state["total_api_calls"] += 1

            try:
                scorer_raw, scorer_response = call_scorer(
                    client=client,
                    model=model,
                    current_route=current_route,
                    current_evaluation=current_evaluation,
                    candidate_route=candidate_route,
                    candidate_evaluation=candidate_evaluation,
                    iteration=iteration,
                )

                scorer_choice = extract_scorer_choice(
                    scorer_raw
                )

            except (RuntimeError, ValueError) as exc:
                scorer_parse_error = str(exc)

                # Scorer yanıtı kullanılamazsa kesin TSPLIB
                # mesafelerine göre deterministik seçim yap.
                if (
                    candidate_evaluation["distance"]
                    < current_evaluation["distance"]
                ):
                    scorer_choice = "candidate"
                else:
                    scorer_choice = "current"

                print(
                    "Scorer yanıtı kullanılamadı; "
                    "Python mesafe karşılaştırması uygulandı."
                )
                print(f"Scorer hatası: {scorer_parse_error}")

        # Geçersiz adayın seçilmesine izin verme.
        if (
            scorer_choice == "candidate"
            and candidate_evaluation["valid"]
        ):
            accepted_route = candidate_route
            accepted_evaluation = candidate_evaluation
            accepted_choice = "candidate"
        else:
            accepted_route = current_route
            accepted_evaluation = current_evaluation
            accepted_choice = "current"

        iteration_result = {
            "iteration": iteration,
            "critic": {
                "raw_response": critic_raw,
                "parsed_route": candidate_route,
                "parse_error": candidate_parse_error,
                "call_error": critic_call_error,
                "evaluation": candidate_evaluation,
                "model_returned": (
                    getattr(critic_response, "model", None)
                    if critic_response is not None
                    else None
                ),
                "usage": (
                    get_usage_data(critic_response)
                    if critic_response is not None
                    else None
                ),
            },
            "scorer": {
                "skipped": scorer_skipped,
                "raw_response": scorer_raw,
                "parsed_choice": scorer_choice,
                "parse_error": scorer_parse_error,
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
            "accepted_route": accepted_route,
            "accepted_evaluation": accepted_evaluation,
        }

        state["iterations"].append(iteration_result)
        state["current_route"] = accepted_route
        state["current_evaluation"] = accepted_evaluation
        state["completed_iterations"] = iteration
        state["timestamp_updated_utc"] = datetime.now(
            timezone.utc
        ).isoformat()

        save_state(state)

        print(f"Scorer seçimi : {scorer_choice}")
        print(f"Kabul edilen  : {accepted_choice}")
        print(
            f"Yeni mesafe   : "
            f"{accepted_evaluation['distance']}"
        )
        print(
            f"Yeni gap (%)  : "
            f"{accepted_evaluation['optimality_gap']}"
        )

        if iteration < args.iterations and args.delay:
            time.sleep(args.delay)

    print("\n=== Deney tamamlandı ===")
    print(
        f"Toplam API çağrısı : {state['total_api_calls']}"
    )
    print(
        f"Son rota mesafesi  : "
        f"{state['current_evaluation']['distance']}"
    )
    print(
        f"Son gap (%)        : "
        f"{state['current_evaluation']['optimality_gap']}"
    )
    print(f"Sonuç dosyası      : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()