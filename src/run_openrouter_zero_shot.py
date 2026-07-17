import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.tsp_utils import evaluate_route, load_eil51_problem


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "output" / "eil51_nodes.png"
OUTPUT_PATH = PROJECT_ROOT / "output" / "openrouter_zero_shot_eil51.json"

DEFAULT_MODEL = "qwen/qwen2.5-vl-32b-instruct:free"


def encode_image(image_path: Path) -> str:
    """Yerel PNG dosyasını base64 metnine dönüştürür."""
    if not image_path.exists():
        raise FileNotFoundError(
            f"Görsel bulunamadı: {image_path}\n"
            "Önce python src\\plot_eil51.py komutunu çalıştır."
        )

    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    """Model yanıtındaki JSON nesnesini ayıklar."""
    cleaned_text = text.strip()

    # Markdown kod bloğu işaretlerini temizle.
    cleaned_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    ).strip()

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned_text, flags=re.DOTALL)

        if match is None:
            raise ValueError(
                "Model yanıtında JSON nesnesi bulunamadı."
            )

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Model yanıtındaki JSON ayrıştırılamadı."
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Model yanıtı bir JSON nesnesi olmalıdır.")

    return parsed


def extract_route(raw_response: str) -> list[int]:
    """Model yanıtından rota listesini çıkarır."""
    parsed = extract_json_object(raw_response)
    route = parsed.get("route")

    if not isinstance(route, list):
        raise ValueError("JSON yanıtında 'route' listesi bulunamadı.")

    if not all(
        isinstance(node, int) and not isinstance(node, bool)
        for node in route
    ):
        raise ValueError("Rotadaki bütün düğümler tam sayı olmalıdır.")

    return route


def build_prompt() -> str:
    return """
You are solving the symmetric Traveling Salesman Problem shown in the image.

The image contains the 51 numbered nodes of the TSPLIB eil51 benchmark.

Requirements:
- Start at node 1.
- Visit every node from 1 through 51 exactly once.
- Return to node 1 at the end.
- Minimize the total route distance.
- Infer a short route from the spatial positions of the nodes.
- Do not omit or repeat any node, except node 1 at the end.
- Return no explanation.

Return exactly one JSON object in this format:
{"route": [1, 2, 3, ..., 51, 1]}
""".strip()


def get_usage_data(response: Any) -> dict[str, Any] | None:
    """Varsa token kullanım bilgilerini sözlüğe dönüştürür."""
    usage = getattr(response, "usage", None)

    if usage is None:
        return None

    if hasattr(usage, "model_dump"):
        return usage.model_dump()

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY bulunamadı. .env dosyasını kontrol et."
        )

    image_base64 = encode_image(IMAGE_PATH)

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    print(f"Model       : {model}")
    print(f"Problem     : eil51")
    print(f"Görsel      : {IMAGE_PATH}")
    print("OpenRouter isteği gönderiliyor...")

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_prompt(),
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
    except Exception as exc:
        raise RuntimeError(
            f"OpenRouter isteği başarısız oldu: {exc}"
        ) from exc

    raw_response = response.choices[0].message.content

    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("Model boş veya geçersiz bir yanıt döndürdü.")

    print("\nHam model yanıtı:")
    print(raw_response)

    problem = load_eil51_problem()

    parse_error = None
    route = None

    try:
        route = extract_route(raw_response)
        evaluation = evaluate_route(problem, route)
    except ValueError as exc:
        parse_error = str(exc)
        evaluation = {
            "route": None,
            "valid": False,
            "distance": None,
            "known_optimum": 426,
            "optimality_gap": None,
        }

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "openrouter_zero_shot",
        "problem": "eil51",
        "model_requested": model,
        "model_returned": getattr(response, "model", None),
        "image_path": str(IMAGE_PATH.relative_to(PROJECT_ROOT)),
        "prompt": build_prompt(),
        "raw_response": raw_response,
        "parsed_route": route,
        "parse_error": parse_error,
        "evaluation": evaluation,
        "usage": get_usage_data(response),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDeğerlendirme:")
    print(f"Geçerli rota : {evaluation['valid']}")
    print(f"Mesafe       : {evaluation['distance']}")
    print(f"Optimum      : {evaluation['known_optimum']}")
    print(f"Gap (%)      : {evaluation['optimality_gap']}")

    if parse_error:
        print(f"Parse hatası : {parse_error}")

    print(f"\nSonuç dosyası: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()