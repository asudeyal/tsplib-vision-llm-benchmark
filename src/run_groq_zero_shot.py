import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.run_openrouter_zero_shot import (
    encode_image,
    extract_route,
    get_usage_data,
)
from src.tsp_utils import evaluate_route, load_eil51_problem


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "output" / "eil51_nodes.png"
OUTPUT_PATH = PROJECT_ROOT / "output" / "groq_zero_shot_eil51.json"

DEFAULT_MODEL = "qwen/qwen3.6-27b"
MAX_BASE64_SIZE = 4 * 1024 * 1024


def build_prompt() -> str:
    return """
Solve the symmetric Traveling Salesman Problem shown in the image.

The image contains exactly 51 numbered nodes belonging to the TSPLIB
eil51 benchmark.

Requirements:
- Start at node 1.
- Visit every integer node from 1 through 51 exactly once.
- Return to node 1 at the end.
- The route must contain exactly 52 integers.
- Do not use nodes below 1 or above 51.
- Avoid long jumps and crossing edges.
- Prefer nearby nodes based on the image.
- Do not use simple numerical order.
- Return no explanation.

Return exactly this JSON structure:
{"route": [1, ..., 1]}
""".strip()


def extract_response_content(response: Any) -> str:
    choices = getattr(response, "choices", None)

    if not choices:
        raise RuntimeError(
            "Groq boş veya eksik completion döndürdü."
        )

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq boş metin yanıtı döndürdü.")

    return content


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY bulunamadı. .env dosyasını kontrol et."
        )

    image_base64 = encode_image(IMAGE_PATH)

    if len(image_base64.encode("utf-8")) > MAX_BASE64_SIZE:
        raise ValueError(
            "Base64 görüntü 4 MB sınırını aşıyor. "
            "Görselin DPI değerini düşür."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    print(f"Model       : {model}")
    print("Problem     : eil51")
    print(f"Görsel      : {IMAGE_PATH}")
    print("Groq isteği gönderiliyor...")

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"},
            extra_body={
                "reasoning_effort": "none",
            },
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
            f"Groq isteği başarısız oldu: {exc}"
        ) from exc

    raw_response = extract_response_content(response)

    print("\nHam model yanıtı:")
    print(raw_response)

    problem = load_eil51_problem()
    route = None
    parse_error = None

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
        "experiment": "groq_zero_shot",
        "problem": "eil51",
        "model_requested": model,
        "model_returned": getattr(response, "model", None),
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