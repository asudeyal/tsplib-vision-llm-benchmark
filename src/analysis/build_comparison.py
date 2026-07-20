import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_JSON = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "summary"
    / "eil51_comparison.json"
)
OUTPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "summary"
    / "eil51_comparison.csv"
)
OUTPUT_FIGURE = (
    PROJECT_ROOT
    / "output"
    / "figures"
    / "eil51_method_comparison.png"
)

KNOWN_OPTIMUM = 426

SOURCES = [
    {
        "method": "OpenRouter Nemotron zero-shot",
        "path": (
            PROJECT_ROOT
            / "output"
            / "archive"
            / "nemotron-nano-12b-v2-vl"
            / "openrouter_zero_shot_eil51.json"
        ),
        "section": "evaluation",
    },
    {
        "method": "Groq Qwen zero-shot",
        "path": (
            PROJECT_ROOT
            / "output"
            / "results"
            / "groq"
            / "zero_shot"
            / "groq_zero_shot_eil51.json"
        ),
        "section": "evaluation",
    },
    {
        "method": "Groq Qwen repair",
        "path": (
            PROJECT_ROOT
            / "output"
            / "results"
            / "groq"
            / "repair"
            / "groq_multi_agent_eil51.json"
        ),
        "section": "final",
    },
    {
        "method": "Groq Qwen optimize",
        "path": (
            PROJECT_ROOT
            / "output"
            / "results"
            / "groq"
            / "optimize"
            / "groq_optimize_eil51.json"
        ),
        "section": "final",
    },
    {
        "method": "Groq route + deterministic 2-opt",
        "path": (
            PROJECT_ROOT
            / "output"
            / "results"
            / "baselines"
            / "two_opt"
            / "eil51_from_groq_route.json"
        ),
        "section": "final",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Sonuç dosyası bulunamadı: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def extract_record(source: dict[str, Any]) -> dict[str, Any]:
    data = load_json(source["path"])
    section = data.get(source["section"], {})

    valid = bool(section.get("valid"))
    distance = section.get("distance")

    gap = section.get("gap_percent")
    if gap is None:
        gap = section.get("optimality_gap")

    if gap is None and distance is not None:
        gap = ((distance - KNOWN_OPTIMUM) / KNOWN_OPTIMUM) * 100

    return {
        "method": source["method"],
        "valid": valid,
        "distance": distance,
        "gap_percent": round(float(gap), 2) if gap is not None else None,
        "source": str(source["path"].relative_to(PROJECT_ROOT)),
    }


def write_outputs(records: list[dict[str, Any]]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "problem": "eil51",
        "known_optimum": KNOWN_OPTIMUM,
        "methods": records,
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "valid",
                "distance",
                "gap_percent",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def create_figure(records: list[dict[str, Any]]) -> None:
    valid_records = [
        record
        for record in records
        if record["valid"] and record["distance"] is not None
    ]

    labels = [record["method"] for record in valid_records]
    distances = [record["distance"] for record in valid_records]

    labels.append("TSPLIB optimum")
    distances.append(KNOWN_OPTIMUM)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(labels, distances)

    ax.set_title("eil51 — Yöntemlere Göre Rota Mesafesi")
    ax.set_ylabel("TSPLIB mesafesi")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    for bar, distance in zip(bars, distances, strict=True):
        ax.annotate(
            str(distance),
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    records = [extract_record(source) for source in SOURCES]

    write_outputs(records)
    create_figure(records)

    print("=== eil51 karşılaştırması hazır ===")
    print(f"{'Yöntem':42} {'Geçerli':8} {'Mesafe':8} {'Gap (%)':8}")

    for record in records:
        print(
            f"{record['method'][:42]:42} "
            f"{str(record['valid']):8} "
            f"{str(record['distance']):8} "
            f"{str(record['gap_percent']):8}"
        )

    print(f"\nJSON  : {OUTPUT_JSON}")
    print(f"CSV   : {OUTPUT_CSV}")
    print(f"Grafik: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
