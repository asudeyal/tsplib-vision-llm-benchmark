from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def replace_required(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Beklenen metin bulunamadı: {path}\n{old}")
    return text.replace(old, new)


def replace_constants_block(path: Path, new_block: str) -> None:
    text = read(path)
    pattern = re.compile(
        r"PROJECT_ROOT\s*=.*?\n\nDEFAULT_MODEL\s*=",
        flags=re.DOTALL,
    )
    if pattern.search(text) is None:
        raise RuntimeError(
            f"PROJECT_ROOT–DEFAULT_MODEL bloğu bulunamadı: {path}"
        )
    replacement = new_block.rstrip() + "\n\nDEFAULT_MODEL ="
    write(path, pattern.sub(replacement, text, count=1))


def update_imports() -> None:
    for base in (PROJECT_ROOT / "src", PROJECT_ROOT / "tests"):
        for path in base.rglob("*.py"):
            text = read(path)
            updated = text.replace(
                "from src.tsp_utils import",
                "from src.core.tsp_utils import",
            ).replace(
                "from src.run_openrouter_zero_shot import",
                "from src.providers.openrouter.run_zero_shot import",
            )
            if updated != text:
                write(path, updated)


def update_core_files() -> None:
    for relative in (
        "src/core/inspect_tsplib.py",
        "src/core/tsp_utils.py",
    ):
        path = PROJECT_ROOT / relative
        text = read(path)
        text = replace_required(
            text,
            "PROJECT_ROOT = Path(__file__).resolve().parent.parent",
            "PROJECT_ROOT = Path(__file__).resolve().parents[2]",
            path,
        )
        write(path, text)


def update_plot_eil51() -> None:
    path = PROJECT_ROOT / "src/visualization/plot_eil51.py"
    text = read(path)
    text = replace_required(
        text,
        "PROJECT_ROOT = Path(__file__).resolve().parent.parent",
        "PROJECT_ROOT = Path(__file__).resolve().parents[2]",
        path,
    )
    text = replace_required(
        text,
        'OUTPUT_DIR = PROJECT_ROOT / "output"',
        'OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"',
        path,
    )
    write(path, text)


def update_plot_route_results() -> None:
    path = PROJECT_ROOT / "src/visualization/plot_route_results.py"
    text = read(path)
    text = replace_required(
        text,
        "PROJECT_ROOT = Path(__file__).resolve().parent.parent",
        "PROJECT_ROOT = Path(__file__).resolve().parents[2]",
        path,
    )
    pattern = re.compile(
        r"RESULT_PATH\s*=.*?"
        r"OPTIMAL_ROUTE_OUTPUT\s*=\s*\(.*?\)\n",
        flags=re.DOTALL,
    )
    new_paths = """RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "openrouter"
    / "zero_shot"
    / "openrouter_zero_shot_eil51.json"
)

OPTIMAL_TOUR_PATH = (
    PROJECT_ROOT
    / "data"
    / "tsplib"
    / "eil51.opt.tour"
)

MODEL_ROUTE_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "figures"
    / "eil51_openrouter_zero_shot_route.png"
)

OPTIMAL_ROUTE_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "figures"
    / "eil51_optimal_route.png"
)
"""
    if pattern.search(text) is None:
        raise RuntimeError(
            f"Rota görselleştirme yol bloğu bulunamadı: {path}"
        )
    write(path, pattern.sub(new_paths, text, count=1))


def update_provider_files() -> None:
    replace_constants_block(
        PROJECT_ROOT / "src/providers/openrouter/run_zero_shot.py",
        """PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_PATH = (
    PROJECT_ROOT / "output" / "figures" / "eil51_nodes.png"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "openrouter"
    / "zero_shot"
    / "openrouter_zero_shot_eil51.json"
)""",
    )

    replace_constants_block(
        PROJECT_ROOT / "src/providers/openrouter/run_multi_agent.py",
        """PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_PATH = (
    PROJECT_ROOT / "output" / "figures" / "eil51_nodes.png"
)

ZERO_SHOT_RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "openrouter"
    / "zero_shot"
    / "openrouter_zero_shot_eil51.json"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "openrouter"
    / "multi_agent"
    / "openrouter_multi_agent_eil51.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "output"
    / "checkpoints"
    / "openrouter"
    / "openrouter_multi_agent_eil51_checkpoint.json"
)""",
    )

    replace_constants_block(
        PROJECT_ROOT / "src/providers/groq/run_zero_shot.py",
        """PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_PATH = (
    PROJECT_ROOT / "output" / "figures" / "eil51_nodes.png"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "groq"
    / "zero_shot"
    / "groq_zero_shot_eil51.json"
)""",
    )

    replace_constants_block(
        PROJECT_ROOT / "src/providers/groq/run_multi_agent.py",
        """PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
)""",
    )


def main() -> None:
    update_imports()
    update_core_files()
    update_plot_eil51()
    update_plot_route_results()
    update_provider_files()
    print("Importlar ve dosya yolları başarıyla güncellendi.")


if __name__ == "__main__":
    main()
