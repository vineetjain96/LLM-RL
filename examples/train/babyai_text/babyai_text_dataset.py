"""
Dataset generation entrypoint for BabyAI-Text.

Usage:
    uv run --extra babyai examples/train/babyai_text/babyai_text_dataset.py --output_dir $HOME/data/babyai_text
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).resolve().parents[3] / "skyrl-train/examples/babyai_text/babyai_text_dataset.py",
        run_name="__main__",
    )
