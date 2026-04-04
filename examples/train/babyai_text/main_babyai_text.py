"""
BabyAI-Text training entrypoint for SkyRL.

Usage:
    uv run --isolated --extra fsdp --extra babyai -m examples.train.babyai_text.main_babyai_text
"""

from skyrl.train.entrypoints.main_base import main


if __name__ == "__main__":
    main()
