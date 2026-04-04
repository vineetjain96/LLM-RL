"""
BabyAI-Text training entrypoint for SkyRL.

Usage:
    uv run --isolated --extra fsdp --extra babyai -m examples.train.babyai_text.main_babyai_text

Since babyai_text is registered in skyrl-gym, this just re-exports the standard entrypoint.
You can also use the standard entrypoint directly:
    uv run --isolated --extra fsdp --extra babyai -m skyrl.train.entrypoints.main_base
"""

from skyrl.train.entrypoints.main_base import main

if __name__ == "__main__":
    main()
