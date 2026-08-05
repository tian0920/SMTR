"""清单 Test 9: README Stage D checkpoint completeness.

The end-to-end (Stage D) command enables critic ablation methods, so it
must pass every required checkpoint plus the formal experiment mode and
five generation seeds; otherwise the documented run cannot be formal.
"""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"


def _stage_d_command() -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(
        r"## Stage D.*?```bash\n(.*?)```", text, flags=re.DOTALL
    )
    assert match is not None, "README must contain a Stage D bash block"
    return match.group(1)


def test_stage_d_passes_all_critic_checkpoints():
    command = _stage_d_command()
    for flag in (
        "--checkpoint-full",
        "--checkpoint-global-transfer-critic",
        "--checkpoint-smtr-no-pair-interaction",
    ):
        assert flag in command, f"Stage D command missing {flag}"


def test_stage_d_uses_formal_mode_and_five_seeds():
    command = _stage_d_command()
    assert "--experiment-mode formal" in command
    assert "--generation-seeds 0 1 2 3 4" in command
