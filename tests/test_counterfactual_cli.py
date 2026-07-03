import json
import subprocess
import sys


def test_collect_counterfactual_writes_jsonl_and_inspect_counts(tmp_path) -> None:
    db = tmp_path / "memory.sqlite"
    output = tmp_path / "paired.jsonl"

    collect = subprocess.run(
        [
            sys.executable,
            "-m",
            "smtr.cli",
            "collect-counterfactual",
            "--db",
            str(db),
            "--episodes",
            "4",
            "--seed",
            "7",
            "--top-k",
            "4",
            "--scenario-mix",
            "balanced",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "record_count=4" in collect.stdout

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert {record["transfer_class"] for record in records} == {
        "positive",
        "negative",
        "neutral_success",
        "neutral_failure",
    }

    inspect = subprocess.run(
        [
            sys.executable,
            "-m",
            "smtr.cli",
            "inspect-paired-records",
            "--input",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "record count: 4" in inspect.stdout
    assert "positive rate: 0.250" in inspect.stdout
    assert "negative rate: 0.250" in inspect.stdout
