import subprocess
import sys


def test_train_and_evaluate_transfer_critic_cli(tmp_path) -> None:
    db = tmp_path / "memory.sqlite"
    records = tmp_path / "records.jsonl"
    checkpoint = tmp_path / "critic.joblib"
    eval_output = tmp_path / "eval.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "smtr.cli",
            "collect-counterfactual",
            "--db",
            str(db),
            "--episodes",
            "8",
            "--seed",
            "7",
            "--top-k",
            "4",
            "--scenario-mix",
            "balanced",
            "--target-policy",
            "scenario-designated",
            "--prefix-mode",
            "stratified",
            "--max-prefix-size",
            "2",
            "--output",
            str(records),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "smtr.cli",
            "train-transfer-critic",
            "--input",
            str(records),
            "--output",
            str(checkpoint),
            "--seed",
            "7",
            "--n-bootstrap",
            "3",
            "--test-fraction",
            "0.25",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "smtr.cli",
            "evaluate-transfer-critic",
            "--input",
            str(records),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(eval_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert checkpoint.exists()
    assert checkpoint.with_suffix(".metadata.json").exists()
    assert checkpoint.with_suffix(".metrics.json").exists()
    assert eval_output.exists()
