"""Default risk budget resolves from the checkpoint epsilon_star (清单 P1-2 5.1, 5.6).

With ``negative_risk_budget=None`` the evaluation must record that the
budget came from the validation-selected epsilon_star stored in each
method's own checkpoint.
"""

from __future__ import annotations

from smtr.marble.end_to_end_evaluation import run_end_to_end_evaluation
from smtr.marble.runtime_visibility_audit import file_digest
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def _setup(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"tasks": []}', encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text('{"records": []}', encoding="utf-8")

    candidates = tmp_path / "candidates.json"
    candidates.write_text('{"candidates": []}', encoding="utf-8")

    pool = tmp_path / "memories.jsonl"
    pool.write_text("", encoding="utf-8")

    critic = FourOutcomeTransferCritic(feature_block="full")
    critic.epsilon_star = 0.2
    ckpt = tmp_path / "full.joblib"
    critic.save(ckpt)

    return {
        "dataset": dataset,
        "splits": splits,
        "candidates": candidates,
        "pool": pool,
        "ckpt": ckpt,
    }


def test_default_budget_uses_checkpoint_epsilon_star(tmp_path):
    files = _setup(tmp_path)
    result = run_end_to_end_evaluation(
        marble_root=tmp_path / "marble",
        dataset_manifest_path=files["dataset"],
        split_manifest_path=files["splits"],
        split="test",
        candidate_manifest_path=files["candidates"],
        memory_pool_path=files["pool"],
        checkpoint_full=files["ckpt"],
        methods=["smtr"],
        generation_seeds=[0, 1, 2],
        experiment_mode="pilot",
        output=tmp_path / "out",
    )

    assert result["risk_budget_source"] == "checkpoint_validation_selection"
    provenance = result["method_risk_budget_provenance"]
    assert len(provenance) == 1
    entry = provenance[0]
    assert entry["method"] == "smtr"
    assert entry["checkpoint_role"] == "full"
    assert entry["checkpoint_digest"] == file_digest(files["ckpt"])
    assert entry["risk_budget"] == 0.2
    assert entry["risk_budget_source"] == "checkpoint_validation_selection"
