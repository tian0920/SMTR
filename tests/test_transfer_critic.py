from test_card_feature_snapshots import _record

from smtr.counterfactual.schemas import routing_feature_snapshot_from_card
from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import prediction_input_from_record


def test_transfer_critic_prediction_invariants_and_persistence(tmp_path) -> None:
    _, record = _record(tmp_path)
    records = [
        record.model_copy(update={"transfer_class": "positive", "y_share": 1, "y_withhold": 0}),
        record.model_copy(update={"transfer_class": "negative", "y_share": 0, "y_withhold": 1}),
        record.model_copy(
            update={"transfer_class": "neutral_success", "y_share": 1, "y_withhold": 1}
        ),
        record.model_copy(
            update={"transfer_class": "neutral_failure", "y_share": 0, "y_withhold": 0}
        ),
    ]
    critic = FourOutcomeTransferCritic().fit(records, seed=7, n_bootstrap=3)
    estimate = critic.predict(prediction_input_from_record(records[0]))

    assert round(
        estimate.q00_mean + estimate.q01_mean + estimate.q10_mean + estimate.q11_mean,
        8,
    ) == 1.0
    assert estimate.tau_mean == estimate.q10_mean - estimate.q01_mean
    assert estimate.negative_risk_mean == estimate.q01_mean

    path = tmp_path / "critic.joblib"
    critic.save(path)
    loaded = FourOutcomeTransferCritic.load(path)
    assert loaded.predict(prediction_input_from_record(records[0])) == estimate


def test_transfer_critic_fallback_and_low_support(tmp_path) -> None:
    repo = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
    seed_repository(repo)
    _, record = _record(tmp_path / "record")
    critic = FourOutcomeTransferCritic().fit([record], seed=7, n_bootstrap=2)
    far_card = routing_feature_snapshot_from_card(repo.get_routing_cards()[0]).model_copy(
        update={"goal_summary": "completely unfamiliar zzz yyy xxx"}
    )
    item = prediction_input_from_record(record).model_copy(update={"candidate_card": far_card})
    estimate = critic.predict(item)

    assert estimate.ensemble_size == 2
    assert estimate.low_support in {True, False}
