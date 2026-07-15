from smtr.router.factual_success_critic import choose_threshold_for_exposure


def test_threshold_matches_validation_mean_exposure_across_invocations() -> None:
    probabilities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    threshold = choose_threshold_for_exposure(
        probabilities=probabilities,
        target_mean_exposure=1.5,
        invocation_count=2,
    )
    assert sum(probability >= threshold for probability in probabilities) == 3


def test_zero_exposure_threshold_withholds_everything() -> None:
    probabilities = [0.9, 0.5]
    threshold = choose_threshold_for_exposure(
        probabilities=probabilities,
        target_mean_exposure=0.0,
        invocation_count=1,
    )
    assert all(probability < threshold for probability in probabilities)
