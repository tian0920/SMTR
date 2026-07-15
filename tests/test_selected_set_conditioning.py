"""Tests for selected-set conditioning policies."""

from smtr.router.conditioning import (
    DynamicSelectedSetConditioning,
    FrozenInitialSelectedSetConditioning,
)


def test_dynamic_conditioning_returns_current_selected_set():
    policy = DynamicSelectedSetConditioning()
    assert policy.critic_selected_set(
        initial_selected_set=("a",),
        current_selected_set=("a", "b"),
    ) == ("a", "b")


def test_static_conditioning_returns_initial_selected_set():
    policy = FrozenInitialSelectedSetConditioning()
    assert policy.critic_selected_set(
        initial_selected_set=("a",),
        current_selected_set=("a", "b"),
    ) == ("a",)
