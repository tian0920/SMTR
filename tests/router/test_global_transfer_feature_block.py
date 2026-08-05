"""清单 Test 2: GlobalTransferCritic feature block.

The baseline router must accept a ``global_transfer`` checkpoint and reject
every other feature block with a ValueError.
"""

from __future__ import annotations

import pytest

from smtr.router.baselines import GlobalTransferCriticRouter
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def test_global_transfer_block_accepted():
    critic = FourOutcomeTransferCritic(feature_block="global_transfer")
    router = GlobalTransferCriticRouter(critic=critic)
    assert router is not None


def test_full_block_rejected():
    critic = FourOutcomeTransferCritic(feature_block="full")
    with pytest.raises(ValueError):
        GlobalTransferCriticRouter(critic=critic)
