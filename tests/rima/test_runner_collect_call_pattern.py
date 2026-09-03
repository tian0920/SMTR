"""Regression test: runner must NOT pass receiver_agent_ids alongside
receiver_memory_payloads to TrajectoryCollector.collect().

Bug: ``_run_task`` passed both legacy ``receiver_agent_ids`` and the new
per-receiver ``receiver_memory_payloads``, which are mutually exclusive
per the collector's API contract.

Fix: only ``receiver_memory_payloads`` is used; the collector derives
receiver IDs from its keys internally.
"""

from __future__ import annotations

import ast
import inspect
import textwrap


def test_runner_does_not_pass_receiver_agent_ids_with_receiver_memory_payloads():
    """Static check: the collector.collect() call in _run_task must NOT
    include a ``receiver_agent_ids`` keyword argument."""
    from experiments.rima.run_continual_transfer import TransferContinualProtocol

    source = inspect.getsource(TransferContinualProtocol._run_task)
    tree = ast.parse(textwrap.dedent(source))

    # Find all calls to .collect(...)
    collect_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "collect":
                collect_calls.append(node)

    assert collect_calls, "Expected at least one collector.collect() call"

    for call in collect_calls:
        kw_names = [kw.arg for kw in call.keywords]
        # If receiver_memory_payloads is present, receiver_agent_ids must NOT be
        if "receiver_memory_payloads" in kw_names:
            assert "receiver_agent_ids" not in kw_names, (
                "collector.collect() must not receive both "
                "receiver_agent_ids and receiver_memory_payloads "
                "(they are mutually exclusive)"
            )
