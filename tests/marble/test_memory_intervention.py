from smtr.marble.memory_injection import MarbleMemoryInjector


def test_memory_injection_share_and_withhold_audits_match_non_memory_sections() -> None:
    base = {
        "system": {"engine": "marble"},
        "task": {"task_id": "1"},
        "tools": {"environment_type": "DB"},
    }
    injector = MarbleMemoryInjector()

    share, share_audit = injector.build_agent_input(
        base_agent_input=base,
        memory_payloads=("use pg_stat_statements",),
        memory_ids=("m1",),
    )
    withhold, withhold_audit = injector.build_agent_input(
        base_agent_input=base,
        memory_payloads=(),
        memory_ids=(),
    )

    assert "memory" in share
    assert "memory" not in withhold
    assert share_audit.contains_memory_section is True
    assert withhold_audit.contains_memory_section is False
    assert share_audit.memory_section_digest is not None
    assert withhold_audit.memory_section_digest is None
    assert share_audit.memory_ids == ("m1",)
    assert withhold_audit.memory_ids == ()
    assert share_audit.system_section_digest == withhold_audit.system_section_digest
    assert share_audit.task_section_digest == withhold_audit.task_section_digest
    assert share_audit.tool_section_digest == withhold_audit.tool_section_digest
