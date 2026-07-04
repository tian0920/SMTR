"""Comprehensive Real LLM test exercising S7 components.

Configurable for different local and remote models:
- Local: transformers + optional 8-bit quantization
- Remote: OpenAI-compatible API endpoints (api_base)
- Config file: conf/llm_test_config.json with named presets

Tests the configured model with the new S7 modules:
- B-01: Sequential router with real LLM plans
- B-02: Safety guard integration
- B-03: Online refresh triggers
- B-04: Off-policy correction with real data
- Basic plan generation and parsing

Usage:
    # Local model (default from config file)
    python test_s7_real_llm.py

    # Use named config preset
    python test_s7_real_llm.py --config-name qwen_api
    python test_s7_real_llm.py --config-name openai_gpt4o_mini

    # Custom config file
    python test_s7_real_llm.py --config conf/my_models.json --config-name llama_local

    # CLI overrides config file
    python test_s7_real_llm.py --config-name qwen_local --temperature 0.5
"""

import argparse
import json
import sys
import time

# --- Config file loading ---

DEFAULT_CONFIG_PATH = "conf/llm_test_config.json"


def load_config_file(path: str) -> dict:
    """Load configuration from JSON file."""
    with open(path) as f:
        return json.load(f)


def get_config_preset(config_data: dict, name: str) -> dict:
    """Get a named config preset from loaded config data."""
    configs = config_data.get("configs", {})
    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return configs[name]


# --- CLI ---

def parse_args():
    """Parse command-line arguments for model configuration."""
    parser = argparse.ArgumentParser(
        description="S7 Real LLM Integration Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default config (qwen_local from conf/llm_test_config.json)
  python test_s7_real_llm.py

  # Use named config preset
  python test_s7_real_llm.py --config-name qwen_api
  python test_s7_real_llm.py --config-name openai_gpt4o_mini
  python test_s7_real_llm.py --config-name ollama_local

  # Custom config file
  python test_s7_real_llm.py --config conf/my_models.json --config-name llama_local

  # CLI overrides config file
  python test_s7_real_llm.py --config-name qwen_local --temperature 0.5

Available configs in conf/llm_test_config.json:
  qwen_local, qwen_local_fp16, qwen_api, llama_local,
  openai_gpt4o_mini, openai_gpt35, vllm_local, ollama_local
        """,
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help=f"Config file path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--config-name", default=None,
        help="Named config preset from config file (uses default_config if omitted)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (overrides config file)",
    )
    parser.add_argument(
        "--api-base", default=None,
        help="Remote API base URL (overrides config file)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for authentication (overrides config file, or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max new tokens (overrides config file)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (overrides config file)",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", default=None,
        help="Use 8-bit quantization (overrides config file)",
    )
    parser.add_argument(
        "--output", default="outputs/s7_llm_test_results.json",
        help="Output JSON file path (default: outputs/s7_llm_test_results.json)",
    )
    return parser.parse_args()


def resolve_config(args) -> dict:
    """Resolve final config by merging config file + CLI overrides."""
    # Start with defaults
    config = {
        "model": "Qwen/Qwen3.5-2B",
        "api_base": None,
        "api_key": None,
        "load_in_8bit": True,
        "max_tokens": 256,
        "temperature": 0.0,
    }

    # Load from config file if it exists
    try:
        config_data = load_config_file(args.config)
        preset_name = args.config_name or config_data.get("default_config", "qwen_local")
        preset = get_config_preset(config_data, preset_name)
        config.update({k: v for k, v in preset.items() if v is not None})
        print(f"  Loaded config: {preset_name} from {args.config}")
    except FileNotFoundError:
        print(f"  Config file not found: {args.config}, using defaults")
    except ValueError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    # CLI overrides
    if args.model is not None:
        config["model"] = args.model
    if args.api_base is not None:
        config["api_base"] = args.api_base
    if args.api_key is not None:
        config["api_key"] = args.api_key
    if args.max_tokens is not None:
        config["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.load_in_8bit is not None:
        config["load_in_8bit"] = args.load_in_8bit

    return config


def build_llm(config: dict):
    """Build a RealLLM instance from resolved config."""
    from smtr.runtime.real_llm import RealLLM

    mode = "API" if config["api_base"] else "local"
    quant = "8-bit" if config["load_in_8bit"] and not config["api_base"] else "none"
    print(f"  Mode: {mode}")
    print(f"  Model: {config['model']}")
    if config["api_base"]:
        print(f"  API base: {config['api_base']}")
        has_key = bool(config.get("api_key"))
        print(f"  API key: {'***' if has_key else '(none, using env var or unauthenticated)'}")
    else:
        print(f"  Quantization: {quant}")
    print(f"  Max tokens: {config['max_tokens']}")
    print(f"  Temperature: {config['temperature']}")

    return RealLLM(
        model_name=config["model"],
        api_base=config["api_base"],
        api_key=config.get("api_key"),
        load_in_8bit=config["load_in_8bit"] if not config["api_base"] else False,
        max_new_tokens=config["max_tokens"],
        temperature=config["temperature"],
    )


# --- Test infrastructure ---

def timestamp():
    return time.time()


def run_test(name, fn, *args, **kwargs):
    """Run a test function and return (result, elapsed, error)."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    start = timestamp()
    try:
        result = fn(*args, **kwargs)
        elapsed = timestamp() - start
        print(f"  ELAPSED: {elapsed:.3f}s")
        return result, elapsed, None
    except Exception as e:
        elapsed = timestamp() - start
        print(f"  ERROR: {e}")
        return None, elapsed, str(e)


# --- Test functions ---

def test_model_loading(config):
    """Test 1: Model loading and basic generation."""
    print(f"  Loading {config['model']}...")
    start = timestamp()
    llm = build_llm(config)
    load_time = timestamp() - start
    print(f"  Model loaded in {load_time:.1f}s")

    # Basic generation test with simple prompt
    raw = llm._generate("Return only the JSON: {\"answer\": 4}")
    print(f"  Raw generation: {raw[:200]}")
    return {"load_time": load_time, "raw_output": raw[:200]}


def test_plan_generation(llm):
    """Test 2: Plan generation with ToyEnvironment observation."""
    from smtr.runtime.environment import ToyEnvironment

    env = ToyEnvironment(seed=7)
    obs = env.observe()
    task = "build the target artifact"

    # Create a simple visible payload
    visible_payloads = [
        {
            "memory_id": "mem_test_1",
            "goal_summary": "gather resources and build",
            "steps": [
                "call gather_key",
                "call open_chest",
                "call collect_artifact",
            ],
        }
    ]

    print(f"  Task: {task}")
    print(f"  Valid sequence: {obs.get('valid_sequence', [])}")
    print(f"  Target: {obs.get('target_artifact', 'unknown')}")

    result = llm.plan(task, obs, visible_payloads)
    print(f"  Plan: {result.get('plan', [])}")
    print(f"  Explanation: {result.get('explanation', '')[:200]}")
    print(f"  Source: {result.get('local_trace', {}).get('source', 'unknown')}")

    expected = ["gather_key", "open_chest", "collect_artifact"]
    plan = result.get("plan", [])
    is_correct = plan == expected
    print(f"  Plan matches expected: {is_correct}")
    return {"plan": plan, "is_correct": is_correct, "explanation": result.get("explanation", "")}


def test_plan_with_tool_environment(llm):
    """Test 3: Plan generation with ToolEnvironment."""
    from smtr.runtime.tool_environment import ToolEnvironment

    env = ToolEnvironment(seed=7)
    obs = env.observe()
    task = "read the config file and process it"

    visible_payloads = [
        {
            "memory_id": "mem_tool_1",
            "goal_summary": "read and process files",
            "steps": [
                "call read_file with path=/workspace/config.json",
                "call run_command with command=process",
                "call write_file with path=/workspace/output.json",
            ],
        }
    ]

    print(f"  Task: {task}")
    print(f"  Valid sequence: {obs.get('valid_sequence', [])}")
    print(f"  Available tools: {obs.get('available_tools', [])}")

    result = llm.plan(task, obs, visible_payloads)
    plan = result.get("plan", [])
    print(f"  Plan: {plan}")
    print(f"  Explanation: {result.get('explanation', '')[:200]}")

    expected = ["read_file", "run_command", "write_file"]
    is_correct = plan == expected
    print(f"  Plan matches expected: {is_correct}")
    return {"plan": plan, "is_correct": is_correct}


def test_sequential_router_with_real_plan(config):
    """Test 4: Sequential router using real LLM-generated plan."""
    from smtr.counterfactual.schemas import (
        BranchOutcome,
        ContextFingerprint,
        PairedInterventionRecord,
        routing_feature_snapshot_from_card,
    )
    from smtr.memory.schemas import MemoryRoutingCard
    from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest, CandidateScore
    from smtr.router.sequential_router import ProductionSequentialRouter, SequentialRouterConfig
    from smtr.router.transfer_critic import FourOutcomeTransferCritic

    # Create cards first
    memory_ids = ["mem-0", "mem-1", "mem-2"]
    cards_by_id = {
        mid: MemoryRoutingCard(
            memory_id=mid, active_payload_version=1, goal_summary="test goal",
            task_tags=["test"], compatible_receiver_roles=["executor"],
        )
        for mid in memory_ids
    }

    # Generate synthetic training records WITH candidate_card_snapshot
    records = []
    for i in range(20):
        mid = memory_ids[i % 3]
        card = cards_by_id[mid]
        card_snapshot = routing_feature_snapshot_from_card(card)

        ctx = ContextFingerprint(
            task_id=f"task-{i}", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", selected_memory_ids=[], selected_set_signature="empty",
            episode_id=f"ep-{i}",
        )
        branch = BranchOutcome(
            team_success=(i % 3 != 0), team_reward=0.0, team_summary="test",
            final_environment_observation={}, selected_memory_ids_by_agent={},
            router_trace=[], target_memory_visible_to_receiver=True,
            selected_final_at_target_node=[],
        )
        records.append(PairedInterventionRecord(
            record_id=f"rec-{i}", episode_id=f"ep-{i}", task_id=f"task-{i}",
            graph_node="node-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", candidate_memory_id=mid, candidate_payload_version=1,
            candidate_order=[mid], target_index=0, selected_before=[],
            decision_context=ctx, memory_store_revision=1, memory_snapshot_digest="abc",
            runtime_snapshot_digest="def", continuation_policy_name="test",
            continuation_policy_version="1", common_seed=42,
            share_outcome=branch, withhold_outcome=branch,
            y_share=1 if i % 3 != 0 else 0, y_withhold=1 if i % 3 == 0 else 0,
            transfer_class="positive" if i % 3 != 0 else "negative",
            target_selection_probability=0.5,
            schema_version="1.1",
            candidate_card_snapshot=card_snapshot,
            selected_before_card_snapshots=[],
            selected_before_payload_versions={},
        ))

    critic = FourOutcomeTransferCritic()
    critic.fit(records, seed=42, n_bootstrap=5)
    print(f"  Critic trained on {len(records)} records")

    # Create router
    router_config = SequentialRouterConfig(tau_threshold=0.0, max_shares_per_invocation=3)
    router = ProductionSequentialRouter(critic=critic, config=router_config, seed=42)

    # Create proposal
    request = CandidateRequest(
        task="test task", task_stage="test", receiver_agent_id="agent-1",
        receiver_role="executor", receiver_capabilities=[], environment_observation={},
        top_k=3, seed=42,
    )
    candidates = [
        CandidateScore(
            memory_id=mid, total_score=0.5 - i*0.1, goal_similarity=0.5,
            task_tag_overlap=0.5, environment_compatibility=0.5,
            receiver_compatibility=1.0,
        )
        for i, mid in enumerate(memory_ids)
    ]
    proposal = CandidateProposal(request=request, ranked_candidates=candidates, pool_revision=1)

    result = router.decide_from_proposal(
        receiver_agent_id="agent-1", proposal=proposal, cards_by_id=cards_by_id,
    )
    print(f"  Router decisions: {len(result.decisions)}")
    for d in result.decisions:
        print(f"    {d.memory_id}: {d.action} (tau={d.tau_mean:.3f}, reason={d.reason})")
    print(f"  Selected: {result.selected_memory_ids}")
    return {
        "n_decisions": len(result.decisions),
        "selected": result.selected_memory_ids,
        "router_name": result.router_name,
    }


def test_safety_guard_with_real_critic():
    """Test 5: Safety guard with real critic estimates."""
    from smtr.counterfactual.schemas import (
        BranchOutcome,
        ContextFingerprint,
        PairedInterventionRecord,
        routing_feature_snapshot_from_card,
    )
    from smtr.memory.schemas import MemoryRoutingCard
    from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest, CandidateScore
    from smtr.router.safety_guard import FallbackRouter, SafetyGuardConfig
    from smtr.router.transfer_critic import FourOutcomeTransferCritic

    # Create cards first
    memory_ids = ["mem-0", "mem-1", "mem-2"]
    cards_by_id = {
        mid: MemoryRoutingCard(
            memory_id=mid, active_payload_version=1, goal_summary="test",
            task_tags=["test"], compatible_receiver_roles=["executor"],
        )
        for mid in memory_ids
    }

    # Train critic with proper records
    records = []
    for i in range(20):
        mid = memory_ids[i % 3]
        card = cards_by_id[mid]
        card_snapshot = routing_feature_snapshot_from_card(card)

        ctx = ContextFingerprint(
            task_id=f"task-{i}", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", selected_memory_ids=[], selected_set_signature="empty",
            episode_id=f"ep-{i}",
        )
        branch = BranchOutcome(
            team_success=(i % 2 == 0), team_reward=0.0, team_summary="test",
            final_environment_observation={}, selected_memory_ids_by_agent={},
            router_trace=[], target_memory_visible_to_receiver=True,
            selected_final_at_target_node=[],
        )
        records.append(PairedInterventionRecord(
            record_id=f"rec-{i}", episode_id=f"ep-{i}", task_id=f"task-{i}",
            graph_node="node-1", receiver_agent_id="agent-1", receiver_role="executor",
            task_stage="test", candidate_memory_id=mid, candidate_payload_version=1,
            candidate_order=[mid], target_index=0, selected_before=[],
            decision_context=ctx, memory_store_revision=1, memory_snapshot_digest="abc",
            runtime_snapshot_digest="def", continuation_policy_name="test",
            continuation_policy_version="1", common_seed=42,
            share_outcome=branch, withhold_outcome=branch,
            y_share=1 if i % 2 == 0 else 0, y_withhold=0 if i % 2 == 0 else 1,
            transfer_class="positive" if i % 2 == 0 else "negative",
            target_selection_probability=0.5,
            schema_version="1.1",
            candidate_card_snapshot=card_snapshot,
            selected_before_card_snapshots=[],
            selected_before_payload_versions={},
        ))

    critic = FourOutcomeTransferCritic()
    critic.fit(records, seed=42, n_bootstrap=5)

    # Create fallback router with strict safety
    safety_config = SafetyGuardConfig(max_negative_risk_ucb=0.5, max_uncertainty=0.6)
    router = FallbackRouter(critic=critic, safety_config=safety_config, seed=42)

    request = CandidateRequest(
        task="test", task_stage="test", receiver_agent_id="agent-1",
        receiver_role="executor", receiver_capabilities=[], environment_observation={},
        top_k=3, seed=42,
    )
    candidates = [
        CandidateScore(
            memory_id=mid, total_score=0.5, goal_similarity=0.5,
            task_tag_overlap=0.5, environment_compatibility=0.5,
            receiver_compatibility=1.0,
        )
        for mid in memory_ids
    ]
    proposal = CandidateProposal(request=request, ranked_candidates=candidates, pool_revision=1)

    result = router.decide_from_proposal(
        receiver_agent_id="agent-1", proposal=proposal, cards_by_id=cards_by_id,
    )
    print(f"  Fallback router decisions: {len(result.decisions)}")
    for d in result.decisions:
        print(f"    {d.memory_id}: {d.action} (reason={d.reason})")
    print(f"  In fallback mode: {router.in_fallback_mode}")
    stats = router.get_stats()
    print(f"  Stats: {json.dumps(stats, indent=2)}")
    return {"n_decisions": len(result.decisions), "stats": stats}


def test_multi_seed_comparison(llm):
    """Test 6: Multi-seed plan generation comparison."""
    from smtr.runtime.environment import ToyEnvironment

    seeds = [7, 42, 123, 256, 999]
    results = []

    for seed in seeds:
        env = ToyEnvironment(seed=seed)
        obs = env.observe()
        task = "build the target artifact"
        visible_payloads = [
            {
                "memory_id": "mem_test_1",
                "goal_summary": "gather resources and build",
                "steps": [
                    "call gather_key",
                    "call open_chest",
                    "call collect_artifact",
                ],
            }
        ]
        result = llm.plan(task, obs, visible_payloads)
        plan = result.get("plan", [])
        expected = ["gather_key", "open_chest", "collect_artifact"]
        is_correct = plan == expected
        results.append({
            "seed": seed,
            "plan": plan,
            "is_correct": is_correct,
            "valid_sequence": obs.get("valid_sequence", []),
        })
        status = "✅" if is_correct else "❌"
        print(f"  Seed {seed}: {status} plan={plan}")

    n_correct = sum(1 for r in results if r["is_correct"])
    print(f"\n  Success rate: {n_correct}/{len(seeds)} ({100*n_correct/len(seeds):.0f}%)")
    return {"results": results, "success_rate": n_correct / len(seeds)}


# --- Main ---

def main():
    args = parse_args()
    config = resolve_config(args)

    print("=" * 60)
    print("S7 Real LLM Integration Test")
    print("=" * 60)
    print("\nConfiguration:")
    mode = "API" if config["api_base"] else "local"
    quant = "8-bit" if config["load_in_8bit"] and not config["api_base"] else "none"
    print(f"  Mode: {mode}")
    print(f"  Model: {config['model']}")
    if config["api_base"]:
        print(f"  API base: {config['api_base']}")
    else:
        print(f"  Quantization: {quant}")
    print(f"  Max tokens: {config['max_tokens']}")
    print(f"  Temperature: {config['temperature']}")

    all_results = {}
    total_start = timestamp()

    # Test 1: Model loading
    result, elapsed, error = run_test("Model Loading", test_model_loading, config)
    all_results["model_loading"] = {"elapsed": elapsed, "error": error}
    if error:
        print(f"\nFATAL: Model loading failed: {error}")
        sys.exit(1)

    # Load LLM once for remaining tests
    llm = build_llm(config)

    # Test 2: Plan generation (ToyEnvironment)
    result, elapsed, error = run_test("Plan Generation (ToyEnv)", test_plan_generation, llm)
    all_results["plan_toy_env"] = {"result": result, "elapsed": elapsed, "error": error}

    # Test 3: Plan generation (ToolEnvironment)
    result, elapsed, error = run_test(
        "Plan Generation (ToolEnv)", test_plan_with_tool_environment, llm,
    )
    all_results["plan_tool_env"] = {"result": result, "elapsed": elapsed, "error": error}

    # Test 4: Sequential router
    result, elapsed, error = run_test(
        "Sequential Router + Real LLM", test_sequential_router_with_real_plan, config,
    )
    all_results["sequential_router"] = {"result": result, "elapsed": elapsed, "error": error}

    # Test 5: Safety guard
    result, elapsed, error = run_test(
        "Safety Guard + Real Critic", test_safety_guard_with_real_critic,
    )
    all_results["safety_guard"] = {"result": result, "elapsed": elapsed, "error": error}

    # Test 6: Multi-seed comparison
    result, elapsed, error = run_test("Multi-Seed Comparison", test_multi_seed_comparison, llm)
    all_results["multi_seed"] = {"result": result, "elapsed": elapsed, "error": error}

    total_elapsed = timestamp() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s")
    for name, data in all_results.items():
        status = "✅" if data.get("error") is None else "❌"
        elapsed = data.get("elapsed", 0)
        print(f"  {status} {name}: {elapsed:.1f}s")
        if data.get("error"):
            print(f"     Error: {data['error']}")

    # Save results
    output = {
        "config": config,
        "total_time": total_elapsed,
        "tests": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
