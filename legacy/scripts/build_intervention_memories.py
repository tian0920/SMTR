"""Build 32 intervention memories: 8 tasks x 4 memory types.

Memory types:
  - beneficial:    Correct diagnostic procedure targeting actual root causes
  - irrelevant:    Same domain but different problem, length-matched
  - conflicting:   Highly related but wrong diagnostic order / misinterpretation
  - role_mismatched: Correct content but framed for wrong agent role
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Task definitions (8 selected sweet-spot tasks)
# ---------------------------------------------------------------------------

TASKS = [
    {"task_id": 51, "domain": "Healthcare", "root_causes": ["VACUUM", "FETCH_LARGE_DATA"],
     "tables": "patients, doctors, appointments, medical_records",
     "scenario": "patient records and appointment scheduling"},
    {"task_id": 52, "domain": "Social Media", "root_causes": ["LOCK_CONTENTION", "REDUNDANT_INDEX"],
     "tables": "users, posts, comments, likes, follows",
     "scenario": "post creation, commenting, and user interactions"},
    {"task_id": 58, "domain": "E-commerce", "root_causes": ["FETCH_LARGE_DATA", "INSERT_LARGE_DATA"],
     "tables": "customers, products, orders, order_items, payments",
     "scenario": "order processing and payment tracking"},
    {"task_id": 59, "domain": "Music Streaming", "root_causes": ["INSERT_LARGE_DATA", "LOCK_CONTENTION"],
     "tables": "users, artists, albums, songs, playlists",
     "scenario": "playlist management and listening activity tracking"},
    {"task_id": 67, "domain": "E-commerce", "root_causes": ["REDUNDANT_INDEX", "VACUUM"],
     "tables": "customers, products, orders, order_items, payments",
     "scenario": "product catalog and order management"},
    {"task_id": 73, "domain": "Transportation", "root_causes": ["VACUUM", "FETCH_LARGE_DATA"],
     "tables": "vehicles, drivers, routes, trips, cargo, maintenance",
     "scenario": "trip tracking and vehicle maintenance logs"},
    {"task_id": 85, "domain": "Finance", "root_causes": ["INSERT_LARGE_DATA", "LOCK_CONTENTION"],
     "tables": "users, accounts, transactions, investments",
     "scenario": "transaction processing and investment tracking"},
    {"task_id": 93, "domain": "Social Media", "root_causes": ["REDUNDANT_INDEX", "VACUUM"],
     "tables": "users, posts, comments, likes, follows, direct_messages",
     "scenario": "media uploads and direct messaging"},
]


# ---------------------------------------------------------------------------
# Memory payload templates per (rc_type, memory_type)
# ---------------------------------------------------------------------------

def _beneficial(task: dict) -> str:
    """Correct diagnostic procedure for the actual root causes."""
    rcs = task["root_causes"]
    domain = task["domain"]
    tables = task["tables"]

    guidance = {
        ("VACUUM", "FETCH_LARGE_DATA"): (
            f"When diagnosing {domain.lower()} database slowdowns affecting {tables}, "
            f"first check pg_stat_all_tables for tables with high dead tuple counts "
            f"and autovacuum lag on {task['scenario'].split(' and ')[0]} tables. "
            f"Then examine pg_stat_statements for queries fetching large result sets "
            f"without LIMIT clauses, especially joins across {task['scenario'].split(' and ')[-1]}. "
            f"Vacuum issues cause table bloat which amplifies sequential scan costs "
            f"for large data fetches."
        ),
        ("LOCK_CONTENTION", "REDUNDANT_INDEX"): (
            f"When diagnosing {domain.lower()} database issues on {tables}, "
            f"first check pg_locks for lock wait chains and row-level contention "
            f"during concurrent {task['scenario']} operations. "
            f"Then examine pg_stat_user_indexes for duplicate or overlapping index "
            f"definitions that slow every write. Lock contention combined with "
            f"redundant indexes creates compounding write bottlenecks."
        ),
        ("FETCH_LARGE_DATA", "INSERT_LARGE_DATA"): (
            f"When investigating {domain.lower()} database performance on {tables}, "
            f"first check pg_stat_statements for large SELECT queries returning "
            f"bulk result sets from {task['scenario'].split(' and ')[0]} tables. "
            f"Then examine pg_stat_activity for concurrent bulk INSERT operations "
            f"on {task['scenario'].split(' and ')[-1]} tables. "
            f"Heavy reads competing with large writes create I/O saturation "
            f"and buffer pool pressure."
        ),
        ("INSERT_LARGE_DATA", "LOCK_CONTENTION"): (
            f"When investigating {domain.lower()} database issues on {tables}, "
            f"first check pg_stat_statements for bulk INSERT operations during "
            f"{task['scenario'].split(' and ')[0]}. "
            f"Then examine pg_locks for row-level lock conflicts on "
            f"{task['scenario'].split(' and ')[-1]} tables when concurrent "
            f"batch inserts and updates compete for the same rows. "
            f"Large batch inserts combined with lock conflicts create "
            f"cascading write bottlenecks."
        ),
        ("REDUNDANT_INDEX", "VACUUM"): (
            f"When diagnosing {domain.lower()} database slowdowns on {tables}, "
            f"first check pg_stat_user_indexes for redundant or unused indexes "
            f"that slow down writes on {task['scenario'].split(' and ')[0]} tables. "
            f"Then examine pg_stat_all_tables for dead tuple accumulation and "
            f"autovacuum delays on {task['scenario'].split(' and ')[-1]} tables. "
            f"Redundant indexes amplify vacuum issues because more index entries "
            f"must be maintained during cleanup."
        ),
    }
    rc_key = tuple(sorted(rcs))
    # Map sorted tuple to the right guidance
    for key, text in guidance.items():
        if set(key) == set(rcs):
            return text
    return guidance.get(tuple(rcs), guidance[tuple(rc_key)])


def _irrelevant(task: dict) -> str:
    """Same domain but different problem, length-matched."""
    domain = task["domain"]
    tables = task["tables"]

    templates = {
        "Healthcare": (
            f"When managing {domain.lower()} database capacity for {tables}, "
            f"monitor disk usage growth rates for patient records and medical history "
            f"tables. Archive completed appointments older than 2 years to cold storage. "
            f"Implement partitioning by date ranges for appointment scheduling tables "
            f"to improve query performance. Regular capacity planning prevents "
            f"unexpected storage exhaustion and maintains system responsiveness."
        ),
        "Social Media": (
            f"When optimizing {domain.lower()} database connection pooling for {tables}, "
            f"configure pool sizes based on concurrent user sessions and API request rates. "
            f"Monitor connection wait times and idle connection counts. "
            f"Implement connection timeout policies for long-running social feed queries. "
            f"Proper pool sizing prevents connection exhaustion during traffic spikes "
            f"and reduces latency for user interactions."
        ),
        "E-commerce": (
            f"When managing {domain.lower()} database replication lag for {tables}, "
            f"monitor replica lag metrics during peak order processing windows. "
            f"Configure synchronous commit settings for payment transactions "
            f"to ensure data consistency. Tune wal_sender_timeout and "
            f"max_wal_senders for optimal replication throughput. "
            f"Replication lag can cause stale product inventory displays."
        ),
        "Music Streaming": (
            f"When optimizing {domain.lower()} database cache hit ratios for {tables}, "
            f"configure shared_buffers and effective_cache_size based on song metadata "
            f"access patterns. Monitor buffer cache hit rates for frequently queried "
            f"playlist and artist lookup tables. Implement query result caching "
            f"for popular song recommendations to reduce database load "
            f"during peak listening hours."
        ),
        "Transportation": (
            f"When managing {domain.lower()} database backup strategies for {tables}, "
            f"implement incremental backups for trip logs and maintenance records. "
            f"Schedule full backups during low-traffic windows between route cycles. "
            f"Monitor backup completion times and verify restore procedures regularly. "
            f"Point-in-time recovery capability is critical for fleet management "
            f"and cargo tracking data integrity."
        ),
        "Finance": (
            f"When optimizing {domain.lower()} database query planning for {tables}, "
            f"analyze execution plans for transaction history queries with date range "
            f"filters. Ensure statistics are up-to-date for investment portfolio "
            f"calculations. Configure random_page_cost and effective_io_concurrency "
            f"for SSD storage. Proper query planning reduces latency for "
            f"real-time balance checks and portfolio valuations."
        ),
    }
    return templates.get(domain, templates["E-commerce"])


def _conflicting(task: dict) -> str:
    """Highly related but wrong diagnostic order / misinterpretation."""
    rcs = task["root_causes"]
    domain = task["domain"]
    tables = task["tables"]

    # Give WRONG advice: swap the diagnostic order, misattribute symptoms
    templates = {
        ("VACUUM", "FETCH_LARGE_DATA"): (
            f"When diagnosing {domain.lower()} database issues on {tables}, "
            f"focus first on query optimization: add covering indexes for all "
            f"SELECT queries on {task['scenario'].split(' and ')[0]} tables. "
            f"The performance problem is almost certainly caused by missing indexes "
            f"rather than maintenance issues. Check pg_stat_statements for slow "
            f"queries and add indexes. Vacuum and dead tuples are rarely the "
            f"primary cause of read performance problems."
        ),
        ("LOCK_CONTENTION", "REDUNDANT_INDEX"): (
            f"When diagnosing {domain.lower()} database issues on {tables}, "
            f"first check pg_stat_user_tables for sequential scan patterns. "
            f"The bottleneck is likely caused by missing indexes on frequently "
            f"queried columns in {task['scenario']}. Add indexes on all WHERE "
            f"clause columns. Lock contention is usually a symptom of long-running "
            f"queries, not a root cause itself. Focus on query optimization first."
        ),
        ("FETCH_LARGE_DATA", "INSERT_LARGE_DATA"): (
            f"When investigating {domain.lower()} database issues on {tables}, "
            f"first check autovacuum settings and increase vacuum frequency. "
            f"The performance degradation is caused by table bloat from dead tuples. "
            f"Run VACUUM FULL on all tables in the {task['scenario']} pipeline. "
            f"Large data operations are normal and the real issue is maintenance "
            f"scheduling. Focus on vacuum configuration rather than query patterns."
        ),
        ("INSERT_LARGE_DATA", "LOCK_CONTENTION"): (
            f"When investigating {domain.lower()} database issues on {tables}, "
            f"first examine pg_stat_user_indexes for unused indexes and drop them. "
            f"Write performance problems are almost always caused by excessive "
            f"indexing on {task['scenario'].split(' and ')[0]} tables. "
            f"Remove all non-primary indexes and rebuild only those needed for "
            f"SELECT queries. Lock conflicts will resolve once index maintenance "
            f"overhead is reduced."
        ),
        ("REDUNDANT_INDEX", "VACUUM"): (
            f"When diagnosing {domain.lower()} database issues on {tables}, "
            f"focus on pg_stat_statements to identify the top-N slowest queries. "
            f"Add composite indexes to cover all multi-column WHERE clauses on "
            f"{task['scenario'].split(' and ')[0]} tables. The root cause is "
            f"insufficient indexing, not index redundancy. More indexes will "
            f"speed up reads. Vacuum issues are a red herring in modern "
            f"PostgreSQL with autovacuum enabled."
        ),
    }
    rc_set = set(rcs)
    for key, text in templates.items():
        if set(key) == rc_set:
            return text
    return templates[("VACUUM", "FETCH_LARGE_DATA")]


def _role_mismatched(task: dict) -> str:
    """Correct content but framed for wrong agent role (frontend developer)."""
    rcs = task["root_causes"]
    domain = task["domain"]
    tables = task["tables"]

    # Same diagnostic knowledge but framed as frontend/UI advice
    templates = {
        ("VACUUM", "FETCH_LARGE_DATA"): (
            f"As a frontend developer working on {domain.lower()} UI for {tables}, "
            f"you should implement pagination and lazy loading for all list views "
            f"showing {task['scenario']}. Use cursor-based pagination with LIMIT/OFFSET "
            f"in your API calls. Cache frequently accessed records in the client-side "
            f"store. The database performance issues you observe are caused by the "
            f"frontend requesting too much data at once without proper pagination."
        ),
        ("LOCK_CONTENTION", "REDUNDANT_INDEX"): (
            f"As a frontend developer building {domain.lower()} interfaces for {tables}, "
            f"implement optimistic locking in your UI forms for {task['scenario']}. "
            f"Add client-side validation to prevent duplicate submissions. Use debouncing "
            f"on search inputs to reduce request frequency. The lock contention you "
            f"observe is caused by the frontend sending too many concurrent write "
            f"requests without proper throttling."
        ),
        ("FETCH_LARGE_DATA", "INSERT_LARGE_DATA"): (
            f"As a frontend developer working on {domain.lower()} dashboards for {tables}, "
            f"implement virtual scrolling for large data tables in {task['scenario']}. "
            f"Use WebSocket connections instead of polling for real-time updates. "
            f"Batch your API requests and use request queuing to avoid overwhelming "
            f"the backend. The insert and fetch performance issues are caused by "
            f"the frontend making too many unbatched requests."
        ),
        ("INSERT_LARGE_DATA", "LOCK_CONTENTION"): (
            f"As a frontend developer building {domain.lower()} data entry forms for {tables}, "
            f"implement client-side queuing for bulk operations in {task['scenario']}. "
            f"Use optimistic UI updates with rollback on conflict. Add retry logic "
            f"with exponential backoff for failed submissions. The lock contention "
            f"you see is caused by the frontend submitting forms without proper "
            f"request coordination."
        ),
        ("REDUNDANT_INDEX", "VACUUM"): (
            f"As a frontend developer working on {domain.lower()} search features for {tables}, "
            f"implement client-side caching with service workers for {task['scenario']}. "
            f"Use IndexedDB to store frequently accessed records offline. Implement "
            f"search result deduplication in the UI layer. The index and vacuum "
            f"issues you observe are caused by the frontend making redundant "
            f"search queries without proper caching."
        ),
    }
    rc_set = set(rcs)
    for key, text in templates.items():
        if set(key) == rc_set:
            return text
    return templates[("VACUUM", "FETCH_LARGE_DATA")]


# ---------------------------------------------------------------------------
# Build memory manifest
# ---------------------------------------------------------------------------

def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def build_memory_manifest(output_path: Path) -> list[dict]:
    """Build 32 memories and write as JSONL manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memories: list[dict] = []

    for task in TASKS:
        tid = task["task_id"]
        domain = task["domain"]
        rcs = task["root_causes"]

        builders = [
            ("beneficial", _beneficial),
            ("irrelevant", _irrelevant),
            ("conflicting", _conflicting),
            ("role_mismatched", _role_mismatched),
        ]
        for mem_type, builder_fn in builders:
            payload = builder_fn(task)
            memory_id = f"interv_{tid}_{mem_type}"
            record = {
                "memory_id": memory_id,
                "task_id": str(tid),
                "memory_type": mem_type,
                "domain": domain,
                "task_root_causes": rcs,
                "payload": payload,
                "payload_digest": _digest(payload),
                "target_receiver_agent_id": "agent1",
                "intended_role": "diagnostic_agent" if mem_type != "role_mismatched" else "frontend_developer",
                "source_or_construction": "human_designed_intervention",
                "contains_final_answer": False,
                "rationale": _rationale(mem_type, task),
            }
            memories.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for m in memories:
            f.write(json.dumps(m, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"Wrote {len(memories)} memories to {output_path}")
    return memories


def _rationale(mem_type: str, task: dict) -> str:
    rcs = " + ".join(task["root_causes"])
    domain = task["domain"]
    rationales = {
        "beneficial": (
            f"Provides correct diagnostic procedure for {rcs} in {domain} database, "
            f"guiding agent through proper investigation order."
        ),
        "irrelevant": (
            f"Same {domain} domain but addresses a different problem (capacity/connection/backup), "
            f"not the actual root causes {rcs}."
        ),
        "conflicting": (
            f"Appears relevant to {domain} database but provides misleading diagnostic order "
            f"and misattributes symptoms away from actual root causes {rcs}."
        ),
        "role_mismatched": (
            f"Contains relevant diagnostic knowledge for {rcs} but framed for "
            f"frontend developer role instead of diagnostic agent."
        ),
    }
    return rationales[mem_type]


# ---------------------------------------------------------------------------
# Build pair manifest
# ---------------------------------------------------------------------------

def build_pair_manifest(
    output_path: Path,
    memory_manifest_path: Path,
    seeds: list[int] = (41, 42, 43),  # type: ignore
) -> None:
    """Build pair manifest: 8 tasks x 3 seeds x 4 memory types = 96 pairs."""
    # Load memories to get memory_ids
    memories: list[dict] = []
    with memory_manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                memories.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs: list[dict] = []
    pair_counter = 0

    for task in TASKS:
        tid = task["task_id"]
        for seed in seeds:
            # Alternate branch order by hash for balance
            order_hash = (tid * 7 + seed * 13) % 2
            branch_order = "share_then_withhold" if order_hash == 0 else "withhold_then_share"
            execution_order = ["share", "withhold"] if branch_order == "share_then_withhold" else ["withhold", "share"]

            for mem_type in ["beneficial", "irrelevant", "conflicting", "role_mismatched"]:
                memory_id = f"interv_{tid}_{mem_type}"
                pair_key = f"interv_{tid}_{mem_type}_s{seed}"
                pair_id = f"interv_pair_{pair_counter:03d}"

                pair = {
                    "pair_key": pair_key,
                    "pair_id": pair_id,
                    "task_id": tid,
                    "memory_id": memory_id,
                    "memory_type": mem_type,
                    "seed": seed,
                    "branch_order": branch_order,
                    "execution_order": execution_order,
                    "receiver_agent_id": "agent1",
                    "order_hash": round((tid * 3 + seed * 7 + pair_counter) % 100 / 100.0, 2),
                    "status": "pending",
                }
                pairs.append(pair)
                pair_counter += 1

    # Write: header + pairs
    header = {
        "_type": "manifest_header",
        "schema_version": "memory_intervention_v1",
        "task_count": len(TASKS),
        "seeds": list(seeds),
        "memory_types": ["beneficial", "irrelevant", "conflicting", "role_mismatched"],
        "pair_count": len(pairs),
        "order_balance": {
            "share_then_withhold": sum(1 for p in pairs if p["branch_order"] == "share_then_withhold"),
            "withhold_then_share": sum(1 for p in pairs if p["branch_order"] == "withhold_then_share"),
        },
        "note": "8 tasks x 3 seeds x 4 memory types = 96 paired runs",
    }

    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, sort_keys=True) + "\n")
        for p in pairs:
            f.write(json.dumps(p, sort_keys=True) + "\n")

    print(f"Wrote {len(pairs)} pairs + header to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build intervention memories and pair manifest")
    parser.add_argument(
        "--output-dir",
        default="artifacts/paper_experiments/memory_intervention",
    )
    parser.add_argument("--seeds", default="41,42,43")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    seeds = [int(x.strip()) for x in args.seeds.split(",")]

    memory_path = output_dir / "memory_manifest.jsonl"
    pair_path = output_dir / "pair_manifest.jsonl"

    build_memory_manifest(memory_path)
    build_pair_manifest(pair_path, memory_path, seeds=seeds)


if __name__ == "__main__":
    main()
