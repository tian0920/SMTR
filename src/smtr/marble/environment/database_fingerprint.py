"""Stable logical fingerprints for MARBLE database initial state."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.environment.isolation import InitialStateBundle


@dataclass(frozen=True)
class DatabaseLogicalFingerprint:
    schema_digest: str
    content_digest: str
    config_digest: str
    combined_digest: str

    def to_json(self) -> dict[str, str]:
        return asdict(self)


def fingerprint_initial_bundle(
    *, initial_state_bundle: InitialStateBundle, branch_workspace: Path | None = None
) -> DatabaseLogicalFingerprint:
    sql = initial_state_bundle.environment_configuration.get("init_sql") or ""
    schema = _schema_summary(sql)
    content = _content_summary(sql)
    config: dict[str, Any] = {
        "database_name": "sysbench",
        "environment_configuration": initial_state_bundle.environment_configuration,
        "tool_configuration": initial_state_bundle.tool_configuration,
        "workspace_files": sorted(initial_state_bundle.workspace_template),
    }
    if branch_workspace is not None:
        init_sql_path = branch_workspace / "init.sql"
        if init_sql_path.exists():
            config["initialization_sql_digest"] = canonical_digest(
                init_sql_path.read_text(encoding="utf-8")
            )
    schema_digest = canonical_digest(schema)
    content_digest = canonical_digest(content)
    config_digest = canonical_digest(config)
    return DatabaseLogicalFingerprint(
        schema_digest=schema_digest,
        content_digest=content_digest,
        config_digest=config_digest,
        combined_digest=canonical_digest(
            {
                "schema_digest": schema_digest,
                "content_digest": content_digest,
                "config_digest": config_digest,
            }
        ),
    )


def _schema_summary(sql: str) -> list[str]:
    statements = _split_sql(sql)
    return sorted(
        _normalize(statement)
        for statement in statements
        if _normalize(statement).upper().startswith(("CREATE TABLE", "CREATE INDEX", "ALTER TABLE"))
    )


def _content_summary(sql: str) -> list[str]:
    statements = _split_sql(sql)
    return sorted(
        _normalize(statement)
        for statement in statements
        if _normalize(statement).upper().startswith(("INSERT", "UPDATE", "DELETE", "COPY"))
    )


def _split_sql(sql: str) -> list[str]:
    return [part.strip() for part in re.split(r";\s*", sql) if part.strip()]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


@dataclass(frozen=True)
class LiveDatabaseFingerprint:
    """Fingerprint computed from a live PostgreSQL connection."""

    schema_digest: str
    table_row_counts: dict[str, int]
    table_count: int
    pg_version: str
    combined_digest: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def fingerprint_live_database(
    *,
    host: str = "localhost",
    port: int = 5432,
    dbname: str = "sysbench",
    user: str = "test",
    password: str = "Test123_456",
) -> LiveDatabaseFingerprint | None:
    """Connect to PostgreSQL and compute a live fingerprint."""
    try:
        import psycopg2
    except ImportError:
        return None
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password,
            connect_timeout=5,
        )
        cur = conn.cursor()
        # Get PostgreSQL version
        cur.execute("SELECT version()")
        pg_version = cur.fetchone()[0]
        # Get table names and row counts
        cur.execute("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)
        tables = [row[0] for row in cur.fetchall()]
        row_counts: dict[str, int] = {}
        for table in tables:
            try:
                cur.execute(f'SELECT count(*) FROM "{table}"')
                row_counts[table] = cur.fetchone()[0]
            except Exception:
                row_counts[table] = -1
        # Get schema digest from pg_dump-like query
        cur.execute("""
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)
        schema_rows = cur.fetchall()
        schema_digest = canonical_digest([list(r) for r in schema_rows])
        cur.close()
        conn.close()
        combined = canonical_digest({
            "schema_digest": schema_digest,
            "table_row_counts": row_counts,
            "table_count": len(tables),
            "pg_version": pg_version,
        })
        return LiveDatabaseFingerprint(
            schema_digest=schema_digest,
            table_row_counts=row_counts,
            table_count=len(tables),
            pg_version=pg_version,
            combined_digest=combined,
        )
    except Exception:
        return None
