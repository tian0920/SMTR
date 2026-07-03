import sqlite3
from pathlib import Path
from uuid import uuid4

from smtr.memory.schemas import ExecutionEvidence, MemoryRoutingCard, ProcedurePayload, utc_now
from smtr.memory.serialization import model_from_json, model_to_json
from smtr.memory.snapshot import MemoryStoreSnapshot


class SQLiteSharedMemoryRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_payload_versions (
                    memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL,
                    PRIMARY KEY (memory_id, version)
                );
                CREATE TABLE IF NOT EXISTS memory_routing_cards (
                    memory_id TEXT PRIMARY KEY,
                    active_payload_version INTEGER NOT NULL,
                    card_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS execution_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    payload_version INTEGER NOT NULL,
                    evidence_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_store_metadata (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    store_revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paired_transfer_evidence (
                    record_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    payload_version INTEGER NOT NULL,
                    transfer_class TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                INSERT OR IGNORE INTO memory_store_metadata (id, store_revision, updated_at)
                VALUES (1, 0, '');
                """
            )

    def _bump_revision(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            UPDATE memory_store_metadata
            SET store_revision = store_revision + 1, updated_at = ?
            WHERE id = 1
            """,
            (utc_now().isoformat(),),
        )

    def create_memory(self, payload: ProcedurePayload, card: MemoryRoutingCard) -> None:
        if payload.memory_id != card.memory_id:
            raise ValueError("payload and routing card memory_id must match")
        if payload.version != card.active_payload_version:
            raise ValueError("card active_payload_version must equal payload version")

        with self._connect() as connection:
            try:
                connection.execute("BEGIN")
                connection.execute(
                    """
                    UPDATE memory_payload_versions
                    SET is_active = 0
                    WHERE memory_id = ?
                    """,
                    (payload.memory_id,),
                )
                connection.execute(
                    """
                    INSERT INTO memory_payload_versions
                    (memory_id, version, payload_json, created_at, is_active)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (
                        payload.memory_id,
                        payload.version,
                        model_to_json(payload),
                        payload.created_at.isoformat(),
                    ),
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO memory_routing_cards
                    (memory_id, active_payload_version, card_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        card.memory_id,
                        card.active_payload_version,
                        model_to_json(card),
                        card.updated_at.isoformat(),
                    ),
                )
                self._bump_revision(connection)
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ValueError(
                    f"memory payload already exists: {payload.memory_id} v{payload.version}"
                ) from exc
            except Exception:
                connection.rollback()
                raise

    def get_routing_cards(self) -> list[MemoryRoutingCard]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT card_json FROM memory_routing_cards ORDER BY memory_id"
            ).fetchall()
        return [model_from_json(MemoryRoutingCard, row["card_json"]) for row in rows]

    def get_routing_card(self, memory_id: str) -> MemoryRoutingCard:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT card_json FROM memory_routing_cards WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return model_from_json(MemoryRoutingCard, row["card_json"])

    def get_payload(self, memory_id: str, version: int | None = None) -> ProcedurePayload:
        with self._connect() as connection:
            if version is None:
                row = connection.execute(
                    """
                    SELECT payload_json FROM memory_payload_versions
                    WHERE memory_id = ? AND is_active = 1
                    """,
                    (memory_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT payload_json FROM memory_payload_versions
                    WHERE memory_id = ? AND version = ?
                    """,
                    (memory_id, version),
                ).fetchone()
        if row is None:
            raise KeyError(f"{memory_id} v{version}")
        return model_from_json(ProcedurePayload, row["payload_json"])

    def get_selected_payloads(self, memory_ids: list[str]) -> list[ProcedurePayload]:
        return [self.get_payload(memory_id) for memory_id in memory_ids]

    def record_execution_evidence(self, evidence: ExecutionEvidence) -> None:
        with self._connect() as connection:
            try:
                connection.execute("BEGIN")
                row = connection.execute(
                    "SELECT card_json FROM memory_routing_cards WHERE memory_id = ?",
                    (evidence.memory_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(evidence.memory_id)
                card = model_from_json(MemoryRoutingCard, row["card_json"])
                if evidence.payload_version != card.active_payload_version:
                    raise ValueError("execution evidence payload_version is not active")

                success_contexts = list(card.execution_success_contexts)
                failure_contexts = list(card.execution_failure_contexts)
                alpha = card.execution_success_alpha
                beta = card.execution_success_beta
                success_count = card.execution_success_count
                failure_count = card.execution_failure_count

                if evidence.execution_success:
                    alpha += 1
                    success_count += 1
                    success_contexts = [*success_contexts, evidence.context][-32:]
                else:
                    beta += 1
                    failure_count += 1
                    failure_contexts = [*failure_contexts, evidence.context][-32:]

                updated_card = card.model_copy(
                    update={
                        "execution_success_alpha": alpha,
                        "execution_success_beta": beta,
                        "execution_success_count": success_count,
                        "execution_failure_count": failure_count,
                        "execution_success_contexts": success_contexts,
                        "execution_failure_contexts": failure_contexts,
                        "updated_at": utc_now(),
                    }
                )
                connection.execute(
                    """
                    INSERT INTO execution_evidence
                    (evidence_id, memory_id, payload_version, evidence_json, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        evidence.memory_id,
                        evidence.payload_version,
                        model_to_json(evidence),
                        evidence.timestamp.isoformat(),
                    ),
                )
                connection.execute(
                    """
                    UPDATE memory_routing_cards
                    SET card_json = ?, updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        model_to_json(updated_card),
                        updated_card.updated_at.isoformat(),
                        updated_card.memory_id,
                    ),
                )
                self._bump_revision(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def current_revision(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT store_revision FROM memory_store_metadata WHERE id = 1"
            ).fetchone()
        return int(row["store_revision"])

    def create_read_snapshot(self) -> MemoryStoreSnapshot:
        cards = self.get_routing_cards()
        return MemoryStoreSnapshot(
            store_revision=self.current_revision(),
            routing_cards=[card.model_copy(deep=True) for card in cards],
            active_payload_versions={
                card.memory_id: card.active_payload_version for card in cards
            },
        )
