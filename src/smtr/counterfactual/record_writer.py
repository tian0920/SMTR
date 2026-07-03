import json
from pathlib import Path

from smtr.counterfactual.schemas import PairedInterventionRecord


class DuplicateRecordError(ValueError):
    pass


class PairedRecordWriter:
    def __init__(self, path: str | Path, *, allow_duplicates: bool = False) -> None:
        self.path = Path(path)
        self.allow_duplicates = allow_duplicates
        if self.path.parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: PairedInterventionRecord) -> None:
        if not self.allow_duplicates and record.record_id in self._record_ids():
            raise DuplicateRecordError(f"record already exists: {record.record_id}")
        line = json.dumps(record.model_dump(mode="json"), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def load(self) -> list[PairedInterventionRecord]:
        if not self.path.exists():
            return []
        records: list[PairedInterventionRecord] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(PairedInterventionRecord.model_validate_json(line))
        return records

    def _record_ids(self) -> set[str]:
        if not self.path.exists() or self.allow_duplicates:
            return set()
        ids: set[str] = set()
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                ids.add(json.loads(line)["record_id"])
        return ids
