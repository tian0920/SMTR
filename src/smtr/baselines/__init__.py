"""Baseline memory controllers for fair comparison with SMTR-TCI.

All baseline controllers inherit from :class:`BaseMemoryController` and
share a uniform interface for memory extraction, update, retrieval, and
statistics.  No baseline modifies the TCI implementation, the persistent
memory bank internals, or any existing experiment code.

Available baselines:

  - ``reflexion``: verbal reflection memory (NeurIPS 2023)
  - ``agile``: experience consolidation (NeurIPS 2024)
  - ``heuristic_memory``: importance-scored memory management (ACL 2026)
  - ``agemem``: learned-style memory controller (ACL 2026)
"""

from smtr.baselines.base_memory_controller import BaseMemoryController

__all__ = ["BaseMemoryController"]
