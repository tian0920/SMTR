# SMTR Architecture

SMTR separates memory handling into two explicit stages:

1. Candidate proposal retrieves high-recall `MemoryRoutingCard` records.
2. The router decides whether each candidate payload should be exposed.

`ProcedurePayload` objects are stored in `SharedMemoryPool`, but neither the
candidate proposer nor the router receives those payloads. Agent-local context is
the only place where selected payloads can appear. With `NoMemoryRouter`, every
candidate is withheld, so each agent receives an empty `visible_payloads` list.

The SQLite implementation extends this split with versioned payload rows and a
single active `MemoryRoutingCard` per memory. `get_routing_cards()` reads only
the routing card table, while `get_selected_payloads()` loads full procedures
only after router selection.
