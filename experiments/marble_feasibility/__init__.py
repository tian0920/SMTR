"""MARBLE Real Environment Feasibility Test.

Validates SMTR feasibility in the real MARBLE environment:
  1. expose/withhold intervention is executable
  2. Stable Y₁, Y₀ paired outcomes exist
  3. SMTR critic can be trained
  4. Positive and negative transfer signal exists
  5. Current pipeline works without modification

Uses existing paired records from real MARBLE engine runs.
"""
