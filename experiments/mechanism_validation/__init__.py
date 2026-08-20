"""SMTR Mechanism Validation Framework.

Independent validation pipeline for SMTR core mechanism.

Does NOT modify:
  - Main training pipeline
  - Existing experiments
  - Critic architecture

All experiments run independently and output unified JSON + Markdown reports.
"""
