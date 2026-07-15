# SMTR formal experiment methods

The default runtime registry contains exactly:

```text
B0
B1-Top1
B1-AllCandidates
B1-Matched
SMTR
```

The optional core-ablation registry, enabled with
`--enable-ablation-methods`, contains exactly:

```text
EffectOnly-SMTR
Static-SMTR
FactualSuccess-SMTR
```

`Robust-SMTR` remains isolated in `smtr.robust` and is not part of either
registry. Removed method identifiers are rejected; there are no runtime
aliases for historical identifiers.

## Formal SMTR

Formal SMTR traverses every proposer candidate in a deterministic seeded
random order. It shares every candidate satisfying:

```text
tau_mean > 0 and negative_risk_mean <= epsilon
```

The selected set is updated after each accepted candidate. There is no share
budget, confidence-bound gate, or support-distance veto. Proposer `top_k`
only bounds candidate search.

## Baselines and ablations

- `B0` shares nothing.
- `B1-Top1` shares the highest-ranked relevance candidate.
- `B1-AllCandidates` shares every proposer candidate.
- `B1-Matched` samples a count from the frozen SMTR validation exposure
  distribution using a stable seed derived from experiment, base episode,
  invocation, and method identifiers, then shares relevance Top-(c).
- `EffectOnly-SMTR` uses the formal checkpoint and dynamic conditioning but
  gates only on `tau_mean > 0`.
- `Static-SMTR` uses the formal checkpoint and gate while the critic always
  sees the invocation's initial selected set; the actual selected set still
  accumulates normally.
- `FactualSuccess-SMTR` uses an independent binary checkpoint trained only on
  `Y_share`, and shares when `p_share_success >= theta`. The threshold is
  frozen after validation exposure matching.

## Commands

```bash
python -m smtr.cli run-experiment \
  --methods B0 B1-Top1 B1-AllCandidates B1-Matched SMTR \
  --db data/smtr_memory.sqlite \
  --critic-checkpoint checkpoints/critic.joblib \
  --budget-manifest-path outputs/budget_manifest.json \
  --output-dir outputs/formal

python -m smtr.cli run-experiment \
  --methods EffectOnly-SMTR Static-SMTR FactualSuccess-SMTR SMTR \
  --enable-ablation-methods \
  --db data/smtr_memory.sqlite \
  --critic-checkpoint checkpoints/critic.joblib \
  --factual-success-checkpoint checkpoints/factual.joblib \
  --output-dir outputs/ablations

python -m smtr.cli run-order-sensitivity \
  --method SMTR \
  --scenario-filter prefix-sensitive \
  --enumerate-permutations \
  --critic-checkpoint checkpoints/critic.joblib \
  --output-dir outputs/order-sensitivity
```

The order diagnostic is SMTR-only, enumerates all 24 permutations for K=4,
keeps final payload presentation in proposer-rank order, and reports only
outcome flip rate, exact selected-set match rate, and mean selected-set
Jaccard.
