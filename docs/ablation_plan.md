# Ablation Plan

## Required Methods

| ID | Method | Description |
|----|--------|-------------|
| B0 | NoMemory | Never share; baseline team performance |
| B1 | Top1Relevance | Share top-1 by card similarity |
| B2 | AllShare | Share all candidates |
| B3 | FactualSuccess | Share only high-evidence memories |
| M0 | SMTR | Full method: $\hat{\tau} > 0 \land \hat{\eta} \leq \beta$ |
| A1 | SMTR-no-risk | Ignore $\hat{\eta}$; share if $\hat{\tau} > 0$ |
| A2 | SMTR-no-writer-receiver | Critic trained without writer-receiver features |

## Ablation Goals

### B0 vs M0
Does selective sharing beat no sharing?

### B2 vs M0
Does selective sharing beat naive all-share? (Negative transfer avoidance)

### B1 vs M0
Does learned routing beat simple relevance? (Critic value)

### M0 vs A1
Does risk constraint matter? (Value of $\hat{\eta}$)

### M0 vs A2
Do writer-receiver features matter? (Cross-agent signal)

## Feature Block Ablation

- `full`: All feature blocks
- `no_writer_receiver`: Remove writer role, receiver role, wr_pair, wr_same_role, capability/tool overlap
- `no_risk`: Full features but router ignores eta_hat

## Expected Outcomes

1. SMTR > NoMemory (selective sharing helps)
2. SMTR > AllShare (risk avoidance matters)
3. SMTR > SMTR-no-risk (eta constraint prevents harm)
4. SMTR > SMTR-no-writer-receiver (cross-agent features improve decisions)
