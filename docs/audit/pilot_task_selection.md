# Pilot Task Selection

**Method**: Difficulty-aware stratified sampling

**Source**: 100 tasks from difficulty profiling

**Per difficulty per domain**: 2 tasks


## Selected Tasks


### Hard (1 tasks)

| Domain | Task ID | Mean Reward |
|--------|---------|-------------|
| research | 83 | 0.0 |

### Medium (0 tasks)

| Domain | Task ID | Mean Reward |
|--------|---------|-------------|

### Easy (10 tasks)

| Domain | Task ID | Mean Reward |
|--------|---------|-------------|
| bargaining | 12 | 1.0 |
| bargaining | 14 | 1.0 |
| coding | 1 | 1.0 |
| coding | 12 | 1.0 |
| database | 11 | 1.0 |
| database | 13 | 1.0 |
| minecraft | 11 | 1.0 |
| minecraft | 13 | 1.0 |
| research | 10 | 1.0 |
| research | 100 | 1.0 |

## Selection Rationale

Tasks are selected based on **measured difficulty** from the no_memory baseline,
not manual choice. Each difficulty tier is represented across all 5 domains.

- **Hard**: reward ≤ 0.5 → baseline fails, memory opportunity exists
- **Medium**: 0.5 < reward ≤ 0.9 → partial improvement margin
- **Easy**: reward > 0.9 → ceiling effect, control group