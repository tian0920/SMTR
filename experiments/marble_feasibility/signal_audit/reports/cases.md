# MARBLE Causal Signal Audit — Case Analysis

## Positive Transfer (10 cases)

### Case 1

- **Task:** 10
- **Memory:** dbproc-01870edcc464-agent1
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** cross_receiver_anchor
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Positive transfer: memory helped. Same task: expected ['INSERT_LARGE_DATA']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.3333.

### Case 2

- **Task:** 10
- **Memory:** dbproc-01870edcc464-agent1
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** cross_receiver_anchor
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Positive transfer: memory helped. Same task: expected ['INSERT_LARGE_DATA']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.3333.

### Case 3

- **Task:** 10
- **Memory:** dbproc-01870edcc464-agent5
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** semantic_top
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Positive transfer: memory helped. Same task: expected ['INSERT_LARGE_DATA']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.3333.

### Case 4

- **Task:** 10
- **Memory:** dbproc-06c28b92d681-agent1
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** semantic_top
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Positive transfer: memory helped. Same task: expected ['INSERT_LARGE_DATA']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.3333.

### Case 5

- **Task:** 10
- **Memory:** dbproc-01870edcc464-agent3
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** semantic_top
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Positive transfer: memory helped. Same task: expected ['INSERT_LARGE_DATA']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.3333.

### Case 6

- **Task:** 11
- **Memory:** dbproc-06c28b92d681-agent1
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** semantic_top
- **Control expected:** ['VACUUM']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['VACUUM']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['VACUUM']
- **Description:** Positive transfer: memory helped. Same task: expected ['VACUUM']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['VACUUM']. F1: control=0.0, share=0.3333.

### Case 7

- **Task:** 17
- **Memory:** dbproc-01870edcc464-agent5
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** semantic_top
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Positive transfer: memory helped. Same task: expected ['INSERT_LARGE_DATA']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.3333.

### Case 8

- **Task:** 19
- **Memory:** dbproc-01870edcc464-agent1
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** cross_receiver_anchor
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Positive transfer: memory helped. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.0, share=0.2857.

### Case 9

- **Task:** 2
- **Memory:** dbproc-01870edcc464-agent3
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** semantic_top
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['LOCK_CONTENTION', 'REDUNDANT_INDEX']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Positive transfer: memory helped. Same task: expected ['INSERT_LARGE_DATA']. Predictions diverged: control predicted ['LOCK_CONTENTION', 'REDUNDANT_INDEX'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.3333.

### Case 10

- **Task:** 21
- **Memory:** dbproc-06c28b92d681-agent1
- **Receiver:** agent1
- **τ:** 1
- **Label:** positive_transfer
- **Source:** semantic_top
- **Control expected:** ['VACUUM']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['VACUUM']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['VACUUM']
- **Description:** Positive transfer: memory helped. Same task: expected ['VACUUM']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['VACUUM']. F1: control=0.0, share=0.3333.

## Negative Transfer (10 cases)

### Case 1

- **Task:** 19
- **Memory:** dbproc-01870edcc464-agent2
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** receiver_incompatible_hard_negative
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Negative transfer: memory hurt. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM'], share predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.3333, share=0.0.

### Case 2

- **Task:** 19
- **Memory:** dbproc-01870edcc464-agent3
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** semantic_top
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Negative transfer: memory hurt. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM'], share predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.3333, share=0.0.

### Case 3

- **Task:** 19
- **Memory:** dbproc-01870edcc464-agent4
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** semantic_top
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Negative transfer: memory hurt. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM'], share predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.3333, share=0.0.

### Case 4

- **Task:** 19
- **Memory:** dbproc-01870edcc464-agent5
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** semantic_top
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES', 'VACUUM']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Negative transfer: memory hurt. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM'], share predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES', 'VACUUM']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.3333, share=0.0.

### Case 5

- **Task:** 26
- **Memory:** dbproc-06c28b92d681-agent4
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** semantic_top
- **Control expected:** ['FETCH_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['FETCH_LARGE_DATA']
- **Share predicted:** ['LOCK_CONTENTION', 'MISSING_INDEXES']
- **Root causes:** ['FETCH_LARGE_DATA']
- **Description:** Negative transfer: memory hurt. Same task: expected ['FETCH_LARGE_DATA']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['LOCK_CONTENTION', 'MISSING_INDEXES']. Root causes: ['FETCH_LARGE_DATA']. F1: control=0.6667, share=0.0.

### Case 6

- **Task:** 3
- **Memory:** dbproc-01870edcc464-agent5
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** semantic_top
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['LOCK_CONTENTION', 'MISSING_INDEXES']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Negative transfer: memory hurt. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM'], share predicted ['LOCK_CONTENTION', 'MISSING_INDEXES']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.3333, share=0.0.

### Case 7

- **Task:** 3
- **Memory:** dbproc-06c28b92d681-agent1
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** semantic_top
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Negative transfer: memory hurt. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM'], share predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.3333, share=0.0.

### Case 8

- **Task:** 3
- **Memory:** dbproc-01870edcc464-agent4
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** semantic_top
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Negative transfer: memory hurt. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM'], share predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.3333, share=0.0.

### Case 9

- **Task:** 3
- **Memory:** dbproc-01870edcc464-agent1
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** cross_receiver_anchor
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Negative transfer: memory hurt. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM'], share predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.3333, share=0.0.

### Case 10

- **Task:** 3
- **Memory:** dbproc-01870edcc464-agent2
- **Receiver:** agent1
- **τ:** -1
- **Label:** negative_transfer
- **Source:** receiver_incompatible_hard_negative
- **Control expected:** ['REDUNDANT_INDEX']
- **Control predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Share expected:** ['REDUNDANT_INDEX']
- **Share predicted:** ['LOCK_CONTENTION', 'MISSING_INDEXES']
- **Root causes:** ['REDUNDANT_INDEX']
- **Description:** Negative transfer: memory hurt. Same task: expected ['REDUNDANT_INDEX']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM'], share predicted ['LOCK_CONTENTION', 'MISSING_INDEXES']. Root causes: ['REDUNDANT_INDEX']. F1: control=0.3333, share=0.0.

## Neutral Success (10 cases)

### Case 1

- **Task:** 13
- **Memory:** dbproc-01870edcc464-agent4
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** semantic_top
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.6667.

### Case 2

- **Task:** 16
- **Memory:** dbproc-01870edcc464-agent3
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** semantic_top
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.6667.

### Case 3

- **Task:** 16
- **Memory:** dbproc-01870edcc464-agent1
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** cross_receiver_anchor
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.3333.

### Case 4

- **Task:** 16
- **Memory:** dbproc-01870edcc464-agent2
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** receiver_incompatible_hard_negative
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.6667.

### Case 5

- **Task:** 16
- **Memory:** dbproc-01870edcc464-agent4
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** semantic_top
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.3333.

### Case 6

- **Task:** 16
- **Memory:** dbproc-06c28b92d681-agent1
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** semantic_top
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION'], share predicted ['FETCH_LARGE_DATA', 'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX', 'VACUUM']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.3333.

### Case 7

- **Task:** 16
- **Memory:** dbproc-01870edcc464-agent3
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** semantic_top
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.6667.

### Case 8

- **Task:** 16
- **Memory:** dbproc-01870edcc464-agent1
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** cross_receiver_anchor
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.6667.

### Case 9

- **Task:** 16
- **Memory:** dbproc-01870edcc464-agent2
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** receiver_incompatible_hard_negative
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.6667.

### Case 10

- **Task:** 16
- **Memory:** dbproc-01870edcc464-agent4
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_success
- **Source:** semantic_top
- **Control expected:** ['LOCK_CONTENTION']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['LOCK_CONTENTION']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['LOCK_CONTENTION']
- **Description:** Neutral: no transfer effect. Same task: expected ['LOCK_CONTENTION']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['LOCK_CONTENTION']. F1: control=0.6667, share=0.6667.

## Neutral Failure (10 cases)

### Case 1

- **Task:** 10
- **Memory:** dbproc-06c28b92d681-agent1
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** semantic_top
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Neutral: no transfer effect. Same task: expected ['INSERT_LARGE_DATA']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.0.

### Case 2

- **Task:** 10
- **Memory:** dbproc-01870edcc464-agent3
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** semantic_top
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Neutral: no transfer effect. Same task: expected ['INSERT_LARGE_DATA']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.0.

### Case 3

- **Task:** 10
- **Memory:** dbproc-01870edcc464-agent1
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** cross_receiver_anchor
- **Control expected:** ['INSERT_LARGE_DATA']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['INSERT_LARGE_DATA']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['INSERT_LARGE_DATA']
- **Description:** Neutral: no transfer effect. Same task: expected ['INSERT_LARGE_DATA']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['INSERT_LARGE_DATA']. F1: control=0.0, share=0.0.

### Case 4

- **Task:** 100
- **Memory:** dbproc-01870edcc464-agent2
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** receiver_incompatible_hard_negative
- **Control expected:** ['FETCH_LARGE_DATA', 'VACUUM']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX']
- **Share expected:** ['FETCH_LARGE_DATA', 'VACUUM']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES']
- **Root causes:** ['VACUUM', 'FETCH_LARGE_DATA']
- **Description:** Neutral: no transfer effect. Same task: expected ['FETCH_LARGE_DATA', 'VACUUM']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX'], share predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES']. Root causes: ['VACUUM', 'FETCH_LARGE_DATA']. F1: control=0.4, share=0.4.

### Case 5

- **Task:** 100
- **Memory:** dbproc-06c28b92d681-agent1
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** semantic_top
- **Control expected:** ['FETCH_LARGE_DATA', 'VACUUM']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX']
- **Share expected:** ['FETCH_LARGE_DATA', 'VACUUM']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES']
- **Root causes:** ['VACUUM', 'FETCH_LARGE_DATA']
- **Description:** Neutral: no transfer effect. Same task: expected ['FETCH_LARGE_DATA', 'VACUUM']. Predictions diverged: control predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX'], share predicted ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES']. Root causes: ['VACUUM', 'FETCH_LARGE_DATA']. F1: control=0.4, share=0.4.

### Case 6

- **Task:** 100
- **Memory:** dbproc-01870edcc464-agent3
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** semantic_top
- **Control expected:** ['FETCH_LARGE_DATA', 'VACUUM']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX']
- **Share expected:** ['FETCH_LARGE_DATA', 'VACUUM']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX']
- **Root causes:** ['VACUUM', 'FETCH_LARGE_DATA']
- **Description:** Neutral: no transfer effect. Same task: expected ['FETCH_LARGE_DATA', 'VACUUM']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'REDUNDANT_INDEX']. Root causes: ['VACUUM', 'FETCH_LARGE_DATA']. F1: control=0.4, share=0.4.

### Case 7

- **Task:** 11
- **Memory:** dbproc-01870edcc464-agent1
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** cross_receiver_anchor
- **Control expected:** ['VACUUM']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['VACUUM']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['VACUUM']
- **Description:** Neutral: no transfer effect. Same task: expected ['VACUUM']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['VACUUM']. F1: control=0.0, share=0.0.

### Case 8

- **Task:** 11
- **Memory:** dbproc-01870edcc464-agent2
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** receiver_incompatible_hard_negative
- **Control expected:** ['VACUUM']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['VACUUM']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['VACUUM']
- **Description:** Neutral: no transfer effect. Same task: expected ['VACUUM']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['VACUUM']. F1: control=0.0, share=0.0.

### Case 9

- **Task:** 11
- **Memory:** dbproc-01870edcc464-agent5
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** semantic_top
- **Control expected:** ['VACUUM']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Share expected:** ['VACUUM']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']
- **Root causes:** ['VACUUM']
- **Description:** Neutral: no transfer effect. Same task: expected ['VACUUM']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION']. Root causes: ['VACUUM']. F1: control=0.0, share=0.0.

### Case 10

- **Task:** 100
- **Memory:** dbproc-06c28b92d681-agent1
- **Receiver:** agent1
- **τ:** 0
- **Label:** neutral_failure
- **Source:** semantic_top
- **Control expected:** ['FETCH_LARGE_DATA', 'VACUUM']
- **Control predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES']
- **Share expected:** ['FETCH_LARGE_DATA', 'VACUUM']
- **Share predicted:** ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES']
- **Root causes:** ['VACUUM', 'FETCH_LARGE_DATA']
- **Description:** Neutral: no transfer effect. Same task: expected ['FETCH_LARGE_DATA', 'VACUUM']. Same predictions: ['FETCH_LARGE_DATA', 'LOCK_CONTENTION', 'MISSING_INDEXES']. Root causes: ['VACUUM', 'FETCH_LARGE_DATA']. F1: control=0.4, share=0.4.
