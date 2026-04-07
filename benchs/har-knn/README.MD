# knn and har-knn

## WISDM and PAMAP2 dataset

| Dataset | Size | Description | Considered HAR parameters | #classes / #features | #training / testing instances |
|---|---|---|---|---|---|
| WISDM \[WISDM2010\] | Small | 1 3-axes accelerometer<br>6 activities | window size of 200, an overlap of 50% | 6/43 | knn: 7,383 / 3,163<br>har-knn: 738,400 / 316,400 |
| PAMAP2 \[PAMAP2\] \[PAMAP2012\] | Medium | 3 3-axes sensors used (accel., magneto., gyroscope) in each of the 3 IMUs<br>Data of 8 users for training, and user #5 data for testing<br>21 Activities + 1 activity representing transient activities | a window size of 300, an overlap of 10% \[FMC23\] | 22/90 | knn: 6,186 / 1,008<br>har-knn: 1,670,250 / 272,190 |
