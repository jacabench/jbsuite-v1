# knn and har-knn

## WISDM and PAMAP2 dataset

| Dataset | Size | Description | Considered HAR parameters | #classes / #features | #training / testing instances |
|---|---|---|---|---|---|
| WISDM \[WISDM2010\] | Small | 1 3-axes accelerometer<br>6 activities | window size of 200, an overlap of 50% | 6/43 | knn: 7,383 / 3,163<br>har-knn: 738,400 / 316,400 |
| PAMAP2 \[PAMAP2\] \[PAMAP2012\] | Medium | 3 3-axes sensors used (accel., magneto., gyroscope) in each of the 3 IMUs<br>Data of 8 users for training, and user #5 data for testing<br>21 Activities + 1 activity representing transient activities | a window size of 300, an overlap of 10% \[FMC23\] | 22/90 | knn: 6,186 / 1,008<br>har-knn: 1,670,250 / 272,190 |

## References

- \[WISDM2010\] Jennifer R. Kwapisz, Gary M. Weiss, and Samuel A. Moore. 2011. Activity recognition using cell phone accelerometers. *SIGKDD Explor. Newsl.* 12, 2 (December 2010), 74–82. doi: 10.1145/1964897.1964918
- \[WISDM\] “WISDM: Wireless Sensor Data Mining,” Department of Computer & Information Science, Fordham University, Bronx, NY, USA.
- \[PAMAP2012\] A. Reiss and D. Stricker, "Introducing a New Benchmarked Dataset for Activity Monitoring," *2012 16th International Symposium on Wearable Computers*, Newcastle, UK, 2012, pp. 108–109, doi: 10.1109/ISWC.2012.13.
- \[PAMAP2\] “PAMAP2 Physical Activity Monitoring.” https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring 
- \[FMC23\] Paulo J.S. Ferreira, Joao Mendes Moreira, and Joao M.P. Cardoso. 2023. A Study on Hyperparameters Configurations for an Efficient Human Activity Recognition System. In *Proceedings of the 8th international Workshop on Sensor-Based Activity Recognition and Artificial Intelligence* (iWOAR '23). Association for Computing Machinery, New York, NY, USA, Article 11, 1–6. doi: 10.1145/3615834.3615851
- \[CH67\] T. Cover and P. Hart. Nearest Neighbor Pattern Classification. *IEEE Transactions on Information Theory*, 13(1):21–27, 1967. doi: 10.1109/TIT.1967.1053964.

## Some knn options

| Variants | k | Top k \| update | Threads |
|---|---|---|---|
| a | k={3, 5, 7, 20}<br>specialized=0<br><br>{k=3, specialized=1} | top_k={1, 2, 3} | single thread |
| b |  | update={1, 2} | single thread |
| a-omp |  | top_k={1, 2, 3} | Number of threads launched by openmp runtime; or specified by a #define macro {1..N} |
| b-omp |  | update={1, 2} | Number of threads launched by openmp runtime; or specified by a #define macro {1..N} |
| b-omp1 |  | update={1, 2} | Number of threads specified by a #define macro {1..N} |

## Knn variants

| Variant | Top HLS function | Main differences |
|---|---|---|
| Knn-ver1.0a | Classify | Distances are calculated and all stored, and then the nearest are determined |
| Knn-ver1.0a-deploy1 | Normalize + classify | + local definition of constants |
| Knn-ver1.0a-deploy2 | Normalize + classify | + local arrays of max-min feature values |
| Knn-ver1.0a-deploy3 | Normalize + classify | + local array of knowledge base |
| Knn-ver1.0b | Classify | Each distance calculated is used for updating the k nearest |
| Knn-ver1.0b-deploy1 | Normalize + classify | + local definition of constants |
| Knn-ver1.0b-deploy2 | Normalize + classify | + local arrays of max-min feature values |
| Knn-ver1.0b-deploy3 | Normalize + classify | + local array of knowledge base |
| Knn-ver1.0a-omp | n.a. | The outermost loop for calculating and storing the distances is parallelized |
| Knn-ver1.0b-omp | n.a. | The outermost loop for calculating distances is parallelized, and the code to update the k nearest is defined as critical |
| Knn-ver1.0a-omp1 | n.a. | The outermost loop for calculating distances and updating the k nearest is parallelized, and a subsequent step merges the individual k nearest |

## Har-knn variants

| Variant | Type | Data storages | Options | READ options | Top HLS function |
|---|---|---|---|---|---|
| Ver1.0a | Streaming is an option | Data communicated from main | w/o specializations and options,<br>TOP_K=1 | 1,2,3,4 | Classify |
| Ver1.0a-opt | Streaming is an option | Data communicated from main | w/ specializations and other options | 1,2,3,4 | Classify |
| Ver1.0a-opt-deploy1 | streaming | Data communicated from main | w/ specializations and other options | 3, 4 | Window-buffer + feature extraction + normalization + classify |
| Ver1.0a-opt-deploy2 | streaming | Only partial data local | w/ specializations and other options | 3, 4 | Window-buffer + feature extraction + normalization + classify |
| Ver1.0a-opt-deploy3 | streaming | All data local | w/ specializations and other options | 3, 4 | Window-buffer + feature extraction + normalization + classify |
| Ver1.0b | Streaming is an option | Data communicated from main | w/o specializations and options, UPDATE=1 | 1,2,3,4 | Classify |
| Ver1.0b-opt | Streaming is an option | Data communicated from main | w/ specializations and other options | 1,2,3,4 | Classify |
| Ver1.0b-opt-deploy1 | streaming | Data communicated from main | w/ specializations and other options | 3, 4 | Window-buffer + feature extraction + normalization + classify |
| Ver1.0b-opt-deploy2 | streaming | Only partial data local | w/ specializations and other options | 3, 4 | Window-buffer + feature extraction + normalization + classify |
| Ver1.0b-opt-deploy3 | streaming | All data local | w/ specializations and other options | 3, 4 | Window-buffer + feature extraction + normalization + classify |
