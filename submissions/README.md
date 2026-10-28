# Submission candidates

| File | Model | Status |
| --- | --- | --- |
| [submission_improved.csv](submission_improved.csv) | TF-IDF + keyword + frozen MPNet embeddings | Current candidate; nested OOF F1 0.7973 |
| [submission.csv](submission.csv) | TF-IDF + keyword | Original reference; OOF F1 0.7699 |
| [submission_sparse_candidate.csv](submission_sparse_candidate.csv) | Sparse search and threshold tuning | Rejected; nested OOF F1 0.7664 |

Each contains `id,target` with 3,263 rows in sample-submission order. These are local predictions, not recorded Kaggle scores. Scripts do not upload files automatically.
