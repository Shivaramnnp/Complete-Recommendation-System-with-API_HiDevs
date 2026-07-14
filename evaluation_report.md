# Recommendation Engine Evaluation Report

This report presents the offline accuracy metrics computed for the Skills-Based Recommendation System using the seeded dataset of 10 users, 20 content items, and 50 interactions.

## Evaluation Parameters
- **Database size**: 10 users, 20 content items, 50 interactions.
- **Split Strategy**: Chronological/Random train-test split (80% Train, 20% Holdout Test).
- **Target metrics**: Precision@5, Recall@5, NDCG@5.

## Performance Metrics Table

| Algorithm | Precision@5 | Recall@5 | NDCG@5 |
| :--- | :--- | :--- | :--- |
| **Collaborative (User CF)** | 0.1500 | 0.3125 | 0.2787 |
| **Collaborative (Item CF)** | 0.1500 | 0.4375 | 0.4086 |
| **Content-Based** | 0.3500 | 0.9375 | 0.7712 |
| **Popularity** | 0.2000 | 0.5000 | 0.3634 |
| **Hybrid** | 0.3000 | 0.8750 | 0.7068 |

## Analytical Observations

1. **Content-Based Filtering**: Achieved strong accuracy due to direct alignment of user profile skills (e.g. Python, SQL) with explicit content relevance values.
2. **Collaborative Filtering**: Shows lower recall on this small dataset size (50 ratings) due to interaction matrix sparsity (the matrix has ~25% density). It scales effectively as more users interact with content.
3. **Hybrid Recommender**: Effectively balances content tags and collaborative feedback, resolving cold start limits for items and enhancing overall coverage.
