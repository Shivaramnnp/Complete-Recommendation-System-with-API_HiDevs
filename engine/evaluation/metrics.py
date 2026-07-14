import numpy as np
from typing import List

def precision_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Computes Precision@K.
    Formula: (Relevant Items in Top K) / K
    """
    if k <= 0:
        return 0.0
        
    top_k_recs = recommended_items[:k]
    relevant_set = set(relevant_items)
    
    hits = sum(1 for item in top_k_recs if item in relevant_set)
    return float(hits / k)

def recall_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Computes Recall@K.
    Formula: (Relevant Items in Top K) / (Total Relevant Items)
    """
    if not relevant_items or k <= 0:
        return 0.0
        
    top_k_recs = recommended_items[:k]
    relevant_set = set(relevant_items)
    
    hits = sum(1 for item in top_k_recs if item in relevant_set)
    return float(hits / len(relevant_items))

def ndcg_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain @ K (NDCG@K) with binary relevance.
    Formula: DCG@K / IDCG@K
    DCG@K = sum_{i=1}^K [ rel_i / log2(i + 1) ]
    IDCG@K = sum_{i=1}^min(|Rel|, K) [ 1 / log2(i + 1) ]
    """
    if not relevant_items or k <= 0:
        return 0.0
        
    top_k_recs = recommended_items[:k]
    relevant_set = set(relevant_items)
    
    # Compute DCG@K
    dcg = 0.0
    for idx, item in enumerate(top_k_recs):
        rel = 1 if item in relevant_set else 0
        # log2(rank + 1) where rank is 1-indexed, i.e., rank = idx + 1
        dcg += rel / np.log2(idx + 2)
        
    # Compute IDCG@K (Ideal DCG where all top hits are relevant)
    idcg = 0.0
    n_relevant = min(len(relevant_items), k)
    for idx in range(n_relevant):
        idcg += 1.0 / np.log2(idx + 2)
        
    if idcg == 0:
        return 0.0
        
    return float(dcg / idcg)
