from engine.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

def test_precision_at_k():
    recs = [1, 2, 3, 4, 5]
    relevant = [2, 4, 6]
    # At K=3: recs are [1, 2, 3]. Hits: 2 is relevant (1 hit). Precision@3 = 1/3
    assert abs(precision_at_k(recs, relevant, 3) - (1.0 / 3)) < 1e-6
    # At K=5: recs are [1, 2, 3, 4, 5]. Hits: 2, 4 (2 hits). Precision@5 = 2/5 = 0.4
    assert precision_at_k(recs, relevant, 5) == 0.4

def test_recall_at_k():
    recs = [1, 2, 3, 4, 5]
    relevant = [2, 4, 6]
    # At K=3: Hits: 2. Total relevant: 3. Recall@3 = 1/3
    assert abs(recall_at_k(recs, relevant, 3) - (1.0 / 3)) < 1e-6
    # At K=5: Hits: 2, 4. Total relevant: 3. Recall@5 = 2/3
    assert abs(recall_at_k(recs, relevant, 5) - (2.0 / 3)) < 1e-6

def test_ndcg_at_k():
    recs = [1, 2, 3, 4, 5]
    relevant = [2, 4, 6]
    
    # At K=3: hits are at ranks 2 (idx 1).
    # DCG@3 = 1 / log2(2 + 1) = 1 / log2(3) = 0.6309
    # IDCG@3 = 1 / log2(2) = 1.0 (since max hits we could get is min(len(relevant), K) = 3)
    # NDCG@3 = DCG@3 / IDCG@3 = 0.6309 / 1.0 (wait, let's verify if IDCG@3 ideal configuration would place 2 hits at first 2 spots:
    # 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309. Wait, yes, for IDCG we assume the top min(|Rel|, K) ranks are filled with hits.
    # So for 3 relevant items and K=3, the ideal ranking would have 3 hits:
    # IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.6309 + 0.5 = 2.1309.
    # Wait, in our case, there are only 3 relevant items in the global set, so the maximum hits we can get in top-3 is 3.
    # Therefore, the ideal DCG (IDCG@3) is indeed 1.0/log2(2) + 1.0/log2(3) + 1.0/log2(4) = 2.1309.
    # Let's calculate DCG@3 for recs:
    # idx 0: item 1 (not rel) -> 0
    # idx 1: item 2 (rel) -> 1 / log2(3)
    # idx 2: item 3 (not rel) -> 0
    # DCG@3 = 1 / log2(3) = 0.63092975
    # IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.63092975 + 0.5 = 2.13092975
    # Expected NDCG@3 = 0.63092975 / 2.13092975 = 0.29608
    
    val = ndcg_at_k(recs, relevant, 3)
    expected = (1.0 / 1.5849625) / (1.0 + 1.0 / 1.5849625 + 0.5)
    assert abs(val - expected) < 1e-6
    
    # If no hits, NDCG should be 0.0
    assert ndcg_at_k(recs, [99], 3) == 0.0
