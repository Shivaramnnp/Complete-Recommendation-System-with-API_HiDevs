import numpy as np
from engine.similarity import cosine_similarity, jaccard_similarity, pearson_correlation

def test_cosine_similarity():
    # Identical vectors
    v1 = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v1, v1) - 1.0) < 1e-6
    
    # Orthogonal vectors
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.0, 1.0])
    assert cosine_similarity(v2, v3) == 0.0
    
    # Opposite vectors
    v4 = np.array([-1.0, -2.0])
    v5 = np.array([1.0, 2.0])
    assert abs(cosine_similarity(v4, v5) - (-1.0)) < 1e-6

def test_jaccard_similarity():
    s1 = {"action", "sci-fi", "thriller"}
    s2 = {"action", "adventure"}
    # intersection: {'action'} (1)
    # union: {'action', 'sci-fi', 'thriller', 'adventure'} (4)
    # jaccard: 1/4 = 0.25
    assert jaccard_similarity(s1, s2) == 0.25
    
    # No overlap
    s3 = {"comedy"}
    assert jaccard_similarity(s1, s3) == 0.0

def test_pearson_correlation():
    # Standard positive correlation
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([2.0, 4.0, 6.0])
    assert abs(pearson_correlation(v1, v2) - 1.0) < 1e-6
    
    # Negative correlation
    v3 = np.array([3.0, 2.0, 1.0])
    assert abs(pearson_correlation(v1, v3) - (-1.0)) < 1e-6
    
    # Aligned indices (ratings > 0 only)
    v4 = np.array([5.0, 0.0, 3.0])  # rated item 0 and item 2
    v5 = np.array([4.0, 1.0, 2.0])  # rated item 0, 1, 2
    # common elements: item 0 (ratings: 5, 4) and item 2 (ratings: 3, 2)
    # means: mean(v4_common) = 4, mean(v5_common) = 3
    # centered: v4_c = [1, -1], v5_c = [1, -1]
    # correlation: (1*1 + -1*-1) / (sqrt(2)*sqrt(2)) = 2 / 2 = 1.0
    assert abs(pearson_correlation(v4, v5) - 1.0) < 1e-6
