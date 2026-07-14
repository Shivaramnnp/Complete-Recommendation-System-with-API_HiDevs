import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two numeric vectors.
    Formula: (v1 . v2) / (||v1|| * ||v2||)
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return float(dot_product / (norm_v1 * norm_v2))

def jaccard_similarity(s1: set, s2: set) -> float:
    """
    Computes Jaccard Similarity between two sets.
    Formula: |s1 intersect s2| / |s1 union s2|
    """
    if not s1 or not s2:
        return 0.0
    
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    
    if union == 0:
        return 0.0
        
    return float(intersection / union)

def pearson_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the Pearson Correlation Coefficient between two vectors.
    Formula: cov(v1, v2) / (std(v1) * std(v2)) after aligning common indices,
    or centered cosine similarity.
    """
    # Find common rated items (where both vectors are non-zero/not NaN)
    # We assume v1 and v2 are aligned dense vectors representing ratings for the same item set.
    # To compute Pearson, we only consider indices where both vectors have non-zero ratings.
    common_idx = np.logical_and(v1 > 0, v2 > 0)
    if not np.any(common_idx):
        return 0.0
        
    v1_common = v1[common_idx]
    v2_common = v2[common_idx]
    
    if len(v1_common) < 2:
        # Not enough overlap to compute correlation
        return 0.0
        
    v1_mean = np.mean(v1_common)
    v2_mean = np.mean(v2_common)
    
    v1_centered = v1_common - v1_mean
    v2_centered = v2_common - v2_mean
    
    num = np.dot(v1_centered, v2_centered)
    den = np.linalg.norm(v1_centered) * np.linalg.norm(v2_centered)
    
    if den == 0:
        return 0.0
        
    return float(num / den)
