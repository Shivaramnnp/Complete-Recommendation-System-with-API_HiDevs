from typing import List, Optional, Dict
from engine.strategy import RecommendationStrategy, UserInteraction, RecommendationItem
from engine.collaborative import CollaborativeFiltering
from engine.content_based import ContentBasedFiltering

class HybridRecommendation(RecommendationStrategy):
    """
    Hybrid Recommendation Strategy combining Collaborative Filtering and Content-Based Filtering.
    Uses score normalization and weighted linear interpolation.
    """
    
    def __init__(
        self,
        cf_weight: float = 0.5,
        cb_weight: float = 0.5,
        cf_strategy: Optional[CollaborativeFiltering] = None,
        cb_strategy: Optional[ContentBasedFiltering] = None
    ):
        """
        :param cf_weight: Weight for Collaborative Filtering scores (0.0 to 1.0)
        :param cb_weight: Weight for Content-Based Filtering scores (0.0 to 1.0)
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        
        # Instantiate defaults if not provided
        self.cf_model = cf_strategy if cf_strategy is not None else CollaborativeFiltering(kind="user")
        self.cb_model = cb_strategy if cb_strategy is not None else ContentBasedFiltering()
        
    def fit(self, interactions: List[UserInteraction], items_info: Optional[List[dict]] = None) -> None:
        # Fit both constituent models
        self.cf_model.fit(interactions, items_info)
        self.cb_model.fit(interactions, items_info)
        
    def recommend(self, user_id: int, k: int = 10) -> List[RecommendationItem]:
        # Generate recommendations from both models
        # Fetching a larger list (k * 3) from each to ensure enough overlap and diverse blend
        fetch_k = max(k * 3, 50)
        
        cf_recs = self.cf_model.recommend(user_id, k=fetch_k)
        cb_recs = self.cb_model.recommend(user_id, k=fetch_k)
        
        if not cf_recs and not cb_recs:
            return []
            
        # Normalize scores from both models to [0, 1] range to avoid magnitude bias
        cf_scores = self._normalize_scores(cf_recs)
        cb_scores = self._normalize_scores(cb_recs)
        
        # Blend scores
        blended_scores: Dict[int, float] = {}
        
        # Combine items from both models
        all_item_ids = set(cf_scores.keys()).union(set(cb_scores.keys()))
        
        for item_id in all_item_ids:
            score_cf = cf_scores.get(item_id, 0.0)
            score_cb = cb_scores.get(item_id, 0.0)
            
            # Weighted sum
            blended_scores[item_id] = (self.cf_weight * score_cf) + (self.cb_weight * score_cb)
            
        # Sort and return top K
        sorted_recs = [
            RecommendationItem(item_id=iid, score=score)
            for iid, score in blended_scores.items()
        ]
        sorted_recs.sort(key=lambda x: x.score, reverse=True)
        
        return sorted_recs[:k]
        
    def _normalize_scores(self, recs: List[RecommendationItem]) -> Dict[int, float]:
        if not recs:
            return {}
            
        scores = [r.score for r in recs]
        min_score = min(scores)
        max_score = max(scores)
        denom = max_score - min_score
        
        normalized: Dict[int, float] = {}
        for r in recs:
            if denom > 0:
                normalized[r.item_id] = (r.score - min_score) / denom
            else:
                normalized[r.item_id] = 1.0  # If all scores are equal, treat similarity as equal/max
                
        return normalized
