from typing import List, Dict
from engine.strategy import RecommendationItem

class WeightedRanker:
    """
    Ranks recommended items by combining personalization score with global rating quality metrics.
    Uses a Bayesian average (IMDb style) for item popularity and blends it with recommendation scores.
    """
    
    def __init__(self, personalization_weight: float = 0.7, min_ratings: int = 5):
        """
        :param personalization_weight: Weight for personalization score (0.0 to 1.0)
        :param min_ratings: Threshold for Bayesian average calculation (m)
        """
        self.personalization_weight = personalization_weight
        self.min_ratings = min_ratings
        
    def rank(
        self,
        personalized_recs: List[RecommendationItem],
        item_avg_ratings: Dict[int, float],
        item_rating_counts: Dict[int, int],
        global_mean_rating: float
    ) -> List[RecommendationItem]:
        """
        Adjusts recommendation scores by blending personalization with item global rating quality.
        Formula:
          BayesianRating = (v / (v + m)) * R + (m / (v + m)) * C
          FinalScore = w * Personalization + (1 - w) * BayesianRating
        """
        ranked_items = []
        
        # Max personalization score for scaling
        max_pers_score = max([r.score for r in personalized_recs]) if personalized_recs else 1.0
        min_pers_score = min([r.score for r in personalized_recs]) if personalized_recs else 0.0
        pers_range = max_pers_score - min_pers_score
        
        # Scale global mean rating to 0-1 range assuming ratings are out of 5 stars
        max_rating = 5.0
        
        for rec in personalized_recs:
            item_id = rec.item_id
            
            # 1. Normalize personalization score to [0, 1]
            norm_pers = (rec.score - min_pers_score) / pers_range if pers_range > 0 else 1.0
            
            # 2. Get item rating stats
            v = item_rating_counts.get(item_id, 0)
            R = item_avg_ratings.get(item_id, global_mean_rating)
            C = global_mean_rating
            m = self.min_ratings
            
            # Compute Bayesian Average
            if v + m > 0:
                bayesian_rating = (v / (v + m)) * R + (m / (v + m)) * C
            else:
                bayesian_rating = C
                
            # Normalize Bayesian Rating to [0, 1] assuming max rating is 5.0
            norm_bayesian = bayesian_rating / max_rating
            
            # 3. Blend scores
            final_score = (self.personalization_weight * norm_pers) + \
                          ((1.0 - self.personalization_weight) * norm_bayesian)
                          
            ranked_items.append(RecommendationItem(item_id=item_id, score=final_score))
            
        # Re-sort descending by the blended score
        ranked_items.sort(key=lambda x: x.score, reverse=True)
        return ranked_items
