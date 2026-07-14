import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Set
from engine.strategy import RecommendationStrategy, UserInteraction, RecommendationItem
from engine.similarity import cosine_similarity

class ContentBasedFiltering(RecommendationStrategy):
    """
    Content-Based Filtering recommendation strategy.
    Builds item profile vectors from attributes (e.g. genres, tags)
    and represents user preferences as a weighted average of rated item vectors.
    """
    
    def __init__(self, feature_field: str = "genres", sep: str = "|"):
        """
        :param feature_field: Name of the metadata field to build item profiles from.
        :param sep: Separator for multi-value categorical tags in that field.
        """
        self.feature_field = feature_field
        self.sep = sep
        
        # Internal state
        self.item_features_df: Optional[pd.DataFrame] = None
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.interactions_df: Optional[pd.DataFrame] = None
        self.popular_items: List[int] = []
        self.vocabulary: List[str] = []
        
    def fit(self, interactions: List[UserInteraction], items_info: Optional[List[dict]] = None) -> None:
        if not interactions or not items_info:
            return
            
        # Store interactions
        data = [
            {"user_id": x.user_id, "item_id": x.item_id, "rating": x.rating}
            for x in interactions
        ]
        self.interactions_df = pd.DataFrame(data)
        
        # Calculate popularity fallback
        item_stats = self.interactions_df.groupby("item_id").agg(
            mean_rating=("rating", "mean"),
            count=("rating", "count")
        )
        self.popular_items = item_stats.sort_values(by=["count", "mean_rating"], ascending=False).index.tolist()
        
        # Build item profile vectors
        self._build_item_profiles(items_info)
        
        # Build user profile vectors
        self._build_user_profiles()
        
    def _build_item_profiles(self, items_info: List[dict]) -> None:
        # 1. Extract all unique terms / tags
        all_terms: Set[str] = set()
        item_raw_features: Dict[int, List[str]] = {}
        
        for item in items_info:
            item_id = item.get("item_id")
            if item_id is None:
                continue
                
            val = item.get(self.feature_field, "")
            if not val:
                terms = []
            elif isinstance(val, list):
                terms = [str(x).strip().lower() for x in val]
            else:
                terms = [x.strip().lower() for x in str(val).split(self.sep)]
                
            item_raw_features[item_id] = terms
            all_terms.update(terms)
            
        self.vocabulary = sorted(list(all_terms))
        vocab_to_idx = {term: idx for idx, term in enumerate(self.vocabulary)}
        
        # 2. Build dense vectors
        n_items = len(items_info)
        n_features = len(self.vocabulary)
        
        if n_features == 0:
            # Fallback if no features present
            self.item_features_df = pd.DataFrame()
            return
            
        feature_matrix = np.zeros((n_items, n_features))
        item_ids = []
        
        for i, item in enumerate(items_info):
            item_id = item["item_id"]
            item_ids.append(item_id)
            terms = item_raw_features.get(item_id, [])
            
            # Simple TF-IDF or binary scoring
            # Let's compute term frequency (TF) and inverse document frequency (IDF) manually!
            # Since terms are usually tags (like genres), TF is binary or counts, and IDF scales down common tags.
            # To be robust, let's start with binary term presence
            for t in terms:
                if t in vocab_to_idx:
                    feature_matrix[i, vocab_to_idx[t]] = 1.0
                    
        # Apply TF-IDF scaling:
        # IDF = log(N / (df + 1))
        df = np.sum(feature_matrix > 0, axis=0)
        idf = np.log((n_items + 1) / (df + 1)) + 1.0
        feature_matrix = feature_matrix * idf
        
        # L2 Normalize item vectors for easier cosine similarity
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        feature_matrix = feature_matrix / norms
        
        self.item_features_df = pd.DataFrame(feature_matrix, index=item_ids)
        
    def _build_user_profiles(self) -> None:
        if self.interactions_df is None or self.item_features_df is None or self.item_features_df.empty:
            return
            
        # Group interactions by user
        grouped = self.interactions_df.groupby("user_id")
        
        for user_id, group in grouped:
            user_vectors = []
            weights = []
            
            for _, row in group.iterrows():
                item_id = int(row["item_id"])
                rating = float(row["rating"])
                
                if item_id in self.item_features_df.index:
                    vector = self.item_features_df.loc[item_id].values
                    # Center the rating to separate positive and negative preferences
                    # Ratings >= 3.0 have positive weight, < 3.0 have negative weight
                    weight = rating - 2.5
                    user_vectors.append(vector)
                    weights.append(weight)
                    
            if user_vectors:
                # Weighted average of rated item vectors
                user_vector = np.average(user_vectors, axis=0, weights=weights)
                # Normalize user vector
                norm = np.linalg.norm(user_vector)
                if norm > 0:
                    user_vector = user_vector / norm
                self.user_profiles[user_id] = user_vector
                
    def recommend(self, user_id: int, k: int = 10) -> List[RecommendationItem]:
        # Cold start checks
        if (self.item_features_df is None or self.item_features_df.empty or 
            self.interactions_df is None):
            return [RecommendationItem(item_id=iid, score=0.0) for iid in self.popular_items[:k]]
            
        # If user is not known, fallback to popularity
        if user_id not in self.user_profiles:
            return [RecommendationItem(item_id=iid, score=0.0) for iid in self.popular_items[:k]]
            
        user_vector = self.user_profiles[user_id]
        
        # Get rated items to filter out
        rated_items = set(self.interactions_df[self.interactions_df["user_id"] == user_id]["item_id"])
        
        predictions = []
        
        # Calculate similarity with all items not yet rated
        for item_id in self.item_features_df.index:
            if item_id in rated_items:
                continue
                
            item_vector = self.item_features_df.loc[item_id].values
            
            # Since vectors are normalized, cosine is just the dot product
            score = float(np.dot(user_vector, item_vector))
            predictions.append(RecommendationItem(item_id=int(item_id), score=score))
            
        # Sort predictions
        predictions.sort(key=lambda x: x.score, reverse=True)
        
        # Fill missing with popular
        if len(predictions) < k:
            pred_items_set = {p.item_id for p in predictions}
            for pop_item in self.popular_items:
                if pop_item not in rated_items and pop_item not in pred_items_set:
                    predictions.append(RecommendationItem(item_id=int(pop_item), score=0.0))
                if len(predictions) >= k:
                    break
                    
        return predictions[:k]
