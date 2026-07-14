import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from engine.strategy import RecommendationStrategy, UserInteraction, RecommendationItem
from engine.similarity import cosine_similarity, pearson_correlation

class CollaborativeFiltering(RecommendationStrategy):
    """
    Collaborative Filtering recommendation strategy.
    Supports User-based and Item-based algorithms.
    """
    
    def __init__(self, kind: str = "user", similarity_metric: str = "cosine", min_common: int = 1):
        """
        :param kind: 'user' for User-based CF, 'item' for Item-based CF
        :param similarity_metric: 'cosine' or 'pearson'
        :param min_common: Minimum number of common items/users needed to calculate similarity
        """
        self.kind = kind.lower()
        self.similarity_metric = similarity_metric.lower()
        self.min_common = min_common
        
        # Internal state
        self.interactions_df: Optional[pd.DataFrame] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[pd.DataFrame] = None
        self.mean_ratings: Dict[int, float] = {}
        self.popular_items: List[int] = []  # Fallback for cold start
        
    def fit(self, interactions: List[UserInteraction], items_info: Optional[List[dict]] = None) -> None:
        if not interactions:
            return
            
        # Convert interactions to DataFrame
        data = [
            {"user_id": x.user_id, "item_id": x.item_id, "rating": x.rating}
            for x in interactions
        ]
        self.interactions_df = pd.DataFrame(data)
        
        # Calculate fallback: popular items (by average rating and count of interactions)
        item_stats = self.interactions_df.groupby("item_id").agg(
            mean_rating=("rating", "mean"),
            count=("rating", "count")
        )
        # Sort by count desc, then mean rating desc
        self.popular_items = item_stats.sort_values(by=["count", "mean_rating"], ascending=False).index.tolist()
        
        # Build User-Item matrix (users as rows, items as columns)
        self.user_item_matrix = self.interactions_df.pivot(
            index="user_id", columns="item_id", values="rating"
        ).fillna(0.0)
        
        # Compute mean ratings per user for centering (useful for Pearson / adjusted cosine)
        for user_id in self.user_item_matrix.index:
            ratings = self.user_item_matrix.loc[user_id]
            non_zero = ratings[ratings > 0]
            self.mean_ratings[user_id] = float(non_zero.mean()) if len(non_zero) > 0 else 0.0
            
        # Compute Similarity Matrix
        self._compute_similarity()
        
    def _compute_similarity(self) -> None:
        if self.user_item_matrix is None:
            return
            
        matrix = self.user_item_matrix
        if self.kind == "item":
            # For Item-based CF, we compute item-item similarities (items as rows)
            matrix = matrix.T
            
        index_names = matrix.index
        n_entities = len(index_names)
        sim_data = np.zeros((n_entities, n_entities))
        
        # Select similarity function
        sim_func = cosine_similarity
        if self.similarity_metric == "pearson":
            sim_func = pearson_correlation
            
        # Compute pairwise similarities
        # Optimize with vector operations where possible, but a standard double loop is reliable and explicit
        dense_matrix = matrix.values
        for i in range(n_entities):
            sim_data[i, i] = 1.0
            v_i = dense_matrix[i]
            for j in range(i + 1, n_entities):
                v_j = dense_matrix[j]
                
                # Check min common interactions
                common = np.logical_and(v_i > 0, v_j > 0)
                if np.sum(common) >= self.min_common:
                    score = sim_func(v_i, v_j)
                else:
                    score = 0.0
                    
                sim_data[i, j] = score
                sim_data[j, i] = score
                
        self.similarity_matrix = pd.DataFrame(sim_data, index=index_names, columns=index_names)
        
    def recommend(self, user_id: int, k: int = 10) -> List[RecommendationItem]:
        # Handle Cold Start (user not seen during fit)
        if (self.user_item_matrix is None or 
            user_id not in self.user_item_matrix.index):
            return [RecommendationItem(item_id=iid, score=0.0) for iid in self.popular_items[:k]]
            
        if self.kind == "user":
            return self._recommend_user_based(user_id, k)
        else:
            return self._recommend_item_based(user_id, k)
            
    def _recommend_user_based(self, user_id: int, k: int) -> List[RecommendationItem]:
        # 1. Get user ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        # 2. Get user similarities
        user_sims = self.similarity_matrix.loc[user_id]
        
        # Top similar users (excluding self, sorted descending)
        similar_users = user_sims.drop(user_id).sort_values(ascending=False)
        similar_users = similar_users[similar_users > 0]  # Only consider positive similarity
        
        predictions = []
        user_mean = self.mean_ratings.get(user_id, 0.0)
        
        # 3. Predict for each unrated item
        for item_id in unrated_items:
            # Users who rated this item
            other_ratings = self.user_item_matrix[item_id]
            active_raters = other_ratings[other_ratings > 0].index
            
            # Intersection of similar users and active raters
            overlapping_users = similar_users.index.intersection(active_raters)
            
            if len(overlapping_users) == 0:
                continue
                
            sim_sum = 0.0
            weighted_sum = 0.0
            
            for other_user in overlapping_users:
                sim = similar_users.loc[other_user]
                rating = other_ratings.loc[other_user]
                other_mean = self.mean_ratings.get(other_user, 0.0)
                
                # Pearson centered prediction: P(u,i) = mean(u) + sum(sim(u,v) * (r_v,i - mean(v))) / sum(sim(u,v))
                if self.similarity_metric == "pearson":
                    weighted_sum += sim * (rating - other_mean)
                else:
                    weighted_sum += sim * rating
                sim_sum += abs(sim)
                
            if sim_sum > 0:
                if self.similarity_metric == "pearson":
                    pred = user_mean + (weighted_sum / sim_sum)
                else:
                    pred = weighted_sum / sim_sum
                predictions.append(RecommendationItem(item_id=int(item_id), score=float(pred)))
                
        # Sort predictions descending
        predictions.sort(key=lambda x: x.score, reverse=True)
        
        # If we have fewer than k predictions, fill with popular items
        if len(predictions) < k:
            rated_items_set = set(user_ratings[user_ratings > 0].index)
            pred_items_set = {p.item_id for p in predictions}
            
            for pop_item in self.popular_items:
                if pop_item not in rated_items_set and pop_item not in pred_items_set:
                    predictions.append(RecommendationItem(item_id=int(pop_item), score=0.0))
                if len(predictions) >= k:
                    break
                    
        return predictions[:k]
        
    def _recommend_item_based(self, user_id: int, k: int) -> List[RecommendationItem]:
        # 1. Get user ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        unrated_items = user_ratings[user_ratings == 0].index
        
        if rated_items.empty:
            # Fallback to popular items if user has rated nothing
            return [RecommendationItem(item_id=iid, score=0.0) for iid in self.popular_items[:k]]
            
        predictions = []
        
        # 2. Predict score for each unrated item based on its similarity to items user rated
        for target_item in unrated_items:
            if target_item not in self.similarity_matrix.index:
                continue
                
            # Similarities of target item to all other items
            item_sims = self.similarity_matrix.loc[target_item]
            
            weighted_sum = 0.0
            sim_sum = 0.0
            
            for rated_item, rating in rated_items.items():
                if rated_item not in item_sims.index:
                    continue
                sim = item_sims.loc[rated_item]
                if sim <= 0:
                    continue
                    
                weighted_sum += sim * rating
                sim_sum += sim
                
            if sim_sum > 0:
                pred = weighted_sum / sim_sum
                predictions.append(RecommendationItem(item_id=int(target_item), score=float(pred)))
                
        # Sort descending
        predictions.sort(key=lambda x: x.score, reverse=True)
        
        # Fill missing up to K with popular items
        if len(predictions) < k:
            rated_items_set = set(rated_items.index)
            pred_items_set = {p.item_id for p in predictions}
            
            for pop_item in self.popular_items:
                if pop_item not in rated_items_set and pop_item not in pred_items_set:
                    predictions.append(RecommendationItem(item_id=int(pop_item), score=0.0))
                if len(predictions) >= k:
                    break
                    
        return predictions[:k]
