import numpy as np
import pandas as pd
import threading
import time
import os
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from sqlalchemy import func
from sqlalchemy.orm import Session
from repositories import user_repository, content_repository, interaction_repository, skill_repository
from models import Interaction, Content, UserSkill, ContentSkill

@dataclass
class RecommendationItem:
    content_id: int
    score: float
    explanation: str

@dataclass
class RecommendationResult:
    user_id: int
    recommendations: List[RecommendationItem]
    algorithm: str
    cached: bool

class SimilarityCalculator:
    """
    Computes mathematical similarity indices between user vectors and content profiles.
    """
    
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculates the Cosine Similarity between two dense numeric vectors.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    @staticmethod
    def jaccard_similarity(s1: Set[Any], s2: Set[Any]) -> float:
        """
        Calculates the Jaccard Similarity between two sets (binary feature overlap).
        """
        if not s1 or not s2:
            return 0.0
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        if union == 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def compute_user_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Computes pairwise Cosine Similarity between users from a User-Item interaction matrix.
        """
        users = user_item_matrix.index
        n_users = len(users)
        sim_matrix = np.zeros((n_users, n_users))
        
        dense_matrix = user_item_matrix.values
        
        for i in range(n_users):
            sim_matrix[i, i] = 1.0
            v_i = dense_matrix[i]
            for j in range(i + 1, n_users):
                v_j = dense_matrix[j]
                
                # Zero out elements if we want centered cosine, or compute raw cosine
                score = SimilarityCalculator.cosine_similarity(v_i, v_j)
                sim_matrix[i, j] = score
                sim_matrix[j, i] = score
                
        return pd.DataFrame(sim_matrix, index=users, columns=users)


class CandidateGenerator:
    """
    Generates retrieval candidate sets from collaborative, content, or popularity filters.
    Filters out items the user has already interacted with.
    """
    
    def __init__(self, user_history_ids: Set[int]):
        self.user_history_ids = user_history_ids

    def generate_collaborative_candidates(
        self,
        user_id: int,
        user_user_sim: pd.DataFrame,
        user_item_matrix: pd.DataFrame,
        top_n_neighbors: int = 5
    ) -> Set[int]:
        """
        Retrieves content IDs rated highly by users with similar profiles.
        """
        if user_id not in user_user_sim.index:
            return set()
            
        # Get top similar users
        similar_users = user_user_sim.loc[user_id].drop(user_id).sort_values(ascending=False)
        top_neighbors = similar_users[similar_users > 0].head(top_n_neighbors).index
        
        candidates = set()
        for neighbor in top_neighbors:
            neighbor_ratings = user_item_matrix.loc[neighbor]
            # Gather items the neighbor liked (rating >= 3.0 or interaction exists)
            liked_items = neighbor_ratings[neighbor_ratings >= 3.0].index
            candidates.update(liked_items)
            
        # Filter out what the user already interacted with
        return candidates.difference(self.user_history_ids)

    def generate_content_candidates(
        self,
        user_skills: List[dict],
        content_skills_map: Dict[int, List[dict]]
    ) -> Set[int]:
        """
        Retrieves content IDs containing skill topics matching the user's skills.
        """
        if not user_skills:
            return set()
            
        user_skill_ids = {us["skill_id"] for us in user_skills}
        candidates = set()
        
        for content_id, skills in content_skills_map.items():
            content_skill_ids = {s["skill_id"] for s in skills}
            # If there is skill overlap, it's a candidate
            if user_skill_ids.intersection(content_skill_ids):
                candidates.add(content_id)
                
        return candidates.difference(self.user_history_ids)

    def generate_popularity_candidates(
        self,
        popularity_scores: List[Tuple[int, int]],
        limit: int = 20
    ) -> List[int]:
        """
        Retrieves top globally popular items as fallback candidates.
        """
        candidates = []
        for content_id, _ in popularity_scores:
            if content_id not in self.user_history_ids:
                candidates.append(content_id)
            if len(candidates) >= limit:
                break
        return candidates


class RecommendationScorer:
    """
    Scores candidates and blends them with analytical rating attributes.
    """
    
    PROFICIENCY_WEIGHTS = {
        "Beginner": 1.0,
        "Intermediate": 2.0,
        "Advanced": 3.0
    }

    @staticmethod
    def score_collaborative(
        user_id: int,
        candidates: Set[int],
        user_user_sim: pd.DataFrame,
        user_item_matrix: pd.DataFrame
    ) -> Dict[int, float]:
        """
        Scores collaborative candidates using user similarity-weighted ratings.
        """
        scores = {}
        if user_id not in user_user_sim.index:
            return {}
            
        user_sims = user_user_sim.loc[user_id]
        
        for content_id in candidates:
            if content_id not in user_item_matrix.columns:
                continue
                
            ratings = user_item_matrix[content_id]
            active_raters = ratings[ratings > 0].index
            
            sim_sum = 0.0
            weighted_sum = 0.0
            
            for rater in active_raters:
                if rater == user_id:
                    continue
                sim = user_sims.get(rater, 0.0)
                if sim > 0:
                    weighted_sum += sim * ratings.loc[rater]
                    sim_sum += sim
                    
            if sim_sum > 0:
                scores[content_id] = float(weighted_sum / sim_sum)
                
        return scores

    @staticmethod
    def score_content(
        user_skills: List[dict],
        candidates: Set[int],
        content_skills_map: Dict[int, List[dict]]
    ) -> Dict[int, float]:
        """
        Scores content candidates based on user proficiency levels and content relevance.
        Formula: Sum_s (UserSkillProficiency(s) * ContentSkillRelevance(s))
        """
        scores = {}
        user_skill_dict = {
            us["skill_id"]: RecommendationScorer.PROFICIENCY_WEIGHTS.get(us["proficiency_level"], 1.0)
            for us in user_skills
        }
        
        for content_id in candidates:
            skills = content_skills_map.get(content_id, [])
            score = 0.0
            for s in skills:
                skill_id = s["skill_id"]
                relevance = s["relevance_score"]
                if skill_id in user_skill_dict:
                    # Score is dot product of user weight and skill relevance
                    score += user_skill_dict[skill_id] * relevance
            if score > 0:
                scores[content_id] = float(score)
                
        return scores

    @staticmethod
    def blend_hybrid_scores(
        cf_scores: Dict[int, float],
        cb_scores: Dict[int, float],
        cf_weight: float = 0.5,
        cb_weight: float = 0.5
    ) -> Dict[int, float]:
        """
        Normalizes scores using Min-Max scaling and blends them linearly.
        """
        def normalize(scores_dict: Dict[int, float]) -> Dict[int, float]:
            if not scores_dict:
                return {}
            vals = list(scores_dict.values())
            min_v, max_v = min(vals), max(vals)
            denom = max_v - min_v
            if denom == 0:
                return {k: 1.0 for k in scores_dict.keys()}
            return {k: (v - min_v) / denom for k, v in scores_dict.items()}

        norm_cf = normalize(cf_scores)
        norm_cb = normalize(cb_scores)
        
        blended = {}
        all_keys = set(norm_cf.keys()).union(set(norm_cb.keys()))
        
        for k in all_keys:
            val_cf = norm_cf.get(k, 0.0)
            val_cb = norm_cb.get(k, 0.0)
            blended[k] = (cf_weight * val_cf) + (cb_weight * val_cb)
            
        return blended

    @staticmethod
    def apply_bayesian_reranking(
        scores: Dict[int, float],
        item_ratings: Dict[int, float],
        item_rating_counts: Dict[int, int],
        global_mean: float,
        personalization_weight: float = 0.7,
        min_ratings: int = 3
    ) -> Dict[int, float]:
        """
        Adjusts recommendation scores by ensembling them with IMDb-style Bayesian ratings.
        """
        if not scores:
            return {}
            
        # Max/min personalization scores for scaling
        vals = list(scores.values())
        min_pers, max_pers = min(vals), max(vals)
        pers_range = max_pers - min_pers
        
        reranked = {}
        max_rating = 5.0
        
        for content_id, p_score in scores.items():
            norm_pers = (p_score - min_pers) / pers_range if pers_range > 0 else 1.0
            
            # Fetch stats
            v = item_rating_counts.get(content_id, 0)
            R = item_ratings.get(content_id, global_mean)
            C = global_mean
            m = min_ratings
            
            # Compute Bayesian average
            bayesian = (v / (v + m)) * R + (m / (v + m)) * C if (v + m) > 0 else C
            norm_bayesian = bayesian / max_rating
            
            # Blend
            reranked[content_id] = (personalization_weight * norm_pers) + \
                                  ((1.0 - personalization_weight) * norm_bayesian)
                                  
        return reranked


class RecommendationEvaluator:
    """
    Evaluates recommendation quality indices using offline verification splits.
    """
    
    @staticmethod
    def precision_at_k(recommended_ids: List[int], test_ids: List[int], k: int) -> float:
        if k <= 0:
            return 0.0
        top_k = recommended_ids[:k]
        hits = sum(1 for item in top_k if item in test_ids)
        return float(hits / k)

    @staticmethod
    def recall_at_k(recommended_ids: List[int], test_ids: List[int], k: int) -> float:
        if not test_ids or k <= 0:
            return 0.0
        top_k = recommended_ids[:k]
        hits = sum(1 for item in top_k if item in test_ids)
        return float(hits / len(test_ids))

    @staticmethod
    def ndcg_at_k(recommended_ids: List[int], test_ids: List[int], k: int) -> float:
        if not test_ids or k <= 0:
            return 0.0
        top_k = recommended_ids[:k]
        test_set = set(test_ids)
        
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in test_set:
                dcg += 1.0 / np.log2(i + 2)
                
        idcg = 0.0
        n_relevant = min(len(test_ids), k)
        for i in range(n_relevant):
            idcg += 1.0 / np.log2(i + 2)
            
        if idcg == 0:
            return 0.0
        return float(dcg / idcg)


class RecommendationOrchestrator:
    """
    Orchestrator managing data ingestion, strategy execution, in-memory caching,
    and feedback routing.
    """
    
    def __init__(self, cache_ttl_seconds: Optional[int] = None):
        self.cache_ttl_seconds = cache_ttl_seconds if cache_ttl_seconds is not None else int(os.getenv("CACHE_TTL_SECONDS", 300))
        self._cache: Dict[Tuple[int, str, int, bool], Tuple[float, List[RecommendationItem]]] = {}
        self._lock = threading.Lock()

    def invalidate_user_cache(self, user_id: int) -> None:
        """
        Invalidates cached recommendations for a specific user.
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k[0] == user_id]
            for k in keys_to_remove:
                self._cache.pop(k, None)

    def record_feedback(
        self,
        db: Session,
        user_id: int,
        content_id: int,
        type: str,
        rating: Optional[float] = None
    ) -> Interaction:
        """
        Records a user interaction feedback in the database and invalidates the user's recommendation cache.
        """
        # Save interaction
        interaction = interaction_repository.record_interaction(db, user_id, content_id, type, rating)
        # Clear cache for this user since their profile just changed
        self.invalidate_user_cache(user_id)
        return interaction

    def orchestrate(
        self,
        db: Session,
        user_id: int,
        algorithm: str = "hybrid",
        k: int = 10,
        apply_reranking: bool = True
    ) -> RecommendationResult:
        """
        Orchestrates recommendations fetching, candidates generation, scoring, and explanation hooks.
        """
        # Check cache
        cache_key = (user_id, algorithm.lower(), k, apply_reranking)
        with self._lock:
            if cache_key in self._cache:
                timestamp, cached_items = self._cache[cache_key]
                if time.time() - timestamp < self.cache_ttl_seconds:
                    return RecommendationResult(user_id=user_id, recommendations=cached_items, algorithm=algorithm, cached=True)
                else:
                    self._cache.pop(cache_key, None)

        # 1. Load data from Repositories
        user_interactions = user_repository.get_user_history(db, user_id)
        all_interactions = interaction_repository.get_multi(db, limit=2000)
        
        user_skills_obj = db.query(UserSkill).filter(UserSkill.user_id == user_id).all()
        user_skills = [{"skill_id": us.skill_id, "proficiency_level": us.proficiency_level} for us in user_skills_obj]
        
        content_skills_obj = db.query(ContentSkill).all()
        content_skills_map = {}
        for cs in content_skills_obj:
            content_skills_map.setdefault(cs.content_id, []).append({
                "skill_id": cs.skill_id,
                "relevance_score": cs.relevance_score
            })
            
        all_skills = skill_repository.get_multi(db, limit=1000)
        skills_name_map = {s.id: s.name for s in all_skills}

        user_history_ids = {x.content_id for x in user_interactions}
        c_generator = CandidateGenerator(user_history_ids)

        # 2. Cold Start Detection
        # A user is cold-start if they have under 3 interactions AND no registered skills.
        is_cold_start = (len(user_interactions) < 3) and (not user_skills)
        
        scored_items: Dict[int, float] = {}
        explanations: Dict[int, str] = {}
        alg_used = algorithm.lower()

        if is_cold_start:
            # Fallback Route: Popularity recommendations
            alg_used = "popularity"
            popularity_stats = [(c.id, count) for c, count in content_repository.get_popularity_scores(db, limit=50)]
            pop_candidates = c_generator.generate_popularity_candidates(popularity_stats, limit=k)
            
            # Simple score matching order rank
            for idx, c_id in enumerate(pop_candidates):
                scored_items[c_id] = float(1.0 - (idx / len(pop_candidates)))
                explanations[c_id] = "Recommended because this is currently one of our top popular courses."
        else:
            # 3. Model Logic Implementation
            # Load collaborative matrix structures if needed
            cf_scores: Dict[int, float] = {}
            cb_scores: Dict[int, float] = {}
            
            if alg_used in ["collaborative", "hybrid"]:
                # Build User-Item Matrix
                df_data = [{"user_id": x.user_id, "content_id": x.content_id, "rating": x.rating or 3.0} for x in all_interactions]
                if df_data:
                    df = pd.DataFrame(df_data)
                    user_item_matrix = df.pivot_table(index="user_id", columns="content_id", values="rating", aggfunc="max").fillna(0.0)
                    
                    # Ensure active user is in the matrix
                    if user_id not in user_item_matrix.index:
                        user_item_matrix.loc[user_id] = 0.0
                        
                    user_user_sim = SimilarityCalculator.compute_user_user_similarity(user_item_matrix)
                    
                    # Generate collaborative candidates
                    cf_candidates = c_generator.generate_collaborative_candidates(user_id, user_user_sim, user_item_matrix)
                    cf_scores = RecommendationScorer.score_collaborative(user_id, cf_candidates, user_user_sim, user_item_matrix)
                    
                    for c_id in cf_scores.keys():
                        explanations[c_id] = "Recommended because other users with similar learning profiles rated this highly."
                        
            if alg_used in ["content_based", "hybrid"]:
                # Generate content-based candidates
                cb_candidates = c_generator.generate_content_candidates(user_skills, content_skills_map)
                cb_scores = RecommendationScorer.score_content(user_skills, cb_candidates, content_skills_map)
                
                for c_id in cb_scores.keys():
                    # Map to the matching skills user is learning
                    matching_skills = []
                    item_skills = {s["skill_id"] for s in content_skills_map.get(c_id, [])}
                    for us in user_skills:
                        if us["skill_id"] in item_skills:
                            matching_skills.append(skills_name_map.get(us["skill_id"], "skills"))
                    skill_str = ", ".join(matching_skills) if matching_skills else "topics in your profile"
                    explanations[c_id] = f"Recommended because this course covers {skill_str}."

            if alg_used == "collaborative":
                scored_items = cf_scores
            elif alg_used == "content_based":
                scored_items = cb_scores
            elif alg_used == "hybrid":
                scored_items = RecommendationScorer.blend_hybrid_scores(cf_scores, cb_scores, cf_weight=0.5, cb_weight=0.5)
                # Blend explanations
                for c_id in scored_items.keys():
                    if c_id in cb_scores and c_id in cf_scores:
                        explanations[c_id] = explanations.get(c_id, "") + " It also matches similar users."
                popularity_stats = [(c.id, count) for c, count in content_repository.get_popularity_scores(db, limit=50)]
                pop_candidates = c_generator.generate_popularity_candidates(popularity_stats, limit=k)
                for idx, c_id in enumerate(pop_candidates):
                    scored_items[c_id] = float(1.0 - (idx / len(pop_candidates)))
                    explanations[c_id] = "Recommended because this is currently one of our top popular courses."

        # 4. Apply Bayesian Reranking
        if apply_reranking and scored_items and alg_used != "popularity":
            item_avgs, item_counts, global_mean = rating_repository_stats(db)
            scored_items = RecommendationScorer.apply_bayesian_reranking(
                scored_items, item_avgs, item_counts, global_mean
            )

        # 5. Rank and Slice to K
        sorted_recs = sorted(scored_items.items(), key=lambda x: x[1], reverse=True)[:k]
        
        recs_out = []
        for c_id, score in sorted_recs:
            recs_out.append(
                RecommendationItem(
                    content_id=c_id,
                    score=float(score),
                    explanation=explanations.get(c_id, "Recommended for your learning path.")
                )
            )
            
        # Store in cache
        with self._lock:
            self._cache[cache_key] = (time.time(), recs_out)
            
        return RecommendationResult(user_id=user_id, recommendations=recs_out, algorithm=alg_used, cached=False)


def rating_repository_stats(db: Session) -> Tuple[Dict[int, float], Dict[int, int], float]:
    """
    Utility fetching interaction rating statistics for Bayesian reranking.
    """
    mean_val = db.query(func.avg(Interaction.rating)).filter(Interaction.rating.isnot(None)).scalar() or 3.0
    
    stats = db.query(
        Interaction.content_id,
        func.avg(Interaction.rating).label("avg_rating"),
        func.count(Interaction.id).label("count_rating")
    ).filter(Interaction.rating.isnot(None)).group_by(Interaction.content_id).all()
    
    item_ratings = {}
    item_rating_counts = {}
    
    for row in stats:
        item_ratings[row.content_id] = float(row.avg_rating)
        item_rating_counts[row.content_id] = int(row.count_rating)
        
    return item_ratings, item_rating_counts, float(mean_val)

# Global orchestrator instance
recommendation_orchestrator = RecommendationOrchestrator()
