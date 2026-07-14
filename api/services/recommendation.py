from typing import List, Optional, Tuple
import random
from sqlalchemy.orm import Session
from api.repositories.user import user_repository
from api.repositories.item import item_repository
from api.repositories.rating import rating_repository
from api.schemas.recommendation import RecommendationResponse, RecommendationItemResponse
from api.schemas.item import ItemResponse
from api.schemas.metrics import SystemEvaluationResponse, AlgorithmMetrics
from api.core.exceptions import UserNotFoundError
from engine.strategy import RecommendationStrategy, UserInteraction
from engine.collaborative import CollaborativeFiltering
from engine.content_based import ContentBasedFiltering
from engine.hybrid import HybridRecommendation
from engine.ranking import WeightedRanker
from engine.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

class RecommendationService:
    """
    Coordinates data retrieval, model fitting, personalized predictions,
    Bayesian reranking, and engine performance evaluation.
    """
    
    def get_strategy(self, algorithm: str) -> RecommendationStrategy:
        """
        Factory method to resolve algorithm string to strategy object.
        """
        alg = algorithm.lower()
        if alg == "user_cf":
            return CollaborativeFiltering(kind="user", similarity_metric="cosine")
        elif alg == "item_cf":
            return CollaborativeFiltering(kind="item", similarity_metric="cosine")
        elif alg == "content_based":
            return ContentBasedFiltering()
        elif alg == "hybrid":
            return HybridRecommendation(cf_weight=0.6, cb_weight=0.4)
        else:
            # Default fallback
            return HybridRecommendation(cf_weight=0.5, cb_weight=0.5)

    def get_recommendations(
        self,
        db: Session,
        user_id: int,
        algorithm: str = "hybrid",
        k: int = 10,
        apply_reranking: bool = True
    ) -> RecommendationResponse:
        # Validate user exists
        user = user_repository.get(db, user_id)
        if not user:
            raise UserNotFoundError(user_id)

        # 1. Fetch raw interactions and item metadata
        interactions = rating_repository.get_all_interactions(db)
        items_metadata = rating_repository.get_all_items_metadata(db)

        # 2. Get and fit recommendation strategy
        strategy = self.get_strategy(algorithm)
        strategy.fit(interactions, items_metadata)

        # 3. Generate raw predictions
        raw_recs = strategy.recommend(user_id, k=k * 2) # Fetch slightly more for reranking

        # 4. Optionally apply Bayesian ranking
        if apply_reranking and raw_recs:
            item_avgs, item_counts, global_mean = rating_repository.get_global_rating_stats(db)
            ranker = WeightedRanker(personalization_weight=0.7)
            ranked_recs = ranker.rank(raw_recs, item_avgs, item_counts, global_mean)
        else:
            ranked_recs = raw_recs

        # Slice to top K
        top_k_recs = ranked_recs[:k]

        # 5. Hydrate items from DB to return item profiles
        item_ids = [r.item_id for r in top_k_recs]
        items_db = item_repository.get_by_ids(db, item_ids)
        items_map = {item.id: item for item in items_db}

        # Format items to schema responses
        recommendations_out = []
        for rec in top_k_recs:
            item_db = items_map.get(rec.item_id)
            item_schema = None
            if item_db:
                item_schema = ItemResponse.model_validate(item_db)
                
            recommendations_out.append(
                RecommendationItemResponse(
                    item_id=rec.item_id,
                    score=rec.score,
                    item=item_schema
                )
            )

        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations_out,
            algorithm_used=algorithm,
            k=k
        )

    def evaluate_system(self, db: Session, k: int = 5, test_ratio: float = 0.2) -> SystemEvaluationResponse:
        """
        Performs offline evaluation of all engine strategies using train-test split on ratings.
        """
        # Fetch all ratings
        interactions = rating_repository.get_all_interactions(db)
        items_metadata = rating_repository.get_all_items_metadata(db)

        if len(interactions) < 20:
            # Not enough data to split
            return SystemEvaluationResponse(k=k, test_split_ratio=test_ratio, metrics=[])

        # 1. Train-test split by interactions
        random.seed(42)
        shuffled = list(interactions)
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - test_ratio))
        train_interactions = shuffled[:split_idx]
        test_interactions = shuffled[split_idx:]

        # Map test set items by user for evaluation check
        # Ground truth: relevant items in test set (ratings >= 3.5 considered relevant)
        ground_truth: dict[int, List[int]] = {}
        for x in test_interactions:
            if x.rating >= 3.5:
                ground_truth.setdefault(x.user_id, []).append(x.item_id)

        # Users who have ground truth relevant items in the test set
        eval_users = list(ground_truth.keys())
        if not eval_users:
            return SystemEvaluationResponse(k=k, test_split_ratio=test_ratio, metrics=[])

        algorithms = ["user_cf", "item_cf", "content_based", "hybrid"]
        metrics_summary = []

        for alg in algorithms:
            # Instantiate strategy
            strategy = self.get_strategy(alg)
            
            # Fit model on training subset
            strategy.fit(train_interactions, items_metadata)

            precisions = []
            recalls = []
            ndcgs = []

            # Evaluate strategy predictions against test set ground truth
            for user_id in eval_users:
                # Generate recommendations
                recs = strategy.recommend(user_id, k=k)
                rec_ids = [r.item_id for r in recs]
                
                relevant = ground_truth[user_id]
                
                precisions.append(precision_at_k(rec_ids, relevant, k))
                recalls.append(recall_at_k(rec_ids, relevant, k))
                ndcgs.append(ndcg_at_k(rec_ids, relevant, k))

            # Store averages
            metrics_summary.append(
                AlgorithmMetrics(
                    algorithm=alg,
                    precision=float(sum(precisions) / len(precisions)) if precisions else 0.0,
                    recall=float(sum(recalls) / len(recalls)) if recalls else 0.0,
                    ndcg=float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0
                )
            )

        return SystemEvaluationResponse(
            k=k,
            test_split_ratio=test_ratio,
            metrics=metrics_summary
        )

recommendation_service = RecommendationService()
