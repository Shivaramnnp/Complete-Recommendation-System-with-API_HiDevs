from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from enum import Enum
import time
from api.core.database import get_db
from api.core.exceptions import DuplicateRatingError, UserNotFoundError, ItemNotFoundError
from api.schemas.recommendation import RatingCreate, RatingResponse, RecommendationResponse
from api.schemas.metrics import SystemEvaluationResponse
from api.repositories.rating import rating_repository
from api.repositories.user import user_repository
from api.repositories.item import item_repository
from api.services.recommendation import recommendation_service

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

class AlgorithmName(str, Enum):
    user_cf = "user_cf"
    item_cf = "item_cf"
    content_based = "content_based"
    hybrid = "hybrid"

@router.post("/ratings", response_model=RatingResponse, status_code=status.HTTP_201_CREATED)
def add_rating(user_id: int, rating_in: RatingCreate, db: Session = Depends(get_db)):
    """
    Submit a user rating for an item. The rating must be between 0.5 and 5.0.
    Updates the system's collaborative filtering interaction matrix dynamically.
    """
    # Verify user exists
    user = user_repository.get(db, user_id)
    if not user:
        raise UserNotFoundError(user_id)
        
    # Verify item exists
    item = item_repository.get(db, rating_in.item_id)
    if not item:
        raise ItemNotFoundError(rating_in.item_id)
        
    # Check for duplicate
    existing = rating_repository.get_by_user_and_item(db, user_id, rating_in.item_id)
    if existing:
        raise DuplicateRatingError(user_id, rating_in.item_id)
        
    rating_data = rating_in.model_dump()
    rating_data["user_id"] = user_id
    if not rating_data.get("timestamp"):
        rating_data["timestamp"] = int(time.time())
        
    return rating_repository.create(db, obj_data=rating_data)

@router.get("/users/{user_id}", response_model=RecommendationResponse)
def get_personalized_recommendations(
    user_id: int,
    algorithm: AlgorithmName = AlgorithmName.hybrid,
    k: int = Query(default=10, ge=1, le=100),
    apply_reranking: bool = Query(default=True, description="Apply Bayesian reranking blended with item popularity"),
    db: Session = Depends(get_db)
):
    """
    Retrieve top-K personalized recommendations for a user.
    Supported algorithms: user_cf, item_cf, content_based, hybrid.
    Optionally applies Bayesian scaling to rank items higher if they are popular and highly rated globally.
    """
    try:
        return recommendation_service.get_recommendations(
            db=db,
            user_id=user_id,
            algorithm=algorithm.value,
            k=k,
            apply_reranking=apply_reranking
        )
    except UserNotFoundError as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating recommendations: {str(e)}"
        )

@router.get("/evaluate", response_model=SystemEvaluationResponse)
def evaluate_recommendation_engines(
    k: int = Query(default=5, ge=1, le=20),
    test_ratio: float = Query(default=0.2, ge=0.05, le=0.5, description="Ratio of ratings to hold out for evaluation"),
    db: Session = Depends(get_db)
):
    """
    Trigger offline accuracy evaluations for all recommendation algorithms (Precision@K, Recall@K, NDCG@K).
    Splits database ratings randomly into train and test sets to compute performance scores.
    """
    return recommendation_service.evaluate_system(db, k=k, test_ratio=test_ratio)
