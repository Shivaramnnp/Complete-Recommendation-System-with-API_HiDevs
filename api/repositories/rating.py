from typing import Optional, List, Dict, Tuple
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from api.models.rating import Rating
from api.models.item import Item
from api.repositories.base import BaseRepository
from engine.strategy import UserInteraction

class RatingRepository(BaseRepository[Rating]):
    def __init__(self):
        super().__init__(Rating)
        
    def get_by_user_and_item(self, db: Session, user_id: int, item_id: int) -> Optional[Rating]:
        statement = select(Rating).where(
            Rating.user_id == user_id,
            Rating.item_id == item_id
        )
        return db.execute(statement).scalar_one_or_none()
        
    def get_all_interactions(self, db: Session) -> List[UserInteraction]:
        """
        Retrieves all ratings formatted for feeding into the recommendation engine strategies.
        """
        statement = select(Rating)
        ratings = db.execute(statement).scalars().all()
        return [
            UserInteraction(
                user_id=r.user_id,
                item_id=r.item_id,
                rating=r.rating,
                timestamp=r.timestamp
            )
            for r in ratings
        ]
        
    def get_all_items_metadata(self, db: Session) -> List[dict]:
        """
        Retrieves item features and metadata formatted for the content-based engine.
        """
        statement = select(Item)
        items = db.execute(statement).scalars().all()
        return [
            {
                "item_id": item.id,
                "genres": item.genres or "",
                "description": item.description or ""
            }
            for item in items
        ]
        
    def get_global_rating_stats(self, db: Session) -> Tuple[Dict[int, float], Dict[int, int], float]:
        """
        Computes item-level stats and overall global mean rating for Bayesian ranking.
        Returns: (item_avg_ratings, item_rating_counts, global_mean_rating)
        """
        # 1. Overall mean
        mean_statement = select(func.avg(Rating.rating))
        global_mean = db.execute(mean_statement).scalar() or 0.0
        
        # 2. Group by item stats
        stats_statement = select(
            Rating.item_id,
            func.avg(Rating.rating).label("avg_rating"),
            func.count(Rating.rating).label("rating_count")
        ).group_by(Rating.item_id)
        
        results = db.execute(stats_statement).all()
        
        item_avg_ratings = {}
        item_rating_counts = {}
        
        for row in results:
            item_avg_ratings[row.item_id] = float(row.avg_rating)
            item_rating_counts[row.item_id] = int(row.rating_count)
            
        return item_avg_ratings, item_rating_counts, float(global_mean)

rating_repository = RatingRepository()
