from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class UserInteraction:
    user_id: int
    item_id: int
    rating: float
    timestamp: Optional[int] = None

@dataclass
class RecommendationItem:
    item_id: int
    score: float

class RecommendationStrategy(ABC):
    """
    Abstract interface for all recommendation algorithms.
    """
    
    @abstractmethod
    def fit(self, interactions: List[UserInteraction], items_info: Optional[List[dict]] = None) -> None:
        """
        Train/fit the model with user interactions and metadata.
        """
        pass
        
    @abstractmethod
    def recommend(self, user_id: int, k: int = 10) -> List[RecommendationItem]:
        """
        Generate top-K recommendations for a specific user.
        """
        pass
