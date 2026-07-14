from pydantic import BaseModel, Field
from typing import Optional, List
from api.schemas.item import ItemResponse

class RatingCreate(BaseModel):
    item_id: int = Field(..., description="ID of the item being rated")
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating value between 0.5 and 5.0")
    timestamp: Optional[int] = Field(None, description="Unix timestamp of rating")

class RatingResponse(BaseModel):
    id: int
    user_id: int
    item_id: int
    rating: float
    timestamp: Optional[int]

    class Config:
        from_attributes = True

class RecommendationItemResponse(BaseModel):
    item_id: int
    score: float
    item: Optional[ItemResponse] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItemResponse]
    algorithm_used: str
    k: int
