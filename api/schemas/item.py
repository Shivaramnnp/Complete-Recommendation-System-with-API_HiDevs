from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ItemBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Item title")
    genres: Optional[str] = Field(None, description="Pipe-separated category strings (e.g. 'Action|Thriller')")
    description: Optional[str] = Field(None, description="Detailed item summary")

class ItemCreate(ItemBase):
    pass

class ItemResponse(ItemBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
