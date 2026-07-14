from typing import List
from sqlalchemy import select
from sqlalchemy.orm import Session
from api.models.item import Item
from api.repositories.base import BaseRepository

class ItemRepository(BaseRepository[Item]):
    def __init__(self):
        super().__init__(Item)
        
    def get_by_ids(self, db: Session, ids: List[int]) -> List[Item]:
        if not ids:
            return []
        statement = select(Item).where(Item.id.in_(ids))
        return list(db.execute(statement).scalars().all())

item_repository = ItemRepository()
