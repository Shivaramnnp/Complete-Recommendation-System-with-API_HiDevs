from typing import Generic, TypeVar, Type, List, Optional, Any
from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from api.models.base import Base

ModelType = TypeVar("ModelType", bound=Base)

class BaseRepository(Generic[ModelType]):
    """
    Generic repository pattern implementing standard CRUD using SQLAlchemy 2.0 standards.
    """
    def __init__(self, model: Type[ModelType]):
        self.model = model

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        statement = select(self.model).where(self.model.id == id)
        return db.execute(statement).scalar_one_or_none()

    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> List[ModelType]:
        statement = select(self.model).offset(skip).limit(limit)
        return list(db.execute(statement).scalars().all())

    def create(self, db: Session, obj_data: dict) -> ModelType:
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, id: Any) -> bool:
        db_obj = self.get(db, id)
        if db_obj:
            db.delete(db_obj)
            db.commit()
            return True
        return False
