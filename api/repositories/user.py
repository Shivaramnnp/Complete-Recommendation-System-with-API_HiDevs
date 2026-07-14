from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from api.models.user import User
from api.repositories.base import BaseRepository

class UserRepository(BaseRepository[User]):
    def __init__(self):
        super().__init__(User)
        
    def get_by_username(self, db: Session, username: str) -> Optional[User]:
        statement = select(User).where(User.username == username)
        return db.execute(statement).scalar_one_or_none()

user_repository = UserRepository()
