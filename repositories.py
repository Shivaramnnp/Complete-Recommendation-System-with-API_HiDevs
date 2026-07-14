from typing import Generic, TypeVar, Type, List, Optional, Any, Tuple
from datetime import datetime
from sqlalchemy import select, func, or_
from sqlalchemy.orm import Session
from models import Base, User, Content, Skill, UserSkill, ContentSkill, Interaction

ModelType = TypeVar("ModelType", bound=Base)

class BaseRepository(Generic[ModelType]):
    """
    Generic repository pattern providing standard CRUD operations using SQLAlchemy 2.0.
    """
    def __init__(self, model: Type[ModelType]):
        self.model = model

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """
        Retrieve a single record by ID.
        """
        statement = select(self.model).where(self.model.id == id)
        return db.execute(statement).scalar_one_or_none()

    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """
        Retrieve multiple records with pagination.
        """
        statement = select(self.model).offset(skip).limit(limit)
        return list(db.execute(statement).scalars().all())

    def create(self, db: Session, obj_data: dict) -> ModelType:
        """
        Create and persist a new record.
        """
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, db_obj: ModelType, obj_data: dict) -> ModelType:
        """
        Update an existing record with new attributes.
        """
        for field, value in obj_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, id: Any) -> bool:
        """
        Delete a record by ID. Returns True if deleted, False if not found.
        """
        db_obj = self.get(db, id)
        if db_obj:
            db.delete(db_obj)
            db.commit()
            return True
        return False

class UserRepository(BaseRepository[User]):
    """
    Repository for User management and interaction retrieval.
    """
    def __init__(self):
        super().__init__(User)

    def get_by_username(self, db: Session, username: str) -> Optional[User]:
        """
        Retrieve a user record matching the username.
        """
        statement = select(User).where(User.username == username)
        return db.execute(statement).scalar_one_or_none()

    def get_user_history(self, db: Session, user_id: int) -> List[Interaction]:
        """
        Retrieve the chronological interaction history of a user.
        """
        statement = select(Interaction).where(Interaction.user_id == user_id).order_by(Interaction.created_at.desc())
        return list(db.execute(statement).scalars().all())

    def add_user_skill(self, db: Session, user_id: int, skill_id: int, proficiency_level: str = "Beginner") -> UserSkill:
        """
        Maps a skill to a user, updating the proficiency_level if the relationship already exists.
        """
        statement = select(UserSkill).where(UserSkill.user_id == user_id, UserSkill.skill_id == skill_id)
        existing = db.execute(statement).scalar_one_or_none()
        
        if existing:
            existing.proficiency_level = proficiency_level
            db.commit()
            db.refresh(existing)
            return existing
        else:
            user_skill = UserSkill(
                user_id=user_id,
                skill_id=skill_id,
                proficiency_level=proficiency_level,
                created_at=datetime.utcnow()
            )
            db.add(user_skill)
            db.commit()
            db.refresh(user_skill)
            return user_skill

class ContentRepository(BaseRepository[Content]):
    """
    Repository for Content catalog management, catalog searching, and analytical recommendation fallbacks.
    """
    def __init__(self):
        super().__init__(Content)

    def get_content_by_skill(self, db: Session, skill_id: int) -> List[Content]:
        """
        Retrieves all content mapped to a specific skill.
        """
        statement = select(Content).join(ContentSkill).where(ContentSkill.skill_id == skill_id)
        return list(db.execute(statement).scalars().all())

    def search_content(self, db: Session, query_str: str) -> List[Content]:
        """
        Searches the catalog by performing a case-insensitive query match on title and description.
        """
        statement = select(Content).where(
            or_(
                Content.title.ilike(f"%{query_str}%"),
                Content.description.ilike(f"%{query_str}%")
            )
        )
        return list(db.execute(statement).scalars().all())

    def get_popularity_scores(self, db: Session, limit: int = 10) -> List[Tuple[Content, int]]:
        """
        Retrieves Content items ordered by total interaction counts.
        Returns a list of tuples containing the Content model and its total interaction count.
        """
        statement = select(
            Content,
            func.count(Interaction.id).label("interaction_count")
        ).outerjoin(
            Interaction, Content.id == Interaction.content_id
        ).group_by(
            Content.id
        ).order_by(
            func.count(Interaction.id).desc()
        ).limit(limit)
        
        results = db.execute(statement).all()
        return [(row[0], int(row[1])) for row in results]

    def get_cold_start_content(self, db: Session, limit: int = 10) -> List[Content]:
        """
        Retrieves fallback content for cold start scenarios (new users).
        Sorts items by average rating descending (nulls last) and then by creation recency.
        """
        statement = select(Content).outerjoin(
            Interaction, Content.id == Interaction.content_id
        ).group_by(
            Content.id
        ).order_by(
            func.avg(Interaction.rating).desc().nullslast(),
            Content.created_at.desc()
        ).limit(limit)
        
        return list(db.execute(statement).scalars().all())

    def map_content_skill(self, db: Session, content_id: int, skill_id: int, relevance_score: float = 1.0) -> ContentSkill:
        """
        Maps a skill to content, updating the relevance_score if the relationship already exists.
        """
        statement = select(ContentSkill).where(ContentSkill.content_id == content_id, ContentSkill.skill_id == skill_id)
        existing = db.execute(statement).scalar_one_or_none()
        
        if existing:
            existing.relevance_score = relevance_score
            db.commit()
            db.refresh(existing)
            return existing
        else:
            content_skill = ContentSkill(
                content_id=content_id,
                skill_id=skill_id,
                relevance_score=relevance_score,
                created_at=datetime.utcnow()
            )
            db.add(content_skill)
            db.commit()
            db.refresh(content_skill)
            return content_skill

class InteractionRepository(BaseRepository[Interaction]):
    """
    Repository for logging and managing interaction events.
    """
    def __init__(self):
        super().__init__(Interaction)

    def record_interaction(
        self,
        db: Session,
        user_id: int,
        content_id: int,
        type: str,
        rating: Optional[float] = None
    ) -> Interaction:
        """
        Records a new interaction event between a user and content.
        """
        interaction = Interaction(
            user_id=user_id,
            content_id=content_id,
            type=type,
            rating=rating,
            created_at=datetime.utcnow()
        )
        db.add(interaction)
        db.commit()
        db.refresh(interaction)
        return interaction

class SkillRepository(BaseRepository[Skill]):
    """
    Repository for managing learning skill taxonomy.
    """
    def __init__(self):
        super().__init__(Skill)

    def get_by_name(self, db: Session, name: str) -> Optional[Skill]:
        """
        Retrieve a skill by its unique name.
        """
        statement = select(Skill).where(Skill.name == name)
        return db.execute(statement).scalar_one_or_none()

# Instantiated repositories
user_repository = UserRepository()
content_repository = ContentRepository()
interaction_repository = InteractionRepository()
skill_repository = SkillRepository()
