from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Text, DateTime, ForeignKey, Float, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    """
    SQLAlchemy Declarative Base class.
    """
    pass

class UserSkill(Base):
    """
    Association model mapping Users to Skills with a proficiency attribute.
    """
    __tablename__ = "user_skills"
    
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True)
    proficiency_level: Mapped[str] = mapped_column(String(50), default="Beginner", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Bi-directional relationships
    user: Mapped["User"] = relationship(back_populates="user_skills")
    skill: Mapped["Skill"] = relationship(back_populates="user_skills")

class ContentSkill(Base):
    """
    Association model mapping Content to Skills with a relevance score (0.0 to 1.0).
    """
    __tablename__ = "content_skills"
    
    content_id: Mapped[int] = mapped_column(ForeignKey("contents.id", ondelete="CASCADE"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True)
    relevance_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Bi-directional relationships
    content: Mapped["Content"] = relationship(back_populates="content_skills")
    skill: Mapped["Skill"] = relationship(back_populates="content_skills")

class User(Base):
    """
    Model representing users in the recommendation system.
    """
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(100), unique=True, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user_skills: Mapped[List["UserSkill"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    interactions: Mapped[List["Interaction"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Content(Base):
    """
    Model representing content (videos, articles, courses) in the catalog.
    """
    __tablename__ = "contents"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "video", "article", "course"
    url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    content_skills: Mapped[List["ContentSkill"]] = relationship(back_populates="content", cascade="all, delete-orphan")
    interactions: Mapped[List["Interaction"]] = relationship(back_populates="content", cascade="all, delete-orphan")

class Skill(Base):
    """
    Model representing skills/learning goals mapping users to target content.
    """
    __tablename__ = "skills"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user_skills: Mapped[List["UserSkill"]] = relationship(back_populates="skill", cascade="all, delete-orphan")
    content_skills: Mapped[List["ContentSkill"]] = relationship(back_populates="skill", cascade="all, delete-orphan")

class Interaction(Base):
    """
    Model tracking interaction events (views, completions, ratings) between users and content.
    """
    __tablename__ = "interactions"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content_id: Mapped[int] = mapped_column(ForeignKey("contents.id", ondelete="CASCADE"), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "view", "click", "bookmark", "complete"
    rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Optional rating scale (e.g. 1.0 to 5.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="interactions")
    content: Mapped["Content"] = relationship(back_populates="interactions")
    
    # Constraints: Prevent duplicate identical interaction logging on same timestamp
    __table_args__ = (
        UniqueConstraint("user_id", "content_id", "type", "created_at", name="uq_user_content_interaction"),
    )
