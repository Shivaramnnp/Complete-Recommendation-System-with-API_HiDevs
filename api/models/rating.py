from sqlalchemy import Integer, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.models.base import Base

class Rating(Base):
    __tablename__ = "ratings"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    item_id: Mapped[int] = mapped_column(Integer, ForeignKey("items.id", ondelete="CASCADE"), nullable=False)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    item = relationship("Item", back_populates="ratings")
    
    # Constraints: A user can rate a specific item only once.
    __table_args__ = (
        UniqueConstraint("user_id", "item_id", name="uq_user_item_rating"),
    )
