import pytest
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
from api.core.database import get_db
from api.models import Base
from api.models.user import User
from api.models.item import Item
from api.models.rating import Rating

from sqlalchemy.pool import StaticPool

# In-memory SQLite for testing with StaticPool to keep the same database connection active
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(name="db_session", scope="function")
def db_session_fixture():
    # Create the database schema
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Drop schema so next test gets clean database
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(name="client", scope="function")
def client_fixture(db_session):
    # Override get_db to return TestingSessionLocal
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
            
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(name="seed_test_data", scope="function")
def seed_test_data_fixture(db_session):
    # Seed items
    movies = [
        Item(id=1, title="Toy Story", genres="Animation|Children"),
        Item(id=2, title="Star Wars", genres="Sci-Fi|Action"),
        Item(id=3, title="Lion King", genres="Animation|Children"),
    ]
    db_session.add_all(movies)
    
    # Seed users
    users = [
        User(id=1, username="alice"),
        User(id=2, username="bob"),
    ]
    db_session.add_all(users)
    db_session.commit()
    
    # Seed ratings
    ratings = [
        Rating(user_id=1, item_id=1, rating=5.0), # Alice loves Toy Story
        Rating(user_id=1, item_id=2, rating=2.0), # Alice dislikes Star Wars
        Rating(user_id=2, item_id=2, rating=5.0), # Bob loves Star Wars
    ]
    db_session.add_all(ratings)
    db_session.commit()
