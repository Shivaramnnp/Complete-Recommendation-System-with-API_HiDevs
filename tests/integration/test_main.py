import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from fastapi import status

from main import app
from database import get_db_session
from models import Base
from repositories import user_repository, content_repository, skill_repository, interaction_repository

@pytest.fixture(name="db")
def db_fixture():
    """
    Independent database session for FastAPI app main.py route integration testing.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    SessionTesting = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionTesting()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(name="client")
def client_fixture(db):
    """
    Test client overriding database dependency with testing session and injecting auth header.
    """
    def override_get_db_session():
        try:
            yield db
        finally:
            pass
            
    app.dependency_overrides[get_db_session] = override_get_db_session
    with TestClient(app) as test_client:
        test_client.headers = {"X-API-Key": "dev-secret-key"}
        yield test_client
    app.dependency_overrides.clear()

def test_request_id_middleware(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0

def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "online"
    assert "cache_keys_count" in data
    assert "avg_latency_ms" in data

def test_unauthorized_access(db):
    # Run tests without API Key header
    with TestClient(app) as client_no_auth:
        # /health check remains public for probes
        res_health = client_no_auth.get("/health")
        assert res_health.status_code == status.HTTP_200_OK
        
        # Protected endpoints must return 401 Unauthorized when missing headers
        res_rec = client_no_auth.get("/recommend/1")
        assert res_rec.status_code == status.HTTP_401_UNAUTHORIZED
        
        # Protected endpoints must return 403 Forbidden when using wrong key
        res_rec_wrong = client_no_auth.get("/recommend/1", headers={"X-API-Key": "wrong-key"})
        assert res_rec_wrong.status_code == status.HTTP_403_FORBIDDEN

def test_get_recommendations_cold_start(client, db):
    # Setup cold user (no interactions or skills)
    user = user_repository.create(db, {"username": "cold_user"})
    c = content_repository.create(db, {"title": "Pop Content", "type": "video"})
    
    # Establish popularity metrics (c has 1 interaction)
    u_other = user_repository.create(db, {"username": "other"})
    interaction_repository.record_interaction(db, u_other.id, c.id, "view")
    
    response = client.get(f"/recommend/{user.id}?algorithm=hybrid&k=3")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["user_id"] == user.id
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["content_id"] == c.id
    assert data["algorithm"] == "popularity" # Cold-start fallback
    assert data["cached"] is False

def test_get_recommendations_not_found(client):
    response = client.get("/recommend/9999")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"]

def test_post_feedback_success(client, db):
    user = user_repository.create(db, {"username": "john"})
    content = content_repository.create(db, {"title": "Intro Python", "type": "course"})
    
    response = client.post("/feedback", json={
        "user_id": user.id,
        "content_id": content.id,
        "type": "click",
        "rating": 4.5
    })
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["status"] == "success"
    assert "interaction_id" in data
    
    # Verify persistence
    history = user_repository.get_user_history(db, user.id)
    assert len(history) == 1
    assert history[0].content_id == content.id
    assert history[0].rating == 4.5

def test_post_feedback_validation_error(client):
    # Submit rating > 5.0 to trigger RequestValidationError
    response = client.post("/feedback", json={
        "user_id": 1,
        "content_id": 1,
        "type": "click",
        "rating": 10.0 # Invalid rating value
    })
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert "detail" in data
    assert "request_id" in data
    assert data["error_type"] == "ValidationError"

def test_post_feedback_user_not_found(client, db):
    content = content_repository.create(db, {"title": "Intro Python", "type": "course"})
    
    response = client.post("/feedback", json={
        "user_id": 9999, # User does not exist
        "content_id": content.id,
        "type": "click"
    })
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"]

def test_get_metrics_endpoint(client, db):
    # Seed sufficient density to test evaluations (>= 20 ratings)
    # Users: 5
    # Items: 5
    for idx in range(1, 6):
        user_repository.create(db, {"username": f"user_{idx}"})
        content_repository.create(db, {"title": f"course_{idx}", "type": "course"})
        
    # Add ratings
    for u in range(1, 6):
        for i in range(1, 6):
            interaction_repository.record_interaction(db, u, i, "complete", rating=4.0)
            
    response = client.get("/metrics?k=3&test_ratio=0.25")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["k"] == 3
    assert "metrics" in data
    
    algorithms = {m["algorithm"] for m in data["metrics"]}
    assert "hybrid" in algorithms
    assert "collaborative" in algorithms
    assert "content_based" in algorithms
