import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import numpy as np
from models import Base, User, Content, Skill, UserSkill, ContentSkill, Interaction
from repositories import user_repository, content_repository, skill_repository, interaction_repository
from recommendation_engine import (
    SimilarityCalculator,
    RecommendationEvaluator,
    RecommendationOrchestrator,
    recommendation_orchestrator
)

@pytest.fixture(name="db")
def db_fixture():
    """
    Independent database session for recommendation engine testing.
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

def test_similarity_calculator():
    # Cosine test
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([0.0, 1.0, 1.0])
    # dot = 1.0, norm = sqrt(2) * sqrt(2) = 2.0 -> sim = 0.5
    assert abs(SimilarityCalculator.cosine_similarity(v1, v2) - 0.5) < 1e-6
    
    # Jaccard test
    s1 = {"Python", "SQL"}
    s2 = {"SQL", "AWS"}
    # intersection: {'SQL'} (1), union: {'Python', 'SQL', 'AWS'} (3) -> sim = 1/3
    assert abs(SimilarityCalculator.jaccard_similarity(s1, s2) - (1.0 / 3.0)) < 1e-6

def test_recommendation_evaluator():
    recs = [1, 2, 3]
    test_ids = [2, 4]
    
    # Precision@2: recs[:2] = [1, 2], hits: [2] -> 1/2 = 0.5
    assert RecommendationEvaluator.precision_at_k(recs, test_ids, 2) == 0.5
    # Recall@3: recs[:3] = [1, 2, 3], hits: [2], total relevant: 2 -> 1/2 = 0.5
    assert RecommendationEvaluator.recall_at_k(recs, test_ids, 3) == 0.5
    # NDCG@2: hits at index 1 -> DCG = 1/log2(3) = 0.6309. IDCG@2 = 1.0 -> NDCG = 0.6309
    ndcg = RecommendationEvaluator.ndcg_at_k(recs, test_ids, 2)
    expected = (1.0 / np.log2(3)) / (1.0 + 1.0 / np.log2(3))
    assert abs(ndcg - expected) < 1e-6

def test_cold_start_fallback(db):
    # Setup users, content
    user = user_repository.create(db, {"username": "cold_user"})
    c1 = content_repository.create(db, {"title": "Course A", "type": "course"})
    c2 = content_repository.create(db, {"title": "Course B", "type": "course"})
    
    # Add interaction to establish popularity stats (c1 has 1 interaction, c2 has 0)
    u_other = user_repository.create(db, {"username": "other"})
    interaction_repository_record(db, u_other.id, c1.id, "view")
    
    # Orchestrate for cold start user (no history, no skills)
    orchestrator = RecommendationOrchestrator(cache_ttl_seconds=10)
    result = orchestrator.orchestrate(db, user.id, algorithm="hybrid", k=5)
    
    assert result.user_id == user.id
    assert len(result.recommendations) > 0
    assert result.algorithm == "popularity" # Falls back to popularity
    assert result.recommendations[0].content_id == c1.id
    assert "popular" in result.recommendations[0].explanation

def test_content_based_recommendations(db):
    # Create target user
    user = user_repository.create(db, {"username": "tech_user"})
    
    # Create skills
    s_py = skill_repository.create(db, {"name": "Python"})
    s_sql = skill_repository.create(db, {"name": "SQL"})
    
    # Add skills to user profile
    user_repository.add_user_skill(db, user.id, s_py.id, "Advanced")
    
    # Create content
    c_py = content_repository.create(db, {"title": "Advanced Python", "type": "course"})
    c_sql = content_repository.create(db, {"title": "SQL Basics", "type": "article"})
    
    # Map content to skills
    content_repository.map_content_skill(db, c_py.id, s_py.id, relevance_score=0.9)
    content_repository.map_content_skill(db, c_sql.id, s_sql.id, relevance_score=0.8)
    
    # Orchestrate
    orchestrator = RecommendationOrchestrator(cache_ttl_seconds=10)
    # User has Advanced Python skill. Should recommend c_py over c_sql
    result = orchestrator.orchestrate(db, user.id, algorithm="content_based", k=5)
    
    assert result.algorithm == "content_based"
    assert len(result.recommendations) == 1
    assert result.recommendations[0].content_id == c_py.id
    assert "Python" in result.recommendations[0].explanation

def test_caching_and_feedback_invalidation(db):
    user = user_repository.create(db, {"username": "cache_user"})
    c = content_repository.create(db, {"title": "General Course", "type": "course"})
    
    # Seed interactions to prevent raw blank error
    u_other = user_repository.create(db, {"username": "other"})
    interaction_repository_record(db, u_other.id, c.id, "view")
    
    orchestrator = RecommendationOrchestrator(cache_ttl_seconds=10)
    
    # First call: not cached
    res1 = orchestrator.orchestrate(db, user.id, algorithm="popularity", k=5)
    assert res1.cached is False
    
    # Second call: cached
    res2 = orchestrator.orchestrate(db, user.id, algorithm="popularity", k=5)
    assert res2.cached is True
    
    # Record feedback -> invalidates cache
    orchestrator.record_feedback(db, user.id, c.id, "click")
    
    # Third call: cache hit should be invalid, recalculates (cached=False)
    res3 = orchestrator.orchestrate(db, user.id, algorithm="popularity", k=5)
    assert res3.cached is False

def interaction_repository_record(db, user_id, content_id, type_str):
    # Utility function to add raw interactions for testing
    from models import Interaction
    from datetime import datetime
    interaction = Interaction(
        user_id=user_id,
        content_id=content_id,
        type=type_str,
        created_at=datetime.utcnow()
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction
