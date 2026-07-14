import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from models import Base, User, Content, Skill, UserSkill, ContentSkill, Interaction
from repositories import user_repository, content_repository, interaction_repository, skill_repository

@pytest.fixture(name="db")
def db_fixture():
    """
    Dedicated db session for testing models.py and repositories.py database layer.
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

def test_user_repository_crud(db):
    # 1. Create user
    user = user_repository.create(db, {"username": "john_doe", "email": "john@example.com"})
    assert user.id is not None
    assert user.username == "john_doe"
    
    # 2. Get by username
    john = user_repository.get_by_username(db, "john_doe")
    assert john is not None
    assert john.id == user.id
    
    # 3. Update user
    user_repository.update(db, john, {"email": "john_new@example.com"})
    updated = user_repository.get(db, john.id)
    assert updated.email == "john_new@example.com"
    
    # 4. Delete user
    assert user_repository.delete(db, john.id) is True
    assert user_repository.get(db, john.id) is None

def test_user_skill_association(db):
    user = user_repository.create(db, {"username": "alice"})
    skill = skill_repository.create(db, {"name": "Python", "description": "Programming language"})
    
    # Add skill
    user_skill = user_repository.add_user_skill(db, user.id, skill.id, "Intermediate")
    assert user_skill.proficiency_level == "Intermediate"
    assert user_skill.user_id == user.id
    assert user_skill.skill_id == skill.id
    
    # Update skill
    user_skill_updated = user_repository.add_user_skill(db, user.id, skill.id, "Advanced")
    assert user_skill_updated.proficiency_level == "Advanced"
    
    # Query via relationship
    db_user = user_repository.get(db, user.id)
    assert len(db_user.user_skills) == 1
    assert db_user.user_skills[0].skill.name == "Python"

def test_content_repository_by_skill_and_search(db):
    # Setup Content & Skills
    c1 = content_repository.create(db, {"title": "Python Basics", "type": "course", "description": "Learn python scripting"})
    c2 = content_repository.create(db, {"title": "SQL Query Optimization", "type": "article", "description": "Database query tuning"})
    
    s_py = skill_repository.create(db, {"name": "Python"})
    s_sql = skill_repository.create(db, {"name": "SQL"})
    
    # Map Content to Skills
    content_repository.map_content_skill(db, c1.id, s_py.id, relevance_score=0.9)
    content_repository.map_content_skill(db, c2.id, s_sql.id, relevance_score=0.8)
    
    # Query by skill
    py_content = content_repository.get_content_by_skill(db, s_py.id)
    assert len(py_content) == 1
    assert py_content[0].title == "Python Basics"
    
    # Search content
    matches = content_repository.search_content(db, "optimization")
    assert len(matches) == 1
    assert matches[0].title == "SQL Query Optimization"
    
    # Case-insensitive matches
    matches_case = content_repository.search_content(db, "BASIC")
    assert len(matches_case) == 1
    assert matches_case[0].title == "Python Basics"

def test_popularity_and_cold_start_queries(db):
    # Setup Content
    c1 = content_repository.create(db, {"title": "Python course", "type": "course"})
    c2 = content_repository.create(db, {"title": "React article", "type": "article"})
    c3 = content_repository.create(db, {"title": "Docker video", "type": "video"})
    
    # Setup Users
    u1 = user_repository.create(db, {"username": "u1"})
    u2 = user_repository.create(db, {"username": "u2"})
    
    # Add interactions:
    # c2 has 2 interactions
    # c1 has 1 interaction
    # c3 has 0 interactions
    interaction_repository.record_interaction(db, u1.id, c2.id, "view", rating=4.5)
    interaction_repository.record_interaction(db, u2.id, c2.id, "like")
    
    interaction_repository.record_interaction(db, u1.id, c1.id, "complete", rating=5.0)
    
    # Test popularity
    pop = content_repository.get_popularity_scores(db, limit=3)
    assert len(pop) == 3
    # First should be c2 (count=2)
    assert pop[0][0].id == c2.id
    assert pop[0][1] == 2
    # Second should be c1 (count=1)
    assert pop[1][0].id == c1.id
    assert pop[1][1] == 1
    
    # Test cold start query: sorted by average rating desc (c1 has avg rating 5.0, c2 has avg rating 4.5, c3 has None)
    cold = content_repository.get_cold_start_content(db, limit=3)
    assert len(cold) == 3
    # c1 (5.0 rating) should be first
    assert cold[0].id == c1.id
    # c2 (4.5 rating) should be second
    assert cold[1].id == c2.id
    # c3 (no rating) should be last
    assert cold[2].id == c3.id

def test_user_history_and_cascade_delete(db):
    u = user_repository.create(db, {"username": "test_user"})
    c = content_repository.create(db, {"title": "Test course", "type": "course"})
    s = skill_repository.create(db, {"name": "Test skill"})
    
    # Map skill
    user_repository.add_user_skill(db, u.id, s.id, "Beginner")
    
    # Add interactions
    interaction_repository.record_interaction(db, u.id, c.id, "click")
    interaction_repository.record_interaction(db, u.id, c.id, "complete", rating=4.0)
    
    # Query history
    history = user_repository.get_user_history(db, u.id)
    assert len(history) == 2
    assert history[0].type == "complete" # sorted desc (newest first)
    assert history[1].type == "click"
    
    # Verify cascades: delete user
    user_repository.delete(db, u.id)
    
    # Interaction repository should have 0 records for this user
    remaining_interactions = db.query(Interaction).filter(Interaction.user_id == u.id).all()
    assert len(remaining_interactions) == 0
    
    # UserSkill association records should be gone
    remaining_skills = db.query(UserSkill).filter(UserSkill.user_id == u.id).all()
    assert len(remaining_skills) == 0
