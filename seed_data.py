import os
import random
from datetime import datetime
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, User, Content, Skill, UserSkill, ContentSkill, Interaction

def seed_database():
    print("Initializing SQLite Database Tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db: Session = SessionLocal()
    try:
        # 1. Seed 5 Skills
        skills = [
            Skill(id=1, name="Python", description="Programming Language"),
            Skill(id=2, name="SQL", description="Database Querying"),
            Skill(id=3, name="AWS", description="Cloud Infrastructure"),
            Skill(id=4, name="Docker", description="Containerization"),
            Skill(id=5, name="Machine Learning", description="Data Science & Modeling")
        ]
        db.add_all(skills)
        db.commit()
        print("Seeded 5 skills.")

        # 2. Seed 10 Users
        users = [
            User(id=i, username=f"user_{i}", email=f"user_{i}@example.com")
            for i in range(1, 11)
        ]
        db.add_all(users)
        db.commit()
        print("Seeded 10 users.")

        # 3. Map Users to Skills (Proficiency Levels)
        # Give some user profiles explicit interests to test Content-Based routing
        user_skills = [
            # User 1 & 2 likes Python & ML
            UserSkill(user_id=1, skill_id=1, proficiency_level="Advanced"),
            UserSkill(user_id=1, skill_id=5, proficiency_level="Beginner"),
            UserSkill(user_id=2, skill_id=1, proficiency_level="Intermediate"),
            UserSkill(user_id=2, skill_id=5, proficiency_level="Intermediate"),
            
            # User 3 & 4 likes SQL & AWS
            UserSkill(user_id=3, skill_id=2, proficiency_level="Advanced"),
            UserSkill(user_id=3, skill_id=3, proficiency_level="Intermediate"),
            UserSkill(user_id=4, skill_id=2, proficiency_level="Beginner"),
            UserSkill(user_id=4, skill_id=3, proficiency_level="Advanced"),
            
            # User 5 & 6 likes AWS & Docker
            UserSkill(user_id=5, skill_id=3, proficiency_level="Intermediate"),
            UserSkill(user_id=5, skill_id=4, proficiency_level="Advanced"),
            UserSkill(user_id=6, skill_id=3, proficiency_level="Beginner"),
            UserSkill(user_id=6, skill_id=4, proficiency_level="Intermediate"),
            
            # User 7 & 8 likes Python & Docker
            UserSkill(user_id=7, skill_id=1, proficiency_level="Intermediate"),
            UserSkill(user_id=7, skill_id=4, proficiency_level="Advanced"),
            UserSkill(user_id=8, skill_id=1, proficiency_level="Advanced"),
            UserSkill(user_id=8, skill_id=4, proficiency_level="Beginner"),
            
            # User 9 has ML interest
            UserSkill(user_id=9, skill_id=5, proficiency_level="Advanced"),
            # User 10 is cold-start (no skills, we will give them no skills)
        ]
        db.add_all(user_skills)
        db.commit()
        print("Seeded user skills.")

        # 4. Seed 20 Content Items
        # content_1 to content_20 with different types
        types = ["course", "video", "article"]
        contents = [
            Content(
                id=i, 
                title=f"Content Topic {i}", 
                type=types[i % 3], 
                description=f"Description summary for content topic {i}"
            )
            for i in range(1, 21)
        ]
        db.add_all(contents)
        db.commit()
        print("Seeded 20 content items.")

        # 5. Map Content to Skills (Relevance Scores)
        # Content 1-4 covers Python (skill 1)
        # Content 5-8 covers SQL (skill 2)
        # Content 9-12 covers AWS (skill 3)
        # Content 13-16 covers Docker (skill 4)
        # Content 17-20 covers Machine Learning (skill 5)
        content_skills = []
        for i in range(1, 21):
            if i <= 4:
                content_skills.append(ContentSkill(content_id=i, skill_id=1, relevance_score=0.9))
            elif i <= 8:
                content_skills.append(ContentSkill(content_id=i, skill_id=2, relevance_score=0.85))
            elif i <= 12:
                content_skills.append(ContentSkill(content_id=i, skill_id=3, relevance_score=0.9))
            elif i <= 16:
                content_skills.append(ContentSkill(content_id=i, skill_id=4, relevance_score=0.8))
            else:
                content_skills.append(ContentSkill(content_id=i, skill_id=5, relevance_score=0.95))
        db.add_all(content_skills)
        db.commit()
        print("Seeded content skills relevance maps.")

        # 6. Seed exactly 50 interactions
        # We will define exactly 50 interactions deterministically to avoid random fluctuations.
        # User 1-9 rate various contents. User 10 is cold-start and gets no ratings.
        raw_interactions = [
            # User 1 (Python/ML profile) rates Python courses (1-4) & ML (17-20)
            (1, 1, "complete", 5.0),
            (1, 2, "view", 4.0),
            (1, 17, "click", 4.5),
            (1, 18, "view", 3.5),
            (1, 5, "click", 2.0), # dislikes SQL course
            
            # User 2 (Python/ML profile)
            (2, 1, "view", 4.5),
            (2, 2, "complete", 4.0),
            (2, 19, "click", 5.0),
            (2, 20, "view", 4.0),
            (2, 6, "click", 3.0),
            
            # User 3 (SQL/AWS profile) rates SQL (5-8) & AWS (9-12)
            (3, 5, "complete", 5.0),
            (3, 6, "view", 4.5),
            (3, 9, "click", 4.0),
            (3, 10, "view", 4.0),
            (3, 13, "click", 3.0),
            
            # User 4 (SQL/AWS profile)
            (4, 5, "view", 4.0),
            (4, 7, "complete", 5.0),
            (4, 11, "click", 4.5),
            (4, 12, "view", 4.0),
            (4, 14, "click", 2.5),
            
            # User 5 (AWS/Docker profile) rates AWS (9-12) & Docker (13-16)
            (5, 9, "complete", 5.0),
            (5, 13, "view", 4.5),
            (5, 14, "click", 4.0),
            (5, 10, "view", 4.0),
            (5, 1, "click", 3.0),
            
            # User 6 (AWS/Docker profile)
            (6, 9, "view", 4.5),
            (6, 15, "complete", 5.0),
            (6, 16, "click", 4.0),
            (6, 11, "view", 3.5),
            (6, 2, "click", 3.0),
            
            # User 7 (Python/Docker profile) rates Python (1-4) & Docker (13-16)
            (7, 3, "complete", 5.0),
            (7, 4, "view", 4.5),
            (7, 13, "click", 4.0),
            (7, 14, "view", 4.0),
            (7, 7, "click", 3.0),
            
            # User 8 (Python/Docker profile)
            (8, 3, "view", 4.0),
            (8, 15, "complete", 4.5),
            (8, 16, "click", 5.0),
            (8, 1, "view", 4.0),
            (8, 8, "click", 2.5),
            
            # User 9 (ML profile) rates ML (17-20)
            (9, 17, "complete", 5.0),
            (9, 18, "view", 4.5),
            (9, 19, "click", 4.0),
            (9, 20, "view", 4.0),
            (9, 9, "click", 3.0),
            
            # Additional ratings to reach exactly 50 interactions
            (1, 3, "click", 4.5),
            (2, 3, "view", 4.0),
            (3, 7, "click", 4.5),
            (4, 6, "view", 4.0),
            (5, 11, "click", 4.5)
        ]
        
        assert len(raw_interactions) == 50, f"Error: Interactions count is {len(raw_interactions)}, must be exactly 50!"
        
        interactions = [
            Interaction(
                user_id=u_id,
                content_id=c_id,
                type=t_str,
                rating=r_val,
                created_at=datetime.utcnow()
            )
            for u_id, c_id, t_str, r_val in raw_interactions
        ]
        
        db.add_all(interactions)
        db.commit()
        print("Seeded exactly 50 interactions.")
        print("Database seeding completed successfully!")
        
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
