import sys
import os

# Add project root to sys.path so we can import api modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from api.core.database import SessionLocal, engine
from api.models.base import Base
from api.models.user import User
from api.models.item import Item
from api.models.rating import Rating

def seed_database():
    print("Re-creating all database tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db: Session = SessionLocal()
    try:
        print("Seeding Users...")
        users = [
            User(username="alice"),       # ID 1
            User(username="bob"),         # ID 2
            User(username="charlie"),     # ID 3
            User(username="david"),       # ID 4
            User(username="eva"),         # ID 5
            User(username="frank"),       # ID 6
        ]
        db.add_all(users)
        db.commit()
        
        print("Seeding Items...")
        items = [
            Item(title="The Matrix", genres="Sci-Fi|Action", description="A hacker learns about the true nature of reality."), # ID 1
            Item(title="Inception", genres="Sci-Fi|Action|Thriller", description="A thief steals corporate secrets through dream-sharing technology."), # ID 2
            Item(title="Interstellar", genres="Sci-Fi|Drama", description="Astronauts travel through a wormhole in search of a new home for humanity."), # ID 3
            Item(title="Gladiator", genres="Action|Drama", description="A former Roman General sets out to exact vengeance against the corrupt emperor."), # ID 4
            Item(title="The Dark Knight", genres="Action|Crime|Drama", description="Batman raises the stakes in his war on crime with the help of Lt. Jim Gordon."), # ID 5
            Item(title="Superbad", genres="Comedy", description="Two co-dependent high school seniors are forced to deal with separation anxiety."), # ID 6
            Item(title="The Hangover", genres="Comedy", description="Three buddies wake up from a bachelor party in Las Vegas with no memory of the night."), # ID 7
            Item(title="Toy Story", genres="Animation|Children|Comedy", description="A cowboy doll is profoundly threatened and jealous when a new spaceman figure replaces him."), # ID 8
            Item(title="Finding Nemo", genres="Animation|Children", description="After his son is captured in the Great Barrier Reef, a timid clownfish embarks on a journey to bring him home."), # ID 9
            Item(title="Pulp Fiction", genres="Crime|Thriller", description="The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine."), # ID 10
        ]
        db.add_all(items)
        db.commit()
        
        print("Seeding Ratings...")
        ratings = [
            # Alice (ID 1) - Likes Sci-Fi/Action
            Rating(user_id=1, item_id=1, rating=5.0), # Matrix
            Rating(user_id=1, item_id=2, rating=5.0), # Inception
            Rating(user_id=1, item_id=3, rating=4.0), # Interstellar
            Rating(user_id=1, item_id=5, rating=4.5), # Dark Knight
            Rating(user_id=1, item_id=9, rating=3.5), # Finding Nemo
            
            # Bob (ID 2) - Likes Comedies
            Rating(user_id=2, item_id=6, rating=5.0), # Superbad
            Rating(user_id=2, item_id=7, rating=4.5), # Hangover
            Rating(user_id=2, item_id=8, rating=4.0), # Toy Story (Comedy element)
            Rating(user_id=2, item_id=9, rating=4.0), # Finding Nemo
            
            # Charlie (ID 3) - Likes Animation/Children
            Rating(user_id=3, item_id=8, rating=5.0), # Toy Story
            Rating(user_id=3, item_id=9, rating=4.5), # Finding Nemo
            Rating(user_id=3, item_id=2, rating=3.0), # Inception
            
            # David (ID 4) - Likes Sci-Fi/Drama
            Rating(user_id=4, item_id=1, rating=4.5), # Matrix
            Rating(user_id=4, item_id=3, rating=5.0), # Interstellar
            Rating(user_id=4, item_id=4, rating=4.0), # Gladiator
            
            # Eva (ID 5) - Likes Crime/Thrillers/Action
            Rating(user_id=5, item_id=10, rating=5.0), # Pulp Fiction
            Rating(user_id=5, item_id=2, rating=4.0), # Inception
            Rating(user_id=5, item_id=5, rating=4.5), # Dark Knight
            Rating(user_id=5, item_id=4, rating=4.0), # Gladiator
            
            # Frank (ID 6) - Sparse hybrid ratings
            Rating(user_id=6, item_id=1, rating=4.0), # Matrix
            Rating(user_id=6, item_id=6, rating=4.0), # Superbad
            Rating(user_id=6, item_id=5, rating=4.5), # Dark Knight
            Rating(user_id=6, item_id=4, rating=4.0), # Gladiator
        ]
        db.add_all(ratings)
        db.commit()
        print("Database seeded successfully with sample recommendations data!")
        
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
