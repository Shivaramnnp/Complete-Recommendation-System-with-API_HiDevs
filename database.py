import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

# SQLite database URL loaded from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/skills_recommendation.db")

# Ensure target database folder exists if using file-based SQLite
if DATABASE_URL.startswith("sqlite:///"):
    db_file_path = DATABASE_URL.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_file_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

# Create engine. SQLite-specific arguments: check_same_thread=False
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

# Enable SQLite Foreign Key Constraints
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# Session local maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session() -> Generator[Session, None, None]:
    """
    Generator yielding a database session context.
    Ensures that connections are closed properly after block completion.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
