import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_TITLE: str = "Recommendation System API"
    API_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # Database URL
    DATABASE_URL: str = "sqlite:///./data/recommendation_system.db"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create instance
settings = Settings()

# Ensure data directory exists
os.makedirs("./data", exist_ok=True)
