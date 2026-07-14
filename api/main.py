from fastapi import FastAPI
from api.core.config import settings
from api.core.database import engine
from api.models.base import Base
from api.routers import users, items, recommendations

# Initialize database tables
# For production systems we would use Alembic migrations,
# but for this SQLite-backed capstone API, we initialize on startup.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="A complete recommendation system API showcasing Collaborative, Content-Based, and Hybrid filtering."
)

# Register routers
app.include_router(users.router)
app.include_router(items.router)
app.include_router(recommendations.router)

@app.get("/", tags=["Health"])
def health_check():
    """
    Health check and metadata endpoint.
    """
    return {
        "status": "online",
        "api_title": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs_url": "/docs"
    }
