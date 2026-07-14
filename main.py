import contextvars
import logging
import uuid
import pandas as pd
import os
import threading
import time
from typing import List, Optional
from fastapi.security import APIKeyHeader
from fastapi import FastAPI, Depends, HTTPException, status, Query, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from database import get_db_session, engine
from models import Base, UserSkill, ContentSkill, Skill
from repositories import user_repository, content_repository, interaction_repository
from recommendation_engine import recommendation_orchestrator, RecommendationEvaluator

# Initialize database tables on startup
# For production systems we would use Alembic migrations,
# but for this SQLite-backed capstone API, we initialize on startup.
Base.metadata.create_all(bind=engine)

# Setup contextvar to carry Request IDs for log correlation
request_id_var = contextvars.ContextVar("request_id", default="-")

class RequestIDFilter(logging.Filter):
    """
    Log filter that injects the current Request ID from the contextvar into the log record.
    """
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

# Configure logging format
logger = logging.getLogger("recommendation_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.addFilter(RequestIDFilter())
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(request_id)s] %(name)s: %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# API Key Configuration & Validator
API_KEY = os.getenv("RECOMMENDER_API_KEY", "dev-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def validate_api_key(api_key: str = Depends(api_key_header)):
    """
    Dependency that enforces client request holds valid X-API-Key headers.
    """
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key."
        )
    return api_key

# Request Latency Statistics Tracker
total_requests = 0
total_latency = 0.0
stats_lock = threading.Lock()

# FastAPI initialization
app = FastAPI(
    title="Skills-Based Recommendation API",
    version="1.0.0",
    description="Production-grade API delivering personalized content recommendations based on skills and profiles."
)

# Request ID & Performance Logging Middleware
@app.middleware("http")
async def request_logger_middleware(request: Request, call_next):
    # Retrieve request ID from headers or generate a new one
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_var.set(req_id)
    
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Incoming request: {request.method} {request.url.path} from IP {client_ip}")
    
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    
    # Update performance metrics safely
    global total_requests, total_latency
    with stats_lock:
        total_requests += 1
        total_latency += duration_ms
        
    response.headers["X-Request-ID"] = req_id
    logger.info(
        f"Outgoing response: {request.method} {request.url.path} - Status: {response.status_code} - Latency: {duration_ms:.2f}ms"
    )
    return response

# Pydantic Schemas for Requests and Responses
class FeedbackRequest(BaseModel):
    user_id: int = Field(..., description="ID of the user performing the interaction")
    content_id: int = Field(..., description="ID of the content being interacted with")
    type: str = Field(..., min_length=2, max_length=50, description="Type of interaction, e.g. 'click', 'view', 'complete'")
    rating: Optional[float] = Field(None, ge=0.5, le=5.0, description="Optional rating score between 0.5 and 5.0")

class FeedbackResponse(BaseModel):
    status: str
    message: str
    interaction_id: int

class RecommendationItemResponse(BaseModel):
    content_id: int
    score: float
    explanation: str

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItemResponse]
    algorithm: str
    cached: bool

class HealthResponse(BaseModel):
    status: str
    database: str
    cache_keys_count: int
    avg_latency_ms: float

class MetricItemResponse(BaseModel):
    algorithm: str
    precision: float
    recall: float
    ndcg: float

class MetricsResponse(BaseModel):
    k: int
    test_split_ratio: float
    metrics: List[MetricItemResponse]

# Custom Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = request_id_var.get()
    logger.warning(f"Validation failure for request: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "request_id": req_id,
            "error_type": "ValidationError"
        }
    )

@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(request: Request, exc: SQLAlchemyError):
    req_id = request_id_var.get()
    logger.error(f"Database error occurred: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal database persistence error occurred. Please retry.",
            "request_id": req_id,
            "error_type": "DatabaseError"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    req_id = request_id_var.get()
    logger.error(f"Unhandled server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected system error occurred. Please contact support.",
            "request_id": req_id,
            "error_type": "InternalServerError"
        }
    )

# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check(db: Session = Depends(get_db_session)):
    """
    Performs system diagnostics verifying database connectivity and listing recommendation cache load.
    """
    db_status = "online"
    try:
        # Simple ping query
        db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Healthcheck database connection failed: {e}")
        db_status = "offline"
        
    cache_keys_count = len(recommendation_orchestrator._cache)
    
    # Calculate average latency
    avg_latency_ms = 0.0
    global total_requests, total_latency
    with stats_lock:
        if total_requests > 0:
            avg_latency_ms = total_latency / total_requests
            
    status_code = status.HTTP_200_OK
    if db_status == "offline":
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if db_status == "online" else "unhealthy",
            "database": db_status,
            "cache_keys_count": cache_keys_count,
            "avg_latency_ms": round(avg_latency_ms, 2)
        }
    )

@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(
    user_id: int,
    algorithm: str = Query(default="hybrid", pattern="^(collaborative|content_based|popularity|hybrid)$"),
    k: int = Query(default=10, ge=1, le=100),
    apply_reranking: bool = Query(default=True),
    db: Session = Depends(get_db_session),
    api_key: str = Depends(validate_api_key)
):
    """
    Retrieves personalized recommendations for a user.
    Falls back automatically to popularity algorithm if the user is a cold start (has no history/skills).
    """
    # Verify user exists
    user = user_repository.get(db, user_id)
    if not user:
        logger.warning(f"Recommendations requested for nonexistent user: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found."
        )
        
    # Orchestrate recommendations
    result = recommendation_orchestrator.orchestrate(
        db=db,
        user_id=user_id,
        algorithm=algorithm,
        k=k,
        apply_reranking=apply_reranking
    )
    
    # Map to schema response
    recs_out = [
        RecommendationItemResponse(
            content_id=rec.content_id,
            score=rec.score,
            explanation=rec.explanation
        )
        for rec in result.recommendations
    ]
    
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recs_out,
        algorithm=result.algorithm,
        cached=result.cached
    )

@app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED, tags=["Feedback"])
def record_feedback(
    req: FeedbackRequest,
    db: Session = Depends(get_db_session),
    api_key: str = Depends(validate_api_key)
):
    """
    Records a user interaction feedback event (clicks, views, ratings) in the database.
    This call automatically invalidates the user's recommendation cache to reflect their updated interest profile immediately.
    """
    # Verify user exists
    user = user_repository.get(db, req.user_id)
    if not user:
        logger.warning(f"Feedback submitted for nonexistent user: {req.user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {req.user_id} not found."
        )
        
    # Verify content exists
    content = content_repository.get(db, req.content_id)
    if not content:
        logger.warning(f"Feedback submitted for nonexistent content: {req.content_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Content with ID {req.content_id} not found."
        )
        
    # Record feedback and clear cache
    interaction = recommendation_orchestrator.record_feedback(
        db=db,
        user_id=req.user_id,
        content_id=req.content_id,
        type=req.type,
        rating=req.rating
    )
    
    logger.info(f"Feedback recorded successfully for user {req.user_id} on content {req.content_id}")
    return FeedbackResponse(
        status="success",
        message="Feedback logged successfully. User cache invalidated.",
        interaction_id=interaction.id
    )

@app.get("/metrics", response_model=MetricsResponse, tags=["Diagnostics"])
def get_offline_metrics(
    k: int = Query(default=3, ge=1, le=20),
    test_ratio: float = Query(default=0.25, ge=0.1, le=0.5),
    db: Session = Depends(get_db_session),
    api_key: str = Depends(validate_api_key)
):
    """
    Computes and retrieves system-wide offline metrics evaluation comparison across all strategies (collaborative, content_based, hybrid, popularity).
    """
    # Retrieve all interactions to verify data density
    all_interactions = interaction_repository.get_multi(db, limit=1000)
    if len(all_interactions) < 20:
        logger.info(f"Insufficient interaction data ({len(all_interactions)}) to split for metrics evaluation.")
        return MetricsResponse(k=k, test_split_ratio=test_ratio, metrics=[])
        
    # Standard random seed for split reproducibility
    import random
    random.seed(42)
    shuffled = list(all_interactions)
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * (1.0 - test_ratio))
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]
    
    # Relevant items: ratings >= 3.5
    ground_truth = {}
    for interaction in test_data:
        if interaction.rating and interaction.rating >= 3.5:
            ground_truth.setdefault(interaction.user_id, []).append(interaction.content_id)
            
    eval_users = list(ground_truth.keys())
    if not eval_users:
        logger.info("No relevant items in test set holdout split.")
        return MetricsResponse(k=k, test_split_ratio=test_ratio, metrics=[])
        
    # We will compute metrics using RecommendationEvaluator
    # We will instantiate mock orchestrators with train subset
    # Load all content features
    content_skills_obj = db.query(ContentSkill).all()
    content_skills_map = {}
    for cs in content_skills_obj:
        content_skills_map.setdefault(cs.content_id, []).append({
            "skill_id": cs.skill_id,
            "relevance_score": cs.relevance_score
        })
        
    # Fetch user skills
    all_user_skills = db.query(UserSkill).all()
    user_skills_map = {}
    for us in all_user_skills:
        user_skills_map.setdefault(us.user_id, []).append({
            "skill_id": us.skill_id,
            "proficiency_level": us.proficiency_level
        })
        
    # Fetch content popularity stats
    popularity_stats = [(c.id, count) for c, count in content_repository.get_popularity_scores(db, limit=50)]
    
    algorithms = ["collaborative", "content_based", "hybrid", "popularity"]
    metric_results = []
    
    for alg in algorithms:
        precisions = []
        recalls = []
        ndcgs = []
        
        # Simple sub-orchestrator evaluation simulation
        # For simplicity, we compute similarity matrices on train subset
        train_user_history = {}
        for x in train_data:
            train_user_history.setdefault(x.user_id, set()).add(x.content_id)
            
        # Collaborative matrices from train
        df_train_data = [{"user_id": x.user_id, "content_id": x.content_id, "rating": x.rating or 3.0} for x in train_data]
        user_user_sim = pd.DataFrame()
        user_item_matrix = pd.DataFrame()
        if df_train_data and alg in ["collaborative", "hybrid"]:
            df_train = pd.DataFrame(df_train_data)
            user_item_matrix = df_train.pivot_table(index="user_id", columns="content_id", values="rating", aggfunc="max").fillna(0.0)
            from recommendation_engine import SimilarityCalculator
            user_user_sim = SimilarityCalculator.compute_user_user_similarity(user_item_matrix)
            
        for user_id in eval_users:
            history = train_user_history.get(user_id, set())
            from recommendation_engine import CandidateGenerator, RecommendationScorer
            c_generator = CandidateGenerator(history)
            
            scored_items = {}
            cf_scores = {}
            cb_scores = {}
            
            if alg in ["collaborative", "hybrid"] and not user_item_matrix.empty and user_id in user_item_matrix.index:
                cf_candidates = c_generator.generate_collaborative_candidates(user_id, user_user_sim, user_item_matrix)
                cf_scores = RecommendationScorer.score_collaborative(user_id, cf_candidates, user_user_sim, user_item_matrix)
                
            if alg in ["content_based", "hybrid"]:
                u_skills = user_skills_map.get(user_id, [])
                cb_candidates = c_generator.generate_content_candidates(u_skills, content_skills_map)
                cb_scores = RecommendationScorer.score_content(u_skills, cb_candidates, content_skills_map)
                
            if alg == "collaborative":
                scored_items = cf_scores
            elif alg == "content_based":
                scored_items = cb_scores
            elif alg == "hybrid":
                scored_items = RecommendationScorer.blend_hybrid_scores(cf_scores, cb_scores)
            elif alg == "popularity":
                pop_candidates = c_generator.generate_popularity_candidates(popularity_stats, limit=k)
                for idx, c_id in enumerate(pop_candidates):
                    scored_items[c_id] = float(1.0 - (idx / len(pop_candidates)))
                    
            sorted_recs = sorted(scored_items.keys(), key=lambda x: scored_items[x], reverse=True)[:k]
            relevant = ground_truth[user_id]
            
            precisions.append(RecommendationEvaluator.precision_at_k(sorted_recs, relevant, k))
            recalls.append(RecommendationEvaluator.recall_at_k(sorted_recs, relevant, k))
            ndcgs.append(RecommendationEvaluator.ndcg_at_k(sorted_recs, relevant, k))
            
        metric_results.append(
            MetricItemResponse(
                algorithm=alg,
                precision=float(sum(precisions) / len(precisions)) if precisions else 0.0,
                recall=float(sum(recalls) / len(recalls)) if recalls else 0.0,
                ndcg=float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0
            )
        )
        
    return MetricsResponse(
        k=k,
        test_split_ratio=test_ratio,
        metrics=metric_results
    )
