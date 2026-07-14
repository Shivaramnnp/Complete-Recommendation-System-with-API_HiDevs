from pydantic import BaseModel
from typing import List

class AlgorithmMetrics(BaseModel):
    algorithm: str
    precision: float
    recall: float
    ndcg: float

class SystemEvaluationResponse(BaseModel):
    k: int
    test_split_ratio: float
    metrics: List[AlgorithmMetrics]
