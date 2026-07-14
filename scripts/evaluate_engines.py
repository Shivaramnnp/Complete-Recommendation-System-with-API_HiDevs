import sys
import os

# Add project root to sys.path so we can import api modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from api.core.database import SessionLocal
from api.services.recommendation import recommendation_service

def evaluate_engines():
    print("=" * 60)
    print("Recommendation System Engine Evaluation CLI")
    print("=" * 60)
    
    db: Session = SessionLocal()
    try:
        # Run system evaluation
        k = 3
        test_ratio = 0.25
        print(f"Running offline evaluations (K={k}, holdout={test_ratio * 100}%)...")
        
        evaluation = recommendation_service.evaluate_system(db, k=k, test_ratio=test_ratio)
        
        if not evaluation.metrics:
            print("\n[WARNING] Insufficient interaction data to run offline evaluation splits.")
            print("Please seed or add more ratings first using 'python scripts/seed_db.py'.")
            return
            
        print("\nResults Table:")
        print("-" * 75)
        print(f"{'Algorithm':<25} | {'Precision@K':<12} | {'Recall@K':<12} | {'NDCG@K':<12}")
        print("-" * 75)
        
        for m in evaluation.metrics:
            print(f"{m.algorithm:<25} | {m.precision:<12.4f} | {m.recall:<12.4f} | {m.ndcg:<12.4f}")
            
        print("-" * 75)
        print("Note: Evaluated on test set target users who have relevant items in holdout.")
        
    except Exception as e:
        print(f"Error executing evaluation: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    evaluate_engines()
