from engine.strategy import UserInteraction
from engine.collaborative import CollaborativeFiltering
from engine.content_based import ContentBasedFiltering
from engine.hybrid import HybridRecommendation

# Define standard interactions for testing
interactions = [
    # User 1 likes item 1 and 2
    UserInteraction(user_id=1, item_id=1, rating=5.0),
    UserInteraction(user_id=1, item_id=2, rating=4.5),
    
    # User 2 likes item 2 and 3
    UserInteraction(user_id=2, item_id=2, rating=4.0),
    UserInteraction(user_id=2, item_id=3, rating=5.0),
    
    # User 3 likes item 1
    UserInteraction(user_id=3, item_id=1, rating=5.0),
]

items_info = [
    {"item_id": 1, "genres": "Sci-Fi|Action"},
    {"item_id": 2, "genres": "Sci-Fi|Thriller"},
    {"item_id": 3, "genres": "Comedy"},
]

def test_collaborative_filtering_user():
    # User-based CF
    cf = CollaborativeFiltering(kind="user", similarity_metric="cosine")
    cf.fit(interactions)
    
    # User 3 has rated item 1.
    # Since User 1 also rated item 1, they are similar.
    # User 1 rated item 2, so User 3 should get a recommendation for item 2!
    recs = cf.recommend(user_id=3, k=2)
    assert len(recs) > 0
    assert recs[0].item_id == 2
    assert recs[0].score > 0.0

def test_collaborative_filtering_item():
    # Item-based CF
    cf = CollaborativeFiltering(kind="item", similarity_metric="cosine")
    cf.fit(interactions)
    
    # User 3 rated item 1. Item 1 and Item 2 are similar because User 1 rated both.
    # User 3 should get a recommendation for Item 2.
    recs = cf.recommend(user_id=3, k=2)
    assert len(recs) > 0
    assert recs[0].item_id == 2
    assert recs[0].score > 0.0

def test_content_based_filtering():
    cb = ContentBasedFiltering(feature_field="genres", sep="|")
    cb.fit(interactions, items_info)
    
    # User 3 rated Item 1 ("Sci-Fi|Action").
    # The other item with "Sci-Fi" is Item 2 ("Sci-Fi|Thriller").
    # Item 3 is "Comedy", which does not match.
    # User 3 should get Item 2 recommended higher than Item 3.
    recs = cb.recommend(user_id=3, k=2)
    assert len(recs) >= 2
    
    # Verify Item 2 is ranked above Item 3
    rec_ids = [r.item_id for r in recs]
    assert rec_ids[0] == 2
    assert rec_ids[1] == 3
    # Score for Item 2 should be greater than Item 3 (Item 3 is comedy, no genre overlap)
    assert recs[0].score > recs[1].score

def test_hybrid_recommendation():
    hybrid = HybridRecommendation(cf_weight=0.5, cb_weight=0.5)
    hybrid.fit(interactions, items_info)
    
    recs = hybrid.recommend(user_id=3, k=2)
    assert len(recs) > 0
    # Both sub-models should suggest item 2 for user 3
    assert recs[0].item_id == 2
