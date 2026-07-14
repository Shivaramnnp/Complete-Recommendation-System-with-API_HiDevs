from fastapi import status

def test_add_rating(client, seed_test_data):
    # Submit rating for Bob (id=2) rating Lion King (id=3)
    response = client.post("/recommendations/ratings?user_id=2", json={
        "item_id": 3,
        "rating": 4.5
    })
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["user_id"] == 2
    assert data["item_id"] == 3
    assert data["rating"] == 4.5

def test_add_duplicate_rating(client, seed_test_data):
    # Alice (id=1) already rated Toy Story (id=1) in seed data
    response = client.post("/recommendations/ratings?user_id=1", json={
        "item_id": 1,
        "rating": 3.0
    })
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "already rated" in response.json()["detail"]

def test_get_recommendations(client, seed_test_data):
    # Get recommendations for Alice (id=1)
    response = client.get("/recommendations/users/1?algorithm=hybrid&k=2")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["user_id"] == 1
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 2
    assert data["algorithm_used"] == "hybrid"
    
    # Each recommendation item should contain hydrated item metadata
    for rec in data["recommendations"]:
        assert "item_id" in rec
        assert "score" in rec
        assert "item" in rec
        assert rec["item"] is not None
        assert "title" in rec["item"]

def test_evaluate_endpoint(client, seed_test_data):
    # Register 5 additional users (IDs 3, 4, 5, 6, 7)
    for idx in range(3, 8):
        client.post("/users/", json={"username": f"eval_user_{idx}"})
    # Register 5 additional items (IDs 4, 5, 6, 7, 8)
    for idx in range(4, 9):
        client.post("/items/", json={"title": f"eval_item_{idx}"})

    # Seed ratings from users 1-7 for items 1-8 to ensure > 20 distinct ratings
    rating_count = 0
    for u in range(1, 8):
        for i in range(1, 9):
            res = client.post(f"/recommendations/ratings?user_id={u}", json={
                "item_id": i,
                "rating": 4.0
            })
            if res.status_code == status.HTTP_201_CREATED:
                rating_count += 1
                
    assert rating_count >= 20

    response = client.get("/recommendations/evaluate?k=2&test_ratio=0.3")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["k"] == 2
    assert "metrics" in data

    # We should have metrics list for different models
    algorithms = {m["algorithm"] for m in data["metrics"]}
    assert "hybrid" in algorithms
    assert "user_cf" in algorithms
    assert "item_cf" in algorithms
    assert "content_based" in algorithms

    # Ensure scores are formatted correctly
    for m in data["metrics"]:
        assert "precision" in m
        assert "recall" in m
        assert "ndcg" in m
