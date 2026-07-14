from fastapi import status

def test_create_item(client):
    response = client.post("/items/", json={
        "title": "Toy Story",
        "genres": "Animation|Children|Comedy",
        "description": "A cowboy toy meets a spaceman toy."
    })
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["title"] == "Toy Story"
    assert data["genres"] == "Animation|Children|Comedy"
    assert "id" in data

def test_get_items(client):
    client.post("/items/", json={"title": "Item A"})
    client.post("/items/", json={"title": "Item B"})
    
    response = client.get("/items/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) >= 2
    titles = {item["title"] for item in data}
    assert "Item A" in titles
    assert "Item B" in titles

def test_get_item_by_id(client):
    res_create = client.post("/items/", json={"title": "Item C"})
    item_id = res_create.json()["id"]
    
    response = client.get(f"/items/{item_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["title"] == "Item C"

def test_get_nonexistent_item(client):
    response = client.get("/items/9999")
    assert response.status_code == status.HTTP_404_NOT_FOUND
