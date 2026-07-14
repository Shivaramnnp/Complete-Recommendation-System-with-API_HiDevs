from fastapi import status

def test_create_user(client):
    response = client.post("/users/", json={"username": "newuser"})
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["username"] == "newuser"
    assert "id" in data

def test_create_duplicate_username(client):
    # Register once
    client.post("/users/", json={"username": "testuser"})
    
    # Register again with same name
    response = client.post("/users/", json={"username": "testuser"})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Username already registered" in response.json()["detail"]

def test_get_users(client):
    client.post("/users/", json={"username": "user1"})
    client.post("/users/", json={"username": "user2"})
    
    response = client.get("/users/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) >= 2
    usernames = {user["username"] for user in data}
    assert "user1" in usernames
    assert "user2" in usernames

def test_get_user_by_id(client):
    res_create = client.post("/users/", json={"username": "user3"})
    user_id = res_create.json()["id"]
    
    response = client.get(f"/users/{user_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["username"] == "user3"

def test_get_nonexistent_user(client):
    response = client.get("/users/9999")
    assert response.status_code == status.HTTP_404_NOT_FOUND
