# test_app.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_generate_caption():
    # Send an image file to the endpoint
    with open("test_image.jpg", "rb") as img_file:
        response = client.post("/caption/", files={"image": img_file})
    assert response.status_code == 200
    assert "caption" in response.json()
