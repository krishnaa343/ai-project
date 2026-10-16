import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app("development")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_check_returns_ok(client):
    """Verify that GET /api/health returns HTTP 200 and status 'ok'."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"

