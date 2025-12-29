from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_register_user():
    """Test user registration"""
    response = client.post(
        "/api/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"
    assert "password" not in data


def test_login_user():
    """Test user login"""
    # First register a user
    client.post(
        "/api/auth/register",
        json={
            "username": "logintest",
            "email": "logintest@example.com",
            "password": "testpassword123"
        }
    )

    # Then login (OAuth2PasswordRequestForm expects form data, not JSON)
    response = client.post(
        "/api/auth/login",
        data={
            "username": "logintest",
            "password": "testpassword123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_protected_endpoint_without_auth():
    """Test accessing protected endpoint without authentication"""
    response = client.get("/api/course-counts")
    assert response.status_code == 401
