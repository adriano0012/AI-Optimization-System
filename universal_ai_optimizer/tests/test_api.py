import pytest
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.app import app, db_manager, auth_manager, optimizer_instance
from api.models import DatabaseManager
from api.auth import AuthManager


@pytest.fixture(scope="module")
def setup_db():
    test_db_url = "sqlite:///test_api.db"
    os.makedirs("data", exist_ok=True)
    dm = DatabaseManager(test_db_url)
    dm.create_tables()
    yield dm
    dm.drop_tables()
    try:
        os.remove("test_api.db")
    except OSError:
        pass


@pytest.fixture
def client(setup_db):
    from api.app import db_manager as db_ref, auth_manager as auth_ref
    import api.app as app_module
    app_module.db_manager = setup_db
    app_module.auth_manager = AuthManager(setup_db)

    from universal_ai_optimizer.core.optimizer import UniversalAIOptimizer
    app_module.optimizer_instance = UniversalAIOptimizer()

    return TestClient(app)


class TestHealth:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] in ['healthy', 'degraded']
        assert 'version' in data
        assert 'uptime_seconds' in data

    def test_readiness(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()['status'] == 'ready'

    def test_version(self, client):
        response = client.get("/version")
        assert response.status_code == 200
        assert 'version' in response.json()


class TestAuth:
    def test_register_user(self, client):
        response = client.post("/v1/auth/register", json={
            "email": "test@example.com",
            "password": "securepass123",
            "name": "Test User"
        })
        assert response.status_code == 200
        data = response.json()
        assert 'access_token' in data
        assert data['token_type'] == 'bearer'

    def test_register_duplicate_email(self, client):
        client.post("/v1/auth/register", json={
            "email": "dup@example.com",
            "password": "pass123456",
            "name": "Dup User"
        })
        response = client.post("/v1/auth/register", json={
            "email": "dup@example.com",
            "password": "pass123456",
            "name": "Dup User 2"
        })
        assert response.status_code == 409

    def test_login_success(self, client):
        client.post("/v1/auth/register", json={
            "email": "login@example.com",
            "password": "mypassword123",
            "name": "Login User"
        })
        response = client.post("/v1/auth/login", json={
            "email": "login@example.com",
            "password": "mypassword123"
        })
        assert response.status_code == 200
        assert 'access_token' in response.json()

    def test_login_wrong_password(self, client):
        client.post("/v1/auth/register", json={
            "email": "wrong@example.com",
            "password": "correctpass",
            "name": "Wrong User"
        })
        response = client.post("/v1/auth/login", json={
            "email": "wrong@example.com",
            "password": "wrongpass"
        })
        assert response.status_code == 401

    def test_create_api_key(self, client):
        reg = client.post("/v1/auth/register", json={
            "email": "apikey@example.com",
            "password": "keypass123",
            "name": "API Key User"
        })
        token = reg.json()['access_token']
        response = client.post("/v1/auth/api-keys",
            json={"name": "test-key", "rate_limit": 500},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert 'key' in data
        assert data['key'].startswith('uai_')

    def test_list_api_keys(self, client):
        reg = client.post("/v1/auth/register", json={
            "email": "listkeys@example.com",
            "password": "keypass123",
            "name": "List Keys User"
        })
        token = reg.json()['access_token']
        client.post("/v1/auth/api-keys",
            json={"name": "key1"},
            headers={"Authorization": f"Bearer {token}"}
        )
        response = client.get("/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert len(response.json()) >= 1


class TestOptimize:
    def test_optimize_basic(self, client):
        response = client.post("/v1/optimize", json={
            "prompt": "What is 2+2?",
            "context": {"task_type": "question_answering"}
        })
        assert response.status_code == 200
        data = response.json()
        assert data['original_prompt'] == "What is 2+2?"
        assert 'optimized_prompt' in data
        assert data['success'] is True

    def test_optimize_empty_prompt(self, client):
        response = client.post("/v1/optimize", json={
            "prompt": "",
            "context": {}
        })
        assert response.status_code == 422

    def test_optimize_code_generation(self, client):
        response = client.post("/v1/optimize", json={
            "prompt": "Write a function to sort a list in Python",
            "context": {"task_type": "code_generation"}
        })
        assert response.status_code == 200
        assert response.json()['success'] is True

    def test_optimize_stream(self, client):
        response = client.post("/v1/optimize/stream", json={
            "prompt": "Hello world",
            "context": {}
        })
        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/x-ndjson'


class TestOrganizations:
    def test_create_organization(self, client):
        reg = client.post("/v1/auth/register", json={
            "email": "orgadmin@example.com",
            "password": "orgpass123",
            "name": "Org Admin"
        })

        # Create token directly with admin role
        import api.app as app_module
        from api.auth import AuthManager
        token = app_module.auth_manager.create_jwt_token(
            user_id="admin-user", role="admin"
        )

        response = client.post("/v1/organizations",
            json={"name": "Test Org", "plan": "pro"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()['name'] == "Test Org"


class TestErrorHandling:
    def test_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_optimize_with_api_key(self, client):
        reg = client.post("/v1/auth/register", json={
            "email": "keyopt@example.com",
            "password": "pass123456",
            "name": "Key Opt User"
        })
        token = reg.json()['access_token']

        key_resp = client.post("/v1/auth/api-keys",
            json={"name": "opt-key"},
            headers={"Authorization": f"Bearer {token}"}
        )
        api_key = key_resp.json()['key']

        response = client.post("/v1/optimize",
            json={"prompt": "Test with API key", "context": {}},
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
