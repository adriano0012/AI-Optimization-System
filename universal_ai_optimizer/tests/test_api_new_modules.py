"""
Tests for API endpoints: Webhooks, Models, Experiments, Prompts, SLA, WebSocket.
"""

import pytest
from api.app import app
from api.models import DatabaseManager
import os


@pytest.fixture(scope="module")
def client():
    from httpx import AsyncClient
    import asyncio

    os.makedirs("data", exist_ok=True)
    from api.app import db_manager
    if db_manager:
        db_manager.drop_tables()
        db_manager.create_tables()

    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_token(client):
    import secrets
    unique = secrets.token_hex(6)
    response = client.post("/v1/auth/register", json={
        "email": f"newmod_{unique}@test.com", "password": "testpass123", "name": "Test"
    })
    if response.status_code == 200:
        return response.json()['access_token']
    # If registration fails (e.g. already exists), try login
    response = client.post("/v1/auth/login", json={
        "email": f"newmod_{unique}@test.com", "password": "testpass123"
    })
    return response.json()['access_token']


@pytest.fixture(scope="module")
def admin_token():
    import api.app as app_module
    return app_module.auth_manager.create_jwt_token(user_id="admin-uid", role="admin")


class TestWebhooks:
    def test_create_webhook(self, client, auth_token):
        response = client.post("/v1/webhooks",
            json={"url": "http://example.com/hook", "events": ["optimization.completed"]},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['id'].startswith('wh-')
        assert data['status'] == 'active'

    def test_list_webhooks(self, client, auth_token):
        response = client.get("/v1/webhooks",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_delete_webhook(self, client, auth_token):
        create = client.post("/v1/webhooks",
            json={"url": "http://del.example.com", "events": ["test"]},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        wh_id = create.json()['id']
        response = client.delete(f"/v1/webhooks/{wh_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert response.json()['deleted'] is True


class TestModelRegistry:
    def test_register_model(self, client, auth_token):
        response = client.post("/v1/models",
            json={"name": "gpt-4-custom", "description": "Custom GPT", "version": "1.0.0"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert response.json()['name'] == 'gpt-4-custom'

    def test_list_models(self, client, auth_token):
        response = client.get("/v1/models",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_promote_model(self, client, admin_token):
        create = client.post("/v1/models",
            json={"name": "promote-test", "version": "1.0.0"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        model_id = create.json()['model_id']
        response = client.post(f"/v1/models/{model_id}/promote",
            json={"version": "1.0.0", "stage": "production"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        assert response.json()['stage'] == 'production'


class TestExperiments:
    def test_create_experiment(self, client, auth_token):
        response = client.post("/v1/experiments",
            json={
                "name": "Test Exp", "description": "Testing",
                "variants": [{"name": "control"}, {"name": "treatment"}],
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert response.json()['name'] == 'Test Exp'

    def test_list_experiments(self, client, auth_token):
        response = client.get("/v1/experiments",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    def test_start_experiment(self, client, admin_token):
        create = client.post("/v1/experiments",
            json={"name": "Start Test", "variants": [{"name": "a"}, {"name": "b"}]},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        exp_id = create.json()['experiment_id']
        response = client.post(f"/v1/experiments/{exp_id}/start",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        assert response.json()['status'] == 'running'


class TestPrompts:
    def test_create_prompt(self, client, auth_token):
        response = client.post("/v1/prompts",
            json={"name": "greeting", "content": "Hello {{name}}!"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert response.json()['name'] == 'greeting'

    def test_list_prompts(self, client, auth_token):
        response = client.get("/v1/prompts",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    def test_create_version(self, client, auth_token):
        create = client.post("/v1/prompts",
            json={"name": "version-test", "content": "v1"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        prompt_id = create.json()['prompt_id']
        response = client.post(f"/v1/prompts/{prompt_id}/versions",
            json={"content": "v2", "changelog": "Updated"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert response.json()['version'] == 2

    def test_render_prompt(self, client, auth_token):
        create = client.post("/v1/prompts",
            json={"name": "render-test", "content": "Hello {{name}}!"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        prompt_id = create.json()['prompt_id']
        response = client.post(f"/v1/prompts/{prompt_id}/render",
            json={"variables": {"name": "World"}},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert response.json()['content'] == 'Hello World!'


class TestSLA:
    def test_create_sla(self, client, admin_token):
        response = client.post("/v1/slas",
            json={
                "name": "Latency SLA", "description": "Avg latency < 500ms",
                "metric": "latency_ms", "target_value": 500.0, "operator": "<=",
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        assert response.json()['name'] == 'Latency SLA'

    def test_list_slas(self, client, auth_token):
        response = client.get("/v1/slas",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    def test_sla_dashboard(self, client, auth_token):
        response = client.get("/v1/slas/dashboard",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    def test_sla_alerts(self, client, auth_token):
        response = client.get("/v1/slas/alerts",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200


class TestAdminStats:
    def test_rate_limiter_stats(self, client, admin_token):
        response = client.get("/v1/admin/rate-limiter/stats",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200

    def test_request_logger_stats(self, client, admin_token):
        response = client.get("/v1/admin/request-logger/stats",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
