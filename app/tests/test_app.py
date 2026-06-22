from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_required_routes_are_registered() -> None:
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    expected = {
        "/imports/local-folder",
        "/assets",
        "/assets/{asset_id}",
        "/visual-units",
        "/visual-units/{visual_unit_id}",
        "/visual-units/{visual_unit_id}/produce",
        "/generation-jobs",
        "/outputs",
        "/outputs/{output_id}",
        "/outputs/{output_id}/qa",
        "/outputs/{output_id}/publish",
    }
    assert expected.issubset(paths)
