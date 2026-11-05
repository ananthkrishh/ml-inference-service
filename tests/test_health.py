from fastapi.testclient import TestClient

from app.main import app


def test_health_ok():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    # "ok" or "degraded" depending on whether a model/Ollama exist on this machine;
    # either way the API itself is alive, so it's 200.
    assert r.json()["status"] in {"ok", "degraded"}


def test_root_redirects_to_docs():
    with TestClient(app) as client:
        r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/docs"


def test_docs_page_hides_static_example_responses():
    with TestClient(app) as client:
        r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger-ui" in r.text
    assert "table.responses-table:not(.live-responses-table)" in r.text
