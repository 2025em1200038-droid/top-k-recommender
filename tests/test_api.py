import pytest
from fastapi.testclient import TestClient

from serving.app import create_app


@pytest.fixture(scope="module")
def client(tiny_dirs):
    with TestClient(create_app(*tiny_dirs)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["n_users"] == 3
    assert body["genres"] == ["Action", "Drama"] and body["model_version"]


def test_predict_returns_ranked_recommendations(client):
    response = client.post("/predict", json={"user_id": 11, "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 11 and body["k"] == 2 and body["latency_ms"] >= 0
    assert [r["rank"] for r in body["recommendations"]] == [1, 2]
    assert {r["movie_id"] for r in body["recommendations"]} == {104, 105}


def test_predict_with_genre(client):
    body = client.post("/predict", json={"user_id": 11, "k": 2, "genre": "Drama"}).json()
    assert body["genre"] == "Drama"
    assert {r["movie_id"] for r in body["recommendations"]} <= {103, 104, 105}


def test_unknown_user_is_400(client):
    response = client.post("/predict", json={"user_id": 424242, "k": 2})
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "bad_request" and "unknown user" in body["message"]
    assert body["details"] == [{"field": "user_id", "message": "unknown user id", "value": 424242}]


def test_invalid_genre_is_400(client):
    response = client.post("/predict", json={"user_id": 11, "k": 2, "genre": "Nope"})
    assert response.status_code == 400
    detail = response.json()["details"][0]
    assert (detail["field"], detail["value"]) == ("genre", "Nope")
    assert "valid genres" in detail["message"]


def test_k_above_what_the_model_retrieves_is_400(client):
    # The tiny model retrieves 2 candidates; the schema alone allows up to 100.
    response = client.post("/predict", json={"user_id": 11, "k": 3})
    assert response.status_code == 400
    detail = response.json()["details"][0]
    assert (detail["field"], detail["value"]) == ("k", 3)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"k": 2},
        {"user_id": 0},
        {"user_id": "11"},
        {"user_id": 11.5},
        {"user_id": True},
        {"user_id": None},
        {"user_id": 11, "k": 0},
        {"user_id": 11, "k": 101},
        {"user_id": 11, "k": "2"},
        {"user_id": 11, "genre": ""},
        {"user_id": 11, "genre": 5},
        {"user_id": 11, "unexpected": 1},
        [],
        "text",
    ],
)
def test_malformed_payloads_are_422(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request" and body["details"]


def test_422_names_the_failing_field_and_value(client):
    body = client.post("/predict", json={"user_id": "abc", "k": 500, "genre": "x" * 300}).json()
    by_field = {d["field"]: d for d in body["details"]}
    assert by_field["user_id"]["value"] == "abc"
    assert by_field["k"]["value"] == 500
    assert len(by_field["genre"]["value"]) < 120  # long values are truncated
    missing = client.post("/predict", json={}).json()["details"][0]
    assert (missing["field"], missing["value"]) == ("user_id", None)


def test_non_finite_numbers_do_not_break_the_error_response(client):
    for raw in (b'{"user_id": NaN}', b'{"user_id": Infinity}', b'{"user_id": 11, "k": -Infinity}'):
        response = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert all(isinstance(d["value"], (str, int, float, type(None))) for d in response.json()["details"])


def test_reranker_failure_returns_200_with_degraded_flag(tiny_dirs, monkeypatch):
    app = create_app(*tiny_dirs)
    with TestClient(app) as c:
        def boom(*args, **kwargs):
            raise RuntimeError("scoring failed")

        monkeypatch.setattr(app.state.recommender._two_tier, "scores", boom)
        response = c.post("/predict", json={"user_id": 11, "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True and len(body["recommendations"]) == 2


def test_malformed_json_is_422_and_wrong_method_is_405(client):
    bad = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert bad.status_code == 422
    wrong = client.get("/predict")
    assert wrong.status_code == 405 and wrong.json()["error"] == "method_not_allowed"
    missing = client.get("/nope")
    assert missing.status_code == 404 and missing.json()["error"] == "not_found"


def test_unexpected_failure_is_500_without_internals(tiny_dirs, monkeypatch):
    app = create_app(*tiny_dirs)
    with TestClient(app, raise_server_exceptions=False) as c:
        def boom(*args, **kwargs):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(app.state.recommender, "recommend", boom)
        response = c.post("/predict", json={"user_id": 11, "k": 2})
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error", "message": "internal server error", "details": []}


def test_missing_artifacts_give_500_but_the_process_stays_up(tmp_path):
    with TestClient(create_app(tmp_path / "missing", tmp_path / "gold")) as c:
        assert c.get("/health").status_code == 500
        response = c.post("/predict", json={"user_id": 11})
        assert response.status_code == 500 and "model unavailable" in response.json()["message"]
