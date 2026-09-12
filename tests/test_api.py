import io

from PIL import Image


def test_health_reports_model(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"]["loaded"] is True


def test_analyze_and_case_lifecycle(client, sample_jpeg_bytes):
    r = client.post("/api/analyze", files={"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ("id", "verdict", "band", "probability_ai", "threshold", "cues", "regions", "explanation", "provenance", "heatmap_url", "image_url"):
        assert k in d
    assert d["verdict"] in ("real", "ai_generated")
    aid = d["id"]

    assert client.get(f"/api/analyses/{aid}").status_code == 200
    assert client.get(f"/api/analyses/{aid}/image").status_code == 200
    assert client.get(f"/api/analyses/{aid}/heatmap").status_code == 200
    lst = client.get("/api/analyses?limit=10").json()
    assert lst["total"] >= 1 and any(i["id"] == aid for i in lst["items"])

    rob = client.post(f"/api/analyses/{aid}/robustness")
    assert rob.status_code == 200
    assert rob.json()["stability"] in ("stable", "moderate", "fragile")
    assert len(rob.json()["items"]) >= 5

    rev = client.patch(f"/api/analyses/{aid}/review", json={"decision": "confirmed_real", "note": "test"})
    assert rev.status_code == 200 and rev.json()["review_decision"] == "confirmed_real"
    assert client.patch(f"/api/analyses/{aid}/review", json={"decision": "bogus"}).status_code == 422

    html = client.get(f"/api/analyses/{aid}/report")
    assert html.status_code == 200 and "EVIDENCE REPORT" in html.text and "sample.jpg" in html.text
    js = client.get(f"/api/analyses/{aid}/report?format=json")
    assert js.status_code == 200 and js.json()["report"]["id"] == aid

    stats = client.get("/api/stats").json()
    assert stats["total"] >= 1

    assert client.delete(f"/api/analyses/{aid}").status_code == 204
    assert client.get(f"/api/analyses/{aid}").status_code == 404


def test_batch_with_mixed_files(client, sample_jpeg_bytes):
    files = [("files", ("a.jpg", sample_jpeg_bytes, "image/jpeg")), ("files", ("b.jpg", sample_jpeg_bytes, "image/jpeg")), ("files", ("bad.txt", b"not an image", "text/plain"))]
    r = client.post("/api/analyze/batch", files=files, data={"name": "unit"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["batch"]["n_items"] == 2 and len(d["errors"]) == 1
    b = client.get(f"/api/batches/{d['batch']['id']}")
    assert b.status_code == 200 and len(b.json()["items"]) == 2
    assert client.get("/api/batches").json()["items"]


def test_rejects_invalid_uploads(client):
    assert client.post("/api/analyze", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/api/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")}).status_code == 400
    tiny = io.BytesIO(); Image.new("RGB", (4, 4)).save(tiny, "PNG")
    assert client.post("/api/analyze", files={"file": ("tiny.png", tiny.getvalue(), "image/png")}).status_code == 400
    assert client.get("/api/analyses/doesnotexist").status_code == 404


def test_model_card_endpoint(client):
    r = client.get("/api/model")
    assert r.status_code == 200
    body = r.json()
    assert "protocol" in body and body["protocol"]["held_out_generators"]
