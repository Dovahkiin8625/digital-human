def test_healthz_ok(client):
    from server.engine.base import gpu_available

    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["engine"] == "mock"
    # The pin is the mock *engine*, but GPU presence is a property of the box,
    # not of the configuration: the same suite runs in the torch-free venv
    # (False) and the GPU venv (True). Assert agreement with the probe.
    assert body["gpu"] is gpu_available()
