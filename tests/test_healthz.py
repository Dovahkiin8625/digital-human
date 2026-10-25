def test_healthz_ok(client):
    from server.engine.base import gpu_available

    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # Every registered model is reported, because several may be resident
    # at once: an avatar names the model that drives it.
    by_name = {m["name"]: m for m in body["models"]}
    assert set(by_name) == {"musetalk", "live2d"}
    assert by_name["live2d"]["client_rendered"] is True
    assert by_name["musetalk"]["client_rendered"] is False
    # GPU presence is a property of the box, not of the configuration: the
    # same suite runs in the torch-free venv (False) and the GPU venv
    # (True). Assert agreement with the probe.
    assert body["gpu"] is gpu_available()
