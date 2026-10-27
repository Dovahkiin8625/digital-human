"""``python run.py`` is the backend's entry point; these pin what it guarantees.

The launcher exists so that starting the backend is one command that cannot
be run from the wrong directory with the wrong environment. Three of those
guarantees are worth a test:

* the working directory is pinned to the repository root, so ``.env``,
  ``data/`` and ``dh-admin-ui/dist`` resolve the same way no matter where the
  command was typed;
* uvicorn is handed the app *factory* rather than an app object, which is
  what makes ``--reload`` work -- uvicorn re-imports the string in the
  reloader process;
* what this box cannot do (torch, CUDA, the MuseTalk weights, the built
  console) is reported before the port is taken, and a missing dependency is
  reported as a message instead of an ImportError traceback.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ``run.py`` lives in the project directory, not in the installed
# ``dh_server`` package, so the directory has to be importable however
# pytest was invoked.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run  # noqa: E402


@pytest.fixture()
def no_uvicorn(monkeypatch):
    """Replace uvicorn.run so main() starts nothing, and record the call."""
    calls: list[tuple[tuple, dict]] = []

    import uvicorn

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return calls


@pytest.fixture()
def clean_settings():
    """The settings cache is process-wide; do not leak this test's env."""
    from dh_server.config import get_settings

    yield
    get_settings.cache_clear()


def test_defaults_match_the_readme_incantation():
    args = run.parse_args([])
    assert (args.host, args.port) == ("0.0.0.0", 8000)
    assert args.reload is False
    assert args.log_level == "info"


def test_flags_override_the_defaults():
    args = run.parse_args(
        ["--host", "127.0.0.1", "--port", "8099", "--reload", "--log-level", "debug"]
    )
    assert (args.host, args.port) == ("127.0.0.1", 8099)
    assert args.reload is True
    assert args.log_level == "debug"


def test_unknown_log_level_is_rejected():
    with pytest.raises(SystemExit):
        run.parse_args(["--log-level", "chatty"])


def test_main_pins_the_project_root_and_serves_the_factory(
    monkeypatch, tmp_path, no_uvicorn, clean_settings
):
    from env import apply_env

    apply_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)  # typed from somewhere else entirely
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert run.main([]) == 0

    assert os.getcwd() == str(run.ROOT)
    # The package is importable, not the repository root: ``dh_server``
    # lives in the project directory.
    assert str(run.PROJECT_ROOT) in sys.path

    assert len(no_uvicorn) == 1
    args, kwargs = no_uvicorn[0]
    # The import string, not the app object: uvicorn re-imports it in the
    # reloader process, which is what --reload needs.
    assert args == ("dh_server.main:create_app",)
    assert kwargs["factory"] is True
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000
    assert kwargs["reload"] is False


def test_reload_watches_only_the_server_package(
    monkeypatch, tmp_path, no_uvicorn, clean_settings
):
    from env import apply_env

    apply_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)

    assert run.main(["--reload", "--port", "8099"]) == 0

    _, kwargs = no_uvicorn[0]
    assert kwargs["reload"] is True
    assert kwargs["reload_dirs"] == [str(run.PROJECT_ROOT / "dh_server")]
    assert kwargs["port"] == 8099


def test_main_refuses_to_start_without_its_dependencies(
    monkeypatch, tmp_path, capsys, no_uvicorn
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "CORE_MODULES", ("definitely_not_installed_xyz",))

    assert run.main([]) == 1

    err = capsys.readouterr().err
    assert "definitely_not_installed_xyz" in err
    # The message has to name the venv that does have the dependencies.
    assert ".venv-musetalk" in err
    assert not no_uvicorn


def test_preflight_names_the_missing_weights(monkeypatch, tmp_path, clean_settings):
    from env import apply_env

    apply_env(monkeypatch, tmp_path)

    data_dir, _ = run._resolve_data_dir()
    warnings, _ = run._preflight(data_dir)

    joined = "\n".join(warnings)
    assert "fetch_musetalk_weights.sh" in joined


def test_preflight_gpu_agrees_with_the_probe(monkeypatch, tmp_path, clean_settings):
    from env import apply_env
    from dh_server.engine.base import gpu_available

    apply_env(monkeypatch, tmp_path)

    data_dir, _ = run._resolve_data_dir()
    _, gpu_name = run._preflight(data_dir)

    # Same probe the server uses for /healthz: the launcher must not claim a
    # GPU the server will not report.
    if gpu_available():
        assert gpu_name
    else:
        assert gpu_name is None
