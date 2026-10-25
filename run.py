"""Start the digital-human backend with one command: ``python run.py``.

This is the README's uvicorn line (``python -m uvicorn --factory
server.main:create_app --host 0.0.0.0 --port 8000``) with the easy-to-get-wrong
parts already done:

* the working directory is pinned to the project root, so ``.env``, ``data/``
  and ``admin-ui/dist`` resolve the same way no matter where the command was
  invoked from;
* ``server`` is importable even when Python was started somewhere else;
* what this particular box can and cannot do -- no torch means no MuseTalk
  runtime, no CUDA means no GPU, missing weights means no avatar pipeline, a
  missing ``admin-ui/dist`` means no console -- is reported *before* uvicorn
  takes the port, instead of surfacing as a failed avatar job minutes later.

The interpreter that runs this file is the interpreter that runs the server.
The MuseTalk runtime lives in ``.venv-musetalk`` (see the README), so activate
that environment first; this launcher says so out loud when the current
interpreter cannot load torch.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

#: The project root -- the directory this file lives in.
ROOT = Path(__file__).resolve().parent

#: The venv that carries the GPU runtime, and the torch-free one for tests.
GPU_VENV = ".venv-musetalk"
CPU_VENV = ".venv"

#: Everything the server imports unconditionally. Checked by import machinery
#: (not by importing) so a broken environment is reported as such instead of
#: as an ImportError traceback from deep inside the app factory.
CORE_MODULES = (
    "uvicorn",
    "fastapi",
    "sqlalchemy",
    "pydantic_settings",
    "aiortc",
    "numpy",
    "PIL",
    "httpx",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="启动数字人服务（FastAPI + aiortc）。",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址（默认：%(default)s）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口（默认：%(default)s）",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="server/ 源码变化时自动重启（仅开发用）",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="uvicorn 日志级别（默认：%(default)s）",
    )
    return parser.parse_args(argv)


def _missing_dependencies() -> list[str]:
    """Core modules this interpreter cannot provide."""
    missing = []
    for name in CORE_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def _dependency_error(missing: list[str]) -> str:
    return (
        "后端依赖未安装（缺少："
        + ", ".join(missing)
        + "）。\n"
        "请先激活一个运行环境，再在项目根目录执行 pip install -e \".[dev]\"：\n"
        f"  {GPU_VENV}\\Scripts\\activate    # 跑真实 MuseTalk（需要 GPU + 权重）\n"
        f"  {CPU_VENV}\\Scripts\\activate        # 只跑后台/单测（无 GPU）\n"
        "Git Bash 里用 source .venv-musetalk/Scripts/activate。"
    )


def _resolve_data_dir() -> tuple[Path, str | None]:
    """The configured data dir, and why it could not be read (if it could not)."""
    try:
        from server.config import get_settings

        data_dir = Path(get_settings().data_dir)
    except Exception as exc:  # noqa: BLE001 -- a broken config must not stop the server
        return ROOT / "data", f"读取配置失败：{exc}"
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    return data_dir, None


def _preflight(data_dir: Path) -> tuple[list[str], str | None]:
    """Warn about what this box cannot do. Returns (warnings, gpu name)."""
    warnings: list[str] = []
    gpu_name: str | None = None

    if not (ROOT / ".env").exists():
        warnings.append("没有 .env，正在使用内置默认配置（可从 .env.example 复制一份）")

    # MuseTalk needs torch and a CUDA device; without either the avatar
    # pipeline cannot run, but Live2D avatars (client-rendered) still can.
    if importlib.util.find_spec("torch") is None:
        warnings.append(
            f"当前解释器没有 torch，MuseTalk 运行时不可用（Live2D 形象不受影响）。"
            f"跑真实模型请用 {GPU_VENV}：source {GPU_VENV}/Scripts/activate"
        )
    else:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
            else:
                warnings.append("没有可用的 CUDA 设备，MuseTalk 会退回 CPU，帧率不可用")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"torch 探测失败：{exc}")

    # The weights the engine loads; the same list the engine checks itself,
    # so the two cannot disagree about whether the runtime is loadable.
    try:
        from server.avatar.musetalk_pipeline import RUNTIME_WEIGHTS

        weights_dir = data_dir / "models" / "musetalk"
        absent = [w for w in RUNTIME_WEIGHTS if not (weights_dir / w).exists()]
        if absent:
            warnings.append(
                f"MuseTalk 权重不完整（{weights_dir}）：缺少 {', '.join(absent)}；"
                "运行 scripts/fetch_musetalk_weights.sh 补齐"
            )
    except Exception:  # noqa: BLE001 -- the weights list is a nicety, not a gate
        pass

    # The lite (FeatherTalk) engine's weights are small and optional: the
    # model is only usable if they were fetched, and the server boots fine
    # without them.
    try:
        from server.lite import vendor

        absent = vendor.missing_weights()
        if absent:
            warnings.append(
                f"轻量引擎权重不完整（{vendor.weights_dir()}）：缺少 {', '.join(absent)}；"
                "运行 scripts/fetch_lite_weights.sh 补齐（否则 lite 模型不可用）"
            )
    except Exception:  # noqa: BLE001 -- a nicety, not a gate
        pass

    if not (ROOT / "admin-ui" / "dist" / "index.html").exists():
        warnings.append("管理台尚未构建（admin-ui/dist/index.html 不存在）：cd admin-ui && npm run build")

    return warnings, gpu_name


def _print_banner(args: argparse.Namespace, data_dir: Path, gpu_name: str | None,
                  warnings: list[str]) -> None:
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    base = f"http://{shown_host}:{args.port}"
    print()
    print("=" * 68)
    print(" 数字人服务 (digital-human)")
    print(f" 解释器   : {sys.executable}")
    print(f" 地址     : {base}")
    print(f" 管理台   : {base}/admin")
    print(f" 健康检查 : {base}/healthz")
    print(f" 数据目录 : {data_dir}")
    if gpu_name:
        print(f" GPU      : {gpu_name}")
    print("=" * 68)
    for warning in warnings:
        print(f" ! {warning}")
    if warnings:
        print("=" * 68)
    print()
    # uvicorn logs to stderr, which is unbuffered; stdout is not when it is
    # redirected to a file. Flush so the banner keeps its place in the log.
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # A console that cannot encode the Chinese banner must not turn into a
    # UnicodeEncodeError before the server ever starts: replace what does not
    # fit instead of raising.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover -- exotic streams
            pass

    # Pin the working directory to the project root: the data dir and the
    # database URL are relative ("data/"), .env is read from the working
    # directory, and the console is mounted from admin-ui/dist. Running from
    # anywhere else would silently point all three somewhere else.
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    missing = _missing_dependencies()
    if missing:
        print(_dependency_error(missing), file=sys.stderr)
        return 1

    data_dir, config_error = _resolve_data_dir()
    warnings, gpu_name = _preflight(data_dir)
    if config_error:
        warnings.insert(0, config_error)
    _print_banner(args, data_dir, gpu_name, warnings)

    import uvicorn

    uvicorn.run(
        # The import string (rather than the app object) is what makes
        # --reload work: uvicorn re-imports it in the reloader process.
        "server.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        # The default is the whole working directory, which includes data/
        # and admin-ui/node_modules; only the server package changes here.
        reload_dirs=[str(ROOT / "server")] if args.reload else None,
    )
    return 0


if __name__ == "__main__":
    # Also the guard uvicorn's reloader needs: on Windows it re-imports this
    # module in the child process, and only the parent may start a server.
    raise SystemExit(main())
