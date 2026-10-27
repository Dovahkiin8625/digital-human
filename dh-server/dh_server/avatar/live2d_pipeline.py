"""The Live2D avatar pipeline: a Cubism project *is* the avatar's pack.

A Live2D avatar is a zipped Cubism project. The pack this pipeline builds
is the extracted project tree plus the config that tells a client which
file is the entry point and which emotion tag maps to which expression.

There is no server-side engine for this model: the browser renders the
project itself, so the server only owns the files and hands them out.
That is why the build is cheap enough to run inside the upload request
(``inline_build``) instead of a background job -- there is no GPU work
and no long decode, just an extract and a validation.

The upload's two extra form fields (``entry`` / ``expressions``) arrive
as ``options``; every other model ignores them.
"""
from __future__ import annotations

import json
from pathlib import Path

from server.avatar.pipeline import AvatarPipeline
from server.config import get_settings
from server.core.errors import AppError
from server.live2d.archive import extract_archive, resolve_entry
from server.live2d.registry import validate_model_dir


def parse_expressions(raw: str | None) -> dict:
    """Parse the operator's ``{"smile": 0}`` map, or refuse it.

    The console rejects a malformed map before submitting, so a failure
    here means a hand-rolled client -- but it must still fail loudly
    rather than register a model whose emotion map silently does nothing.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise AppError(
            "live2d_invalid_config", f"expressions is not valid JSON: {exc}", 422
        ) from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
        for k, v in parsed.items()
    ):
        raise AppError(
            "live2d_invalid_config",
            'expressions must be an object of {"tag": index}',
            422,
        )
    return parsed


class Live2DAvatarPipeline(AvatarPipeline):
    """Extract + validate a Cubism project archive.

    ``process`` writes the project tree into ``out_dir`` (the avatar's own
    directory, which is what the model-host route serves from) and returns
    the config block. On any failure it raises before returning, and the
    caller removes the directory -- a rejected upload must leave neither
    files nor a row behind.
    """

    # Nothing here touches the GPU and nothing decodes a video, so the
    # build can finish inside the upload request.
    inline_build = True

    def process(
        self,
        avatar_id: str,
        source_path: Path,
        out_dir: Path,
        progress,
        *,
        options: dict | None = None,
    ) -> dict:
        settings = get_settings()
        options = options or {}
        out_dir = Path(out_dir)

        progress("extract", 30)
        extract_archive(
            Path(source_path),
            out_dir,
            max_entries=settings.live2d_max_archive_entries,
            max_unpacked_bytes=settings.live2d_max_unpacked_bytes,
            max_entry_bytes=settings.live2d_max_entry_bytes,
        )

        progress("validate", 80)
        config: dict = {"expressions": parse_expressions(options.get("expressions"))}
        entry = options.get("entry") or None
        config["model"] = {"entry": resolve_entry(out_dir, entry)}
        validate_model_dir(out_dir, config)
        return config
