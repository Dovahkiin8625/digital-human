"""Safe extraction of an uploaded Cubism project.

An avatar upload is operator input an attacker can shape: a zip can name a
member ``../../.env``, ship a symlink aimed at the data dir, or expand a
few hundred KB into hundreds of MB. These tests pin the boundary -- what
lands on disk, and what never does.
"""

import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

from server.core.errors import AppError
from server.live2d.archive import extract_archive, find_entries, resolve_entry

LIMITS = dict(max_entries=512, max_unpacked_bytes=512_000_000, max_entry_bytes=64_000_000)

MODEL = {
    "Version": 3,
    "FileReferences": {"Moc": "model.moc3", "Textures": ["texture_00.png"]},
}


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _archive(tmp_path: Path, entries: dict[str, bytes], name: str = "m.zip") -> Path:
    path = tmp_path / name
    path.write_bytes(_zip_bytes(entries))
    return path


def _flat_project(**extra: str) -> dict[str, bytes]:
    files = {
        "model.model3.json": json.dumps(MODEL).encode(),
        "model.moc3": b"\x00" * 16,
        "texture_00.png": b"\x89PNG",
    }
    files.update(extra)
    return files


def test_extracts_a_flat_project(tmp_path):
    src = _archive(tmp_path, _flat_project())
    dest = tmp_path / "out"
    extract_archive(src, dest, **LIMITS)
    assert (dest / "model.moc3").read_bytes() == b"\x00" * 16
    assert json.loads((dest / "model.model3.json").read_text())["Version"] == 3


def test_refuses_a_parent_escape(tmp_path):
    src = _archive(tmp_path, {**{f"ok{i}.bin": b"x" for i in range(3)}, "../../.env": b"secret"})
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **LIMITS)
    assert "illegal path" in exc.value.message
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("name", ["/etc/passwd", "C:\\Windows\\win.ini", "\\\\server\\share\\x"])
def test_refuses_an_absolute_name(tmp_path, name):
    """A zip written on one OS can carry a path the other one would honour.
    Whichever spelling arrives, only names that start under the destination
    are written."""
    src = _archive(tmp_path, {name: b"secret"})
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **LIMITS)
    assert "illegal path" in exc.value.message


def test_refuses_a_symlink(tmp_path):
    """A planted link is a write primitive for a LATER member, and a trap
    for anything that later follows it. One attribute read to refuse."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    src = tmp_path / "m.zip"
    src.write_bytes(buf.getvalue())
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **LIMITS)
    assert "symbolic link" in exc.value.message


def test_refuses_too_many_entries(tmp_path):
    entries = {f"f{i}.bin": b"x" for i in range(5)}
    src = _archive(tmp_path, entries)
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **{**LIMITS, "max_entries": 4})
    assert "5 entries" in exc.value.message


def test_refuses_an_oversized_total(tmp_path):
    """The whole point of the unpacked cap: 4 MB of zeros compresses to a
    few hundred bytes, so a 300 MB archive-size limit does nothing here."""
    src = _archive(tmp_path, {"bomb.bin": b"\0" * 4_000_000})
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **{**LIMITS, "max_unpacked_bytes": 1_000})
    assert "1000 bytes" in exc.value.message


def test_refuses_an_oversized_single_entry(tmp_path):
    src = _archive(tmp_path, {"big.bin": b"\0" * 2_000_000})
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **{**LIMITS, "max_entry_bytes": 1_000})
    assert "big.bin" in exc.value.message


def test_a_lying_header_cannot_inflate_the_archive(tmp_path):
    """The uncompressed-size field is attacker-controlled: a hostile zip
    can claim 8 bytes and hide a deflate bomb behind the claim.

    CPython truncates every read to the declared size, so the bomb never
    arrives; were a runtime to stop doing that, the byte counter in
    ``extract_archive`` is the backstop. The promise is the same either
    way, and it is the one that matters: what lands on disk stays under
    the per-file limit, whatever was behind the lie.
    """
    payload = b"\0" * 4_000_000
    raw = bytearray(_zip_bytes({"bomb.bin": payload}))
    # The local file header carries the size at 22; the central-directory
    # record ("PK\\x01\\x02") carries it at +24. Patch both so the header
    # and the directory agree on the smaller lie.
    raw[22:26] = (8).to_bytes(4, "little")
    cd = bytes(raw).index(b"PK\x01\x02")
    raw[cd + 24: cd + 28] = (8).to_bytes(4, "little")
    src = tmp_path / "m.zip"
    src.write_bytes(bytes(raw))

    dest = tmp_path / "out"
    limit = 1_000
    try:
        extract_archive(src, dest, **{**LIMITS, "max_entry_bytes": limit})
    except AppError:
        pass  # refused outright -- also acceptable
    written = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    assert written <= limit


def test_reports_a_body_that_is_not_a_zip(tmp_path):
    src = tmp_path / "m.zip"
    src.write_bytes(b"this is not a zip archive")
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **LIMITS)
    assert "not a readable zip" in exc.value.message


# ----- entry discovery ----------------------------------------------------

def test_finds_a_root_level_entry(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    for name, data in _flat_project().items():
        (root / name).write_bytes(data)
    assert find_entries(root) == ["model.model3.json"]


def test_finds_an_entry_inside_the_folder_a_zip_keeps(tmp_path):
    """Right-clicking Hiyori/ in Explorer zips the folder too, so the entry
    lands at Hiyori/model.model3.json. This is the common real-world shape;
    a hardcoded model.model3.json would fail on it."""
    root = tmp_path / "m"
    for name, data in _flat_project().items():
        target = root / "Hiyori" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert find_entries(root) == ["Hiyori/model.model3.json"]


def test_finds_a_bare_model3_json(tmp_path):
    """Cubism exports ``<Name>.model3.json``, but a hand-built project can
    be called plain ``model3.json``. The pattern needs a dot before
    "model3", so without this the file is there and the error says it is
    not."""
    root = tmp_path / "m"
    root.mkdir()
    (root / "model3.json").write_text("{}")
    assert find_entries(root) == ["model3.json"]


def test_ignores_macos_resource_forks(tmp_path):
    """Finder adds an __MACOSX/ tree of AppleDouble files named ``._<original>``.
    Those still end in .json, so unfiltered they double the candidate count
    and turn a valid upload into an ambiguous one."""
    root = tmp_path / "m"
    root.mkdir()
    for name, data in _flat_project().items():
        (root / name).write_bytes(data)
    macosx = root / "__MACOSX"
    macosx.mkdir()
    (macosx / "._model.model3.json").write_bytes(b"\x00\x05\x16\x07")
    assert find_entries(root) == ["model.model3.json"]


def test_resolve_entry_returns_an_explicit_choice(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "a.model3.json").write_text("{}")
    (root / "b.model3.json").write_text("{}")
    assert resolve_entry(root, "b.model3.json") == "b.model3.json"


def test_resolve_entry_names_the_candidates_when_ambiguous(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "a.model3.json").write_text("{}")
    (root / "b.model3.json").write_text("{}")
    with pytest.raises(AppError) as exc:
        resolve_entry(root, None)
    assert "a.model3.json" in exc.value.message
    assert "b.model3.json" in exc.value.message


def test_resolve_entry_reports_an_archive_with_no_model(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "readme.txt").write_text("hi")
    with pytest.raises(AppError) as exc:
        resolve_entry(root, None)
    assert "model3.json" in exc.value.message
