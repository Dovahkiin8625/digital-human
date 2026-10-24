"""Safe extraction of an uploaded Cubism project archive.

A zip is the only sane way to hand a browser a Cubism project -- it is a
dozen interlinked files -- but it is also attacker-controlled input. A
member can be named ``../../.env``, be a symlink aimed at the data dir, or
expand a few hundred KB into hundreds of MB. ``ZipFile.extractall`` checks
none of these before Python 3.12, so extraction here is member by member:
every write is validated against the destination root, and every bomb
metric is counted as it goes.
"""

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from server.core.errors import AppError

_BAD = "live2d_invalid_model"


def _fail(message: str) -> AppError:
    return AppError(_BAD, message, 422)


def _is_absolute(name: str) -> bool:
    """True when ``name`` names somewhere outside any destination.

    Both flavours are checked because the two spellings mean different
    things on different hosts: ``PurePosixPath("/etc/passwd")`` is
    absolute but ``PureWindowsPath("/etc/passwd")`` is not, while
    ``PureWindowsPath("C:/x")`` is absolute and ``PurePosixPath("C:/x")``
    is a perfectly legal relative name. An archive written on one OS must
    not be honoured on the other.
    """
    return PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute()


def _member_target(dest: Path, name: str) -> Path:
    """Resolve one member under ``dest``, or refuse.

    Three escapes, and ``is_relative_to`` alone catches only the last: a
    ``..`` segment, an absolute or drive-qualified name, and a name that
    lands on a symlink an earlier member planted. The first two are
    checked textually because on some hosts they do not even survive
    ``dest / name``.
    """
    if (
        not name
        or _is_absolute(name)
        or ".." in PurePosixPath(name).parts
        or ".." in PureWindowsPath(name).parts
    ):
        raise _fail(f"illegal path in archive: {name!r}")
    target = (dest / name).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise _fail(f"illegal path in archive: {name!r}")
    return target


def _reject_link(info: zipfile.ZipInfo) -> None:
    """Refuse a member that is a symbolic link.

    ``ZipFile.open`` on a link member yields whatever bytes the archiver
    stored after the link path, not the pointed-at file, so extraction
    alone does not read through it. The danger is a planted link that a
    LATER member writes through, and a member any later build step
    follows. Neither is worth carrying, and the check costs one read.
    """
    if stat.S_ISLNK(info.external_attr >> 16):
        raise _fail(f"archive contains a symbolic link: {info.filename!r}")


def extract_archive(
    src: Path,
    dest: Path,
    *,
    max_entries: int,
    max_unpacked_bytes: int,
    max_entry_bytes: int,
) -> None:
    """Extract ``src`` into ``dest``, refusing every unsafe shape.

    The declared uncompressed size is counted before anything is written,
    so a bomb is rejected without first filling the disk with it. The copy
    loop counts the bytes that actually arrive as well: the header is
    attacker-controlled, and CPython's ``ZipExtFile`` happens to truncate
    every read to the declared size -- but that is a property of one
    implementation, not the reason to be correct.

    On any failure the partially written tree is left in place for the
    caller to remove -- this function does not own ``dest``.
    """
    dest = Path(dest)
    try:
        with zipfile.ZipFile(src) as zf:
            members = zf.infolist()
            if len(members) > max_entries:
                raise _fail(f"archive has {len(members)} entries; limit is {max_entries}")

            declared_total = 0
            written_total = 0
            for info in members:
                _reject_link(info)
                if info.is_dir():
                    # Directory entries carry no bytes; parents are created
                    # on demand by the member writes below.
                    continue
                name = info.filename
                if info.file_size > max_entry_bytes:
                    raise _fail(
                        f"{name} unpacks to {info.file_size} bytes; "
                        f"per-file limit is {max_entry_bytes}"
                    )
                declared_total += info.file_size
                if declared_total > max_unpacked_bytes:
                    raise _fail(f"archive unpacks to more than {max_unpacked_bytes} bytes")

                target = _member_target(dest, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with zf.open(info) as member, target.open("wb") as out:
                    while True:
                        chunk = member.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_entry_bytes:
                            raise _fail(
                                f"{name} unpacks past the per-file limit of "
                                f"{max_entry_bytes} bytes"
                            )
                        out.write(chunk)
                written_total += written
                if written_total > max_unpacked_bytes:
                    raise _fail(f"archive unpacks to more than {max_unpacked_bytes} bytes")
    except zipfile.BadZipFile as exc:
        raise _fail(f"not a readable zip archive: {exc}") from exc


def find_entries(root: Path) -> list[str]:
    """Every model entry file in an extracted project, as sorted POSIX paths.

    Cubism exports ``<Name>.model3.json``; a hand-built project can be
    called plain ``model3.json``, so both spellings count.

    macOS archives add an ``__MACOSX/`` sibling tree of AppleDouble files
    named ``._<original>``. Those names still end in ``.json``, so
    without the filter they double the count and make a valid upload look
    ambiguous.
    """
    root = Path(root)
    candidates = {p for p in root.rglob("*.model3.json") if p.is_file()}
    candidates |= {p for p in root.rglob("model3.json") if p.is_file()}
    return sorted(
        p.relative_to(root).as_posix()
        for p in candidates
        if not p.name.startswith("._")
        and "__MACOSX" not in p.relative_to(root).parts
    )


def resolve_entry(root: Path, requested: str | None) -> str:
    """The ``model3.json`` to register, or an error naming the candidates.

    An operator who leaves the field blank gets auto-detection, which is
    what makes the common right-click-compress shape work: the folder is
    inside the zip, so the entry is ``Hiyori/model.model3.json`` and not
    the ``model.model3.json`` a hardcoded default would claim.
    """
    found = find_entries(root)
    if requested:
        return requested
    if len(found) == 1:
        return found[0]
    if not found:
        raise _fail("no *.model3.json in the archive; is this a Cubism project?")
    raise _fail(
        "the archive holds several models -- name one explicitly: " + ", ".join(found[:10])
    )
