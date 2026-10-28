"""Guards on the values that reach the filesystem.

Two of them, both about the same thing: an id or a file name that comes
from a request ends up as a path, and a path that comes from a request can
point anywhere.

* :func:`require_uuid_hex` -- the avatar ids this service hands out are
  ``uuid4().hex``, so anything else is not one of ours and is refused
  with the same 404 a missing avatar gets. That keeps the endpoint from
  being a probe for which ids exist.
* :func:`derive_suffix` -- the extension of an uploaded file becomes part
  of a file name, so a separator or a ``..`` in it is refused.
"""
import re

from dh_server.core.errors import AppError

_UUID_HEX = re.compile(r"\A[0-9a-f]{32}\Z")


def require_uuid_hex(value: str) -> str:
    """Return ``value`` if it is one of our ids, or 404."""
    if not _UUID_HEX.match(value):
        raise AppError("not_found", f"avatar {value} not found", 404)
    return value


def derive_suffix(filename: str | None) -> str:
    """The extension to keep for an uploaded file.

    ``None`` and a name without a dot both mean "png", which is what the
    photo avatars are. The suffix becomes part of a file name under the
    avatar's directory, so a separator or a ``..`` in it is refused.
    """
    ext = filename.rsplit(".", 1)[-1] if filename and "." in filename else "png"
    suffix = "." + ext
    if any(sep in suffix for sep in ("/", "\\")) or ".." in suffix:
        raise AppError("invalid_filename", "filename must not contain path separators or '..'", 400)
    return suffix
