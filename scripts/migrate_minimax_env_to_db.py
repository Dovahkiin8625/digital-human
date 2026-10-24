"""One-time migration: move the MiniMax TTS config out of .env and into SQLite.

The ``DH_MINIMAX_*`` env layer is gone -- MiniMax is configured from the
Models tab, which writes to the ``settings`` table. Boxes that upgraded
with the old vars still have them in their ``.env``, so this script
carries the values across so the synthesise path keeps working.

Usage::

    python scripts/migrate_minimax_env_to_db.py [--env-file .env] [--dry-run]

Rules:

* Only fills in keys the ``settings`` table does not already have -- a row
  the operator wrote through the console always wins, so re-running is
  safe and never clobbers a newer edit.
* Empty env values are skipped: an empty row and no row behave the same
  (``server.tts.config.DEFAULTS`` already covers them).
* Never prints a value. It reports the key names and whether each one
  was carried over.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: ``DH_*`` env var -> ``settings`` table key.
ENV_TO_DB_KEY: dict[str, str] = {
    "DH_MINIMAX_API_KEY": "minimax_api_key",
    "DH_MINIMAX_GROUP_ID": "minimax_group_id",
    "DH_MINIMAX_BASE_URL": "minimax_base_url",
    "DH_MINIMAX_MODEL": "minimax_model",
    "DH_MINIMAX_SAMPLE_RATE": "minimax_sample_rate",
}

#: Keys whose value is an integer rather than a string.
INT_KEYS: frozenset[str] = frozenset({"minimax_sample_rate"})


def read_env(path: Path) -> dict[str, str]:
    """Parse ``path`` for the ``DH_MINIMAX_*`` vars, if the file exists."""
    from dotenv import dotenv_values

    if not path.exists():
        return {}
    values = dotenv_values(path)
    return {k: v for k, v in values.items() if k in ENV_TO_DB_KEY and v}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be carried over without writing",
    )
    args = parser.parse_args(argv)

    env_path = Path(args.env_file)
    env_values = read_env(env_path)
    if not env_values:
        print(f"no DH_MINIMAX_* values in {env_path} -- nothing to migrate")
        return 0

    sys.path.insert(0, str(REPO_ROOT))
    from server.db.models import Setting
    from server.db.session import get_engine, init_db, reset_engine

    reset_engine()
    engine = get_engine()
    init_db(engine)
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=engine)()
    carried: list[str] = []
    skipped: list[str] = []
    try:
        for env_name, db_key in ENV_TO_DB_KEY.items():
            raw = env_values.get(env_name)
            if raw is None or raw == "":
                skipped.append(f"{db_key} (not set in .env)")
                continue
            if db.get(Setting, db_key) is not None:
                skipped.append(f"{db_key} (already in the settings table)")
                continue
            value: object = int(raw) if db_key in INT_KEYS else raw
            if args.dry_run:
                carried.append(f"{db_key} (would be carried over)")
                continue
            db.add(Setting(key=db_key, value=json.dumps(value)))
            carried.append(f"{db_key} (carried over)")
        if not args.dry_run:
            db.commit()
    finally:
        db.close()
        reset_engine()

    for line in carried:
        print(f"  {line}")
    for line in skipped:
        print(f"  {line}")
    if args.dry_run:
        print("\ndry run -- nothing written")
    else:
        print("\nMiniMax config now lives in the settings table.")
        print("Remove the DH_MINIMAX_* lines from .env when you're happy with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
