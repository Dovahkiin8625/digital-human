from sqlalchemy.engine import make_url

from server.config import get_settings
from server.db.session import get_engine


def _assert_engine_matches_settings():
    assert get_engine().url.database == make_url(get_settings().database_url).database


def test_first_client_gets_own_database(client):
    _assert_engine_matches_settings()


def test_second_client_gets_own_database(client):
    _assert_engine_matches_settings()
