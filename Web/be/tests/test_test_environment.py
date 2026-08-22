from app.core.config import settings


def test_test_environment_disables_external_persistence():
    assert settings.app_env == "test"
    assert settings.database_url is None
    assert settings.redis_url is None
