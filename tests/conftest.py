import pytest

import tiny


@pytest.fixture(scope="session")
def tiny_system():
    return tiny.build_system()


@pytest.fixture(scope="session")
def tiny_dirs(tmp_path_factory, tiny_system):
    """(artifacts_dir, gold_dir) of a small system, built once and treated as read-only."""
    return tiny.write_artifacts(tmp_path_factory.mktemp("tiny"), tiny_system)
