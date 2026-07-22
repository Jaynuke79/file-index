import pytest

from file_index.config import Config
from file_index.index import Index


@pytest.fixture
def tmp_env(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    cfg = Config()
    cfg.roots = [root]
    cfg.data_dir = tmp_path / "state"
    cfg.data_dir.mkdir()
    index = Index(cfg.db_path)
    yield cfg, index, root
    index.close()
