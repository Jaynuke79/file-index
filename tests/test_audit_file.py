"""The plain-text audit.log mirror appends each DB audit row exactly once."""

from file_index.cli import _append_audit_file


def _lines(cfg):
    if not cfg.audit_log_path.exists():
        return []
    return cfg.audit_log_path.read_text().splitlines()


def test_repeated_mirroring_appends_each_row_once(tmp_env):
    cfg, index, root = tmp_env
    index.audit("move", "/a/x", "/b/x", "grouping")
    index.audit("mkdir", None, "/b", "new folder")

    _append_audit_file(cfg, index)
    assert len(_lines(cfg)) == 2

    # second run with no new rows: nothing appended
    _append_audit_file(cfg, index)
    assert len(_lines(cfg)) == 2

    # a new row is appended exactly once, without repeating the old ones
    index.audit("rename", "/b/x", "/b/y", "clearer name")
    _append_audit_file(cfg, index)
    lines = _lines(cfg)
    assert len(lines) == 3
    assert sum("move /a/x" in ln for ln in lines) == 1
    assert "rename /b/x -> /b/y" in lines[-1]


def test_mirror_survives_reopen(tmp_env):
    from file_index.index import Index

    cfg, index, root = tmp_env
    index.audit("move", "/a/x", "/b/x", "grouping")
    _append_audit_file(cfg, index)

    index2 = Index(cfg.db_path)  # fresh connection, same db
    _append_audit_file(cfg, index2)  # cursor persisted: no duplicates
    assert len(_lines(cfg)) == 1
    index2.close()
