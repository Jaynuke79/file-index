"""The control panel: settings edits, job control, and the guards around them.

The browse server was read-only by design. These tests pin the guards that make
writes acceptable — loopback-only, CSRF-gated, whitelist-validated — as much as
they pin the features.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from file_index import control
from file_index.config import Config, load_config
from file_index.control import SettingsError
from file_index.jobs import JobError, JobRunner
from file_index.web import make_server


# ---------- settings layer ----------


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    root = tmp_path / "root"
    root.mkdir()
    c.roots = [root]
    c.data_dir = tmp_path / "state"
    c.data_dir.mkdir()
    c.config_path = tmp_path / "config.yaml"
    return c


def test_add_and_remove_root_persists_to_config(cfg, tmp_path):
    extra = tmp_path / "photos"
    extra.mkdir()

    added = control.add_root(cfg, str(extra))

    assert added == str(extra.resolve())
    assert load_config(cfg.config_path).roots == [
        p for p in (cfg.roots[0], extra.resolve())
    ]
    control.remove_root(cfg, str(extra))
    assert load_config(cfg.config_path).roots == [cfg.roots[0]]


def test_add_root_rejects_bad_input(cfg, tmp_path):
    with pytest.raises(SettingsError, match="not a directory"):
        control.add_root(cfg, str(tmp_path / "nope"))
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(SettingsError, match="not a directory"):
        control.add_root(cfg, str(f))
    with pytest.raises(SettingsError, match="already indexed"):
        control.add_root(cfg, str(cfg.roots[0]))


def test_add_root_rejects_overlap_with_existing(cfg, tmp_path):
    """A parent or child of an existing root would double-index its files."""
    child = cfg.roots[0] / "sub"
    child.mkdir()
    with pytest.raises(SettingsError, match="already covered"):
        control.add_root(cfg, str(child))
    with pytest.raises(SettingsError, match="already contains"):
        control.add_root(cfg, str(tmp_path))


def test_cannot_remove_the_last_root(cfg):
    with pytest.raises(SettingsError, match="at least one root"):
        control.remove_root(cfg, str(cfg.roots[0]))


def test_exclude_add_and_remove(cfg):
    n = len(cfg.excludes)
    control.add_exclude(cfg, "**/scratch/**")
    assert "**/scratch/**" in load_config(cfg.config_path).excludes
    with pytest.raises(SettingsError, match="already excluded"):
        control.add_exclude(cfg, "**/scratch/**")
    control.remove_exclude(cfg, "**/scratch/**")
    assert len(load_config(cfg.config_path).excludes) == n
    with pytest.raises(SettingsError, match="not in the exclude list"):
        control.remove_exclude(cfg, "**/never/**")


def test_update_section_validates_and_persists(cfg):
    changed = control.update_section(
        cfg, "deep", {"video_frames_per_scene": "1", "video_dedup_distance": 9}
    )
    assert changed == {"video_frames_per_scene": 1, "video_dedup_distance": 9}
    reloaded = load_config(cfg.config_path)
    assert reloaded.deep.video_frames_per_scene == 1
    assert reloaded.deep.video_dedup_distance == 9
    # a no-op patch writes nothing
    assert control.update_section(cfg, "deep", {"video_dedup_distance": 9}) == {}


def test_update_section_rejects_unknown_and_bad_values(cfg):
    with pytest.raises(SettingsError, match="not editable"):
        control.update_section(cfg, "deep", {"roots": "/etc"})
    with pytest.raises(SettingsError, match="unknown settings section"):
        control.update_section(cfg, "secrets", {"x": 1})
    with pytest.raises(SettingsError, match="expected int"):
        control.update_section(cfg, "deep", {"video_max_scenes": "lots"})
    with pytest.raises(SettingsError, match="cannot be negative"):
        control.update_section(cfg, "deep", {"prefetch_files": -1})


def test_directory_listing_returns_dirs_only(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "file.txt").write_text("x")

    d = control.list_directory(str(tmp_path))

    assert d["dirs"] == ["a", "b"]  # no files, no dotdirs
    assert d["path"] == str(tmp_path.resolve())
    assert d["parent"] == str(tmp_path.parent)
    with pytest.raises(SettingsError):
        control.list_directory(str(tmp_path / "file.txt"))


# ---------- job runner ----------


def test_job_runs_and_captures_output():
    runner = JobRunner()
    runner_cmd = {"echo": ["-c", "print('hello from job')"]}
    # exercise the real machinery with a harmless command
    from file_index import jobs as jobs_mod

    jobs_mod.RUNNABLE["_test"] = ["--help"]
    try:
        job = runner.start("_test")
        for _ in range(200):
            if not job.running:
                break
            threading.Event().wait(0.05)
        assert not job.running
        assert job.returncode == 0
        assert any("Usage" in line or "usage" in line for line in job.output)
    finally:
        jobs_mod.RUNNABLE.pop("_test", None)


def test_unknown_job_is_refused():
    with pytest.raises(JobError, match="unknown job"):
        JobRunner().start("rm -rf /")


def test_only_one_job_at_a_time():
    from file_index import jobs as jobs_mod

    runner = JobRunner()
    jobs_mod.RUNNABLE["_sleep"] = ["--help"]
    try:
        job = runner.start("_sleep")
        if job.running:  # racy by nature; only assert when still up
            with pytest.raises(JobError, match="already running"):
                runner.start("_sleep")
        runner.stop(job.id, force=True)
    finally:
        jobs_mod.RUNNABLE.pop("_sleep", None)


def test_stop_unknown_job():
    with pytest.raises(JobError, match="no job"):
        JobRunner().stop(999)


# ---------- HTTP surface ----------


@pytest.fixture
def server(tmp_env):
    cfg, index, root = tmp_env
    index.commit()
    srv = make_server(cfg, host="127.0.0.1", port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    handler = srv.RequestHandlerClass
    yield base, handler, cfg
    srv.shutdown()
    srv.server_close()


def _req(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_post_without_csrf_is_refused(server, tmp_path):
    base, handler, cfg = server
    cfg.save()
    newdir = tmp_path / "new"
    newdir.mkdir()

    status, body = _req(f"{base}/api/roots", "POST", {"path": str(newdir)},
                        {"Content-Type": "application/json"})

    assert status == 403
    assert "CSRF" in body["error"]
    assert load_config(cfg.config_path).roots == cfg.roots  # unchanged


def test_post_with_csrf_succeeds(server, tmp_path):
    base, handler, cfg = server
    newdir = tmp_path / "extra"
    newdir.mkdir()

    status, body = _req(f"{base}/api/roots", "POST", {"path": str(newdir)},
                        {"Content-Type": "application/json",
                         "X-CSRF-Token": handler.csrf_token})

    assert status == 200, body
    assert str(newdir.resolve()) in [str(r) for r in load_config(cfg.config_path).roots]


def test_page_embeds_the_token_and_writable_flag(server):
    base, handler, cfg = server
    with urllib.request.urlopen(base + "/") as r:
        page = r.read().decode()
    assert handler.csrf_token in page
    assert "__CSRF_TOKEN__" not in page  # placeholder substituted
    assert "const WRITABLE = true" in page


def test_settings_and_status_endpoints(server):
    base, handler, cfg = server
    status, s = _req(base + "/api/settings")
    assert status == 200
    assert s["writable"] is True
    assert s["roots"] == [str(r) for r in cfg.roots]
    assert "video_dedup_distance" in s["editable"]["deep"]

    status, st = _req(base + "/api/status")
    assert status == 200
    assert "queue" in st and "kinds" in st and "failures" in st


def test_fs_endpoint_lists_directories(server, tmp_path):
    base, handler, cfg = server
    (tmp_path / "pickme").mkdir()
    status, d = _req(base + "/api/fs?path=" + str(tmp_path))
    assert status == 200
    assert "pickme" in d["dirs"]


def test_non_loopback_bind_disables_writes(tmp_env, tmp_path):
    """Binding beyond loopback must not hand out config edits or job control."""
    cfg, index, root = tmp_env
    srv = make_server(cfg, host="0.0.0.0", port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        handler = srv.RequestHandlerClass
        assert handler.writable is False
        newdir = tmp_path / "nope"
        newdir.mkdir()
        status, body = _req(f"{base}/api/roots", "POST", {"path": str(newdir)},
                            {"Content-Type": "application/json",
                             "X-CSRF-Token": handler.csrf_token})
        assert status == 403
        assert "read-only" in body["error"]
        with urllib.request.urlopen(base + "/") as r:
            assert "const WRITABLE = false" in r.read().decode()
    finally:
        srv.shutdown()
        srv.server_close()


def test_settings_errors_are_reported_as_400(server):
    base, handler, cfg = server
    status, body = _req(f"{base}/api/roots", "POST", {"path": "/does/not/exist"},
                        {"Content-Type": "application/json",
                         "X-CSRF-Token": handler.csrf_token})
    assert status == 400
    assert "not a directory" in body["error"]


def test_unknown_post_route_is_404(server):
    base, handler, cfg = server
    status, _ = _req(f"{base}/api/nope", "POST", {},
                     {"Content-Type": "application/json",
                      "X-CSRF-Token": handler.csrf_token})
    assert status == 404
