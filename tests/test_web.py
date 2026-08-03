"""Tests for the browse web UI: Store queries and the HTTP endpoints."""

import json
import threading
import urllib.request

import pytest

from file_index.web import Store, _fts_escape, make_server


@pytest.fixture
def populated(tmp_env):
    """Index with one captioned image, one video, one text file, one deleted."""
    cfg, index, root = tmp_env

    from PIL import Image

    img_path = root / "cat.png"
    Image.new("RGB", (64, 48), (200, 120, 40)).save(img_path)

    fid_img = index.upsert_file(str(img_path), "h1", img_path.stat().st_size, 100.0, "image/png", "image")
    index.store_content(
        fid_img, "vlm_image", "image-1.2",
        "A ginger cat sleeping on a windowsill",
        meta={"description": "A ginger cat sleeping on a windowsill in the sun",
              "ocr_text": "", "type": "photo", "objects": ["cat", "windowsill"],
              "people_count": 0, "inferred_context": "pet photo"},
    )
    index.set_tier_status(fid_img, 2, "done")

    fid_vid = index.upsert_file(str(root / "clip.mp4"), "h2", 10, 200.0, "video/mp4", "video")
    index.store_content(fid_vid, "video_summary", "video-1.0", "A short match of Smite")

    fid_txt = index.upsert_file(str(root / "notes.txt"), "h3", 5, 300.0, "text/plain", "text")
    index.store_content(fid_txt, "text", "text-1.0", "meeting notes about taxes")

    fid_gone = index.upsert_file(str(root / "gone.png"), "h4", 5, 400.0, "image/png", "image")
    # captioned BEFORE deletion: must not count toward summary()["captioned"]
    index.store_content(fid_gone, "vlm_image", "image-1.2", "an old screenshot")
    index.mark_deleted(fid_gone)

    index.commit()
    return cfg, {"img": fid_img, "vid": fid_vid, "txt": fid_txt, "gone": fid_gone}


def test_fts_escape_neutralizes_syntax():
    assert _fts_escape('cat AND "dog" OR (x)') == '"cat" "AND" """dog""" "OR" "(x)"'


def test_summary(populated):
    cfg, _ = populated
    s = Store(cfg.db_path).summary()
    assert s["kinds"] == {"image": 1, "video": 1, "text": 1}
    assert s["total"] == 3
    assert s["captioned"] == 2  # vlm_image + video_summary; plain text is not a caption stage


def test_list_files_captions_and_order(populated):
    cfg, ids = populated
    r = Store(cfg.db_path).list_files()
    assert r["total"] == 3
    by_id = {f["id"]: f for f in r["files"]}
    assert ids["gone"] not in by_id
    # vlm meta description is preferred over the flattened body
    assert by_id[ids["img"]]["caption"].startswith("A ginger cat sleeping on a windowsill in the sun")
    assert by_id[ids["img"]]["vlm_type"] == "photo"
    assert by_id[ids["vid"]]["caption_stage"] == "video_summary"
    # newest mtime first
    assert [f["id"] for f in r["files"]] == [ids["txt"], ids["vid"], ids["img"]]


def test_list_files_filters(populated):
    cfg, ids = populated
    store = Store(cfg.db_path)
    r = store.list_files(kind="image")
    assert [f["id"] for f in r["files"]] == [ids["img"]]
    r = store.list_files(captioned_only=True)
    assert {f["id"] for f in r["files"]} == {ids["img"], ids["vid"]}
    r = store.list_files(limit=1)
    assert r["total"] == 3 and len(r["files"]) == 1


def test_search(populated):
    cfg, ids = populated
    store = Store(cfg.db_path)
    r = store.list_files(q="ginger cat")
    assert [f["id"] for f in r["files"]] == [ids["img"]]
    r = store.list_files(q="smite", kind="text")
    assert r["total"] == 0
    # hostile FTS syntax must not raise
    assert store.list_files(q='"unbalanced AND NEAR(')["total"] == 0


def test_file_detail(populated):
    cfg, ids = populated
    store = Store(cfg.db_path)
    d = store.file_detail(ids["img"])
    stages = {c["stage"]: c for c in d["content"]}
    assert stages["vlm_image"]["meta"]["objects"] == ["cat", "windowsill"]
    assert store.file_detail(999999) is None
    assert store.file_detail(ids["gone"]) is None


@pytest.fixture
def server(populated):
    cfg, ids = populated
    srv = make_server(cfg, host="127.0.0.1", port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", ids
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_http_index_and_api(server):
    base, ids = server
    status, headers, body = _get(base + "/")
    assert status == 200 and b"file-index browser" in body

    status, _, body = _get(base + "/api/files?kind=image")
    assert status == 200
    data = json.loads(body)
    assert data["total"] == 1 and data["files"][0]["id"] == ids["img"]

    status, _, _ = _get(base + "/api/file/999999")
    assert status == 404


def test_http_thumb(server):
    base, ids = server
    status, headers, body = _get(f"{base}/thumb/{ids['img']}")
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert body[:2] == b"\xff\xd8"  # JPEG magic
    # missing on disk (clip.mp4 was never created)
    status, _, _ = _get(f"{base}/thumb/{ids['vid']}")
    assert status == 404


def test_http_media_and_ranges(server):
    base, ids = server
    status, headers, body = _get(f"{base}/media/{ids['img']}")
    assert status == 200 and headers["Content-Type"] == "image/png"
    full = body

    status, headers, body = _get(f"{base}/media/{ids['img']}", {"Range": "bytes=0-3"})
    assert status == 206
    assert body == full[:4]
    assert headers["Content-Range"] == f"bytes 0-3/{len(full)}"

    status, _, body = _get(f"{base}/media/{ids['img']}", {"Range": f"bytes={len(full) - 2}-"})
    assert status == 206 and body == full[-2:]

    status, _, _ = _get(f"{base}/media/{ids['img']}", {"Range": f"bytes={len(full) + 10}-"})
    assert status == 416

    # malformed range (start > end) is ignored: full 200 response
    status, headers, body = _get(f"{base}/media/{ids['img']}", {"Range": "bytes=500-100"})
    assert status == 200
    assert body == full
    assert int(headers["Content-Length"]) == len(full)

    # deleted files are never served
    status, _, _ = _get(f"{base}/media/{ids['gone']}")
    assert status == 404


def test_browse_without_index_errors_and_creates_no_db(tmp_path, monkeypatch):
    """`browse` on a fresh machine must say "run scan first", not create an
    empty db (constructing Index would create the file and defeat the check)."""
    from typer.testing import CliRunner

    from file_index import cli
    from file_index.config import Config

    cfg = Config()
    cfg.roots = [tmp_path]
    cfg.data_dir = tmp_path / "state"
    cfg.data_dir.mkdir()
    monkeypatch.setattr(cli, "load_config", lambda: cfg)

    result = CliRunner().invoke(cli.app, ["browse", "--no-open"])
    assert result.exit_code == 1
    assert "scan" in result.output
    assert not cfg.db_path.exists()


def test_host_header_validation_blocks_rebinding(server):
    """A request whose Host header names a foreign domain (DNS rebinding)
    must be rejected even though it reaches the loopback socket."""
    import http.client

    base, ids = server
    port = int(base.rsplit(":", 1)[1])

    def get_with_host(host_header):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("GET", "/api/summary", skip_host=True)
        conn.putheader("Host", host_header)
        conn.endheaders()
        resp = conn.getresponse()
        status, body = resp.status, resp.read()
        conn.close()
        return status, body

    status, body = get_with_host("evil.example.com")
    assert status == 403
    assert b"api" not in body or b"kinds" not in body  # no data leaked

    status, _ = get_with_host(f"attacker.net:{port}")
    assert status == 403

    # legitimate spellings still work
    assert get_with_host(f"127.0.0.1:{port}")[0] == 200
    assert get_with_host(f"localhost:{port}")[0] == 200


def test_non_loopback_bind_skips_host_filtering(populated):
    """Binding beyond loopback is an explicit exposure choice; Host filtering
    can't enumerate the machine's names, so it is disabled (with a CLI warning)."""
    import http.client

    cfg, ids = populated
    srv = make_server(cfg, host="0.0.0.0", port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("GET", "/api/summary", skip_host=True)
        conn.putheader("Host", "some-lan-name.local")
        conn.endheaders()
        assert conn.getresponse().status == 200
        conn.close()
    finally:
        srv.shutdown()
        srv.server_close()


def test_browse_warns_on_non_loopback_host(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from file_index import cli, web
    from file_index.config import Config
    from file_index.index import Index

    cfg = Config()
    cfg.roots = [tmp_path]
    cfg.data_dir = tmp_path / "state"
    cfg.data_dir.mkdir()
    Index(cfg.db_path).close()  # index exists so browse proceeds
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    served = {}
    monkeypatch.setattr(web, "serve", lambda *a, **k: served.setdefault("called", True))

    result = CliRunner().invoke(cli.app, ["browse", "--host", "0.0.0.0", "--no-open"])
    assert result.exit_code == 0
    assert "WARNING" in result.output and "unauthenticated" in result.output
    assert served.get("called")

    result = CliRunner().invoke(cli.app, ["browse", "--no-open"])  # loopback: quiet
    assert "WARNING" not in result.output
