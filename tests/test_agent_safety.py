"""The whitelist is the project's core safety guarantee — these tests fail if
any agent tool, the organize planner, or `organize --apply` stops rejecting
paths outside the configured roots.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from file_index import agent as agent_mod
from file_index.agent import AgentTools, propose_organization


@pytest.fixture
def tools(tmp_env):
    cfg, index, root = tmp_env
    (root / "inside.txt").write_text("indexed content")
    return AgentTools(cfg, index), cfg, index, root


# ---------- _check_path / per-tool enforcement ----------


def test_check_path_accepts_inside_roots(tools):
    t, cfg, index, root = tools
    assert t._check_path(str(root / "inside.txt")) == root / "inside.txt"


@pytest.mark.parametrize("bad", ["/etc/passwd", "~/.ssh/id_rsa", "/"])
def test_check_path_rejects_outside_roots(tools, bad):
    t, cfg, index, root = tools
    with pytest.raises(PermissionError):
        t._check_path(bad)


def test_check_path_rejects_traversal(tools):
    t, cfg, index, root = tools
    escape = str(root / ".." / ".." / "etc" / "passwd")
    with pytest.raises(PermissionError):
        t._check_path(escape)


def test_check_path_rejects_symlink_leaving_root(tools, tmp_path):
    t, cfg, index, root = tools
    secret = tmp_path / "outside_secret.txt"
    secret.write_text("password")
    link = root / "looks_local.txt"
    os.symlink(secret, link)
    # resolve() follows the link out of the root — must be refused
    with pytest.raises(PermissionError):
        t._check_path(str(link))


@pytest.mark.parametrize(
    "tool", ["read_file_content", "list_directory", "get_file_info"]
)
def test_every_path_tool_enforces_whitelist(tools, tool):
    t, cfg, index, root = tools
    # dispatch() converts the refusal into a model-visible error string
    out = t.dispatch(tool, {"path": "/etc"})
    assert out.startswith("error: PermissionError")


def test_dispatch_rejects_unknown_tool(tools):
    t, cfg, index, root = tools
    assert "unknown tool" in t.dispatch("rm_rf", {})


def test_list_directory_lists_only_inside(tools):
    t, cfg, index, root = tools
    (root / "sub").mkdir()
    out = t.list_directory(str(root))
    assert "inside.txt" in out and "sub/" in out


# ---------- organize planner filtering ----------


def _plan_with(actions, monkeypatch, target):
    client = MagicMock()
    client.generate.return_value = json.dumps(
        {"summary": "tidy up", "actions": actions}
    )
    monkeypatch.setattr(agent_mod, "OllamaClient", lambda url: client)
    return client


def test_propose_organization_drops_paths_outside_target(tools, monkeypatch, tmp_path):
    t, cfg, index, root = tools
    outside = tmp_path / "elsewhere" / "stolen.txt"
    _plan_with(
        [
            {"action": "move", "src": str(root / "inside.txt"),
             "dst": str(root / "docs" / "inside.txt"), "reason": "ok"},
            {"action": "move", "src": str(root / "inside.txt"),
             "dst": str(outside), "reason": "escapes target"},
            {"action": "move", "src": str(root / ".." / "passwd"),
             "dst": str(root / "p"), "reason": "traversal src"},
            {"action": "delete", "src": str(root / "inside.txt"),
             "dst": None, "reason": "not an allowed verb"},
        ],
        monkeypatch, root,
    )
    plan = propose_organization(cfg, index, root)
    assert len(plan.actions) == 1
    assert plan.actions[0].dst == str(root / "docs" / "inside.txt")


def test_propose_organization_refuses_directory_outside_roots(tools, monkeypatch, tmp_path):
    t, cfg, index, root = tools
    _plan_with([], monkeypatch, root)
    with pytest.raises(PermissionError):
        propose_organization(cfg, index, tmp_path / "not_a_root")


def test_delete_candidates_are_kept_but_never_executable(tools, monkeypatch):
    """delete_candidate survives filtering (it is informational) but is not in
    the set cli.organize will ever apply."""
    t, cfg, index, root = tools
    _plan_with(
        [{"action": "delete_candidate", "src": str(root / "inside.txt"),
          "dst": None, "reason": "duplicate"}],
        monkeypatch, root,
    )
    plan = propose_organization(cfg, index, root)
    assert [a.action for a in plan.actions] == ["delete_candidate"]
    doable = [a for a in plan.actions if a.action in ("move", "rename", "mkdir")]
    assert doable == []


# ---------- organize --apply guards ----------


def test_apply_moves_file_and_updates_index(tools, monkeypatch):
    from typer.testing import CliRunner

    from file_index import cli
    from file_index.agent import OrganizePlan, PlanAction

    t, cfg, index, root = tools
    src, dst = root / "inside.txt", root / "docs" / "inside.txt"
    fid = index.upsert_file(str(src), "h1", 5, 1.0, "text/plain", "text")
    index.commit()

    monkeypatch.setattr(cli, "_load", lambda: (cfg, index))
    monkeypatch.setattr(
        cli.agent_mod if hasattr(cli, "agent_mod") else agent_mod,
        "propose_organization",
        lambda c, i, d: OrganizePlan("s", [PlanAction("move", str(src), str(dst), "r")]),
    )

    result = CliRunner().invoke(cli.app, ["organize", str(root), "--apply"], input="y\n")
    assert result.exit_code == 0, result.output
    assert dst.exists() and not src.exists()
    assert index.get_file_by_path(str(dst)) is not None
    assert index.get_file_by_path(str(dst))["id"] == fid


def test_apply_refuses_action_escaping_roots(tools, monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from file_index import cli
    from file_index.agent import OrganizePlan, PlanAction

    t, cfg, index, root = tools
    src = root / "inside.txt"
    outside = tmp_path / "escaped.txt"
    monkeypatch.setattr(cli, "_load", lambda: (cfg, index))
    monkeypatch.setattr(
        agent_mod, "propose_organization",
        lambda c, i, d: OrganizePlan("s", [PlanAction("move", str(src), str(outside), "r")]),
    )

    result = CliRunner().invoke(cli.app, ["organize", str(root), "--apply"], input="y\n")
    assert not outside.exists()  # PermissionError caught and reported
    assert src.exists()


def test_apply_skips_when_target_exists(tools, monkeypatch):
    from typer.testing import CliRunner

    from file_index import cli
    from file_index.agent import OrganizePlan, PlanAction

    t, cfg, index, root = tools
    src, dst = root / "inside.txt", root / "taken.txt"
    dst.write_text("do not clobber me")
    monkeypatch.setattr(cli, "_load", lambda: (cfg, index))
    monkeypatch.setattr(
        agent_mod, "propose_organization",
        lambda c, i, d: OrganizePlan("s", [PlanAction("move", str(src), str(dst), "r")]),
    )

    CliRunner().invoke(cli.app, ["organize", str(root), "--apply"], input="y\n")
    assert dst.read_text() == "do not clobber me"
    assert src.exists()


def test_without_apply_nothing_changes(tools, monkeypatch):
    from typer.testing import CliRunner

    from file_index import cli
    from file_index.agent import OrganizePlan, PlanAction

    t, cfg, index, root = tools
    src, dst = root / "inside.txt", root / "docs" / "inside.txt"
    monkeypatch.setattr(cli, "_load", lambda: (cfg, index))
    monkeypatch.setattr(
        agent_mod, "propose_organization",
        lambda c, i, d: OrganizePlan("s", [PlanAction("move", str(src), str(dst), "r")]),
    )

    result = CliRunner().invoke(cli.app, ["organize", str(root)])
    assert result.exit_code == 0
    assert src.exists() and not dst.exists()
    assert "Read-only" in result.output
