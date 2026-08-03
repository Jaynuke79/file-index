"""Agent loop (`ask`) and organization planner (`organize`).

The agent runs qwen3 (config.models.agent) through Ollama's tool-calling chat
API with read-only tools over the index and the whitelisted filesystem roots.

Safety: every tool enforces the whitelist. `organize` only ever PRINTS a plan;
applying it happens in cli.py behind --apply + interactive confirmation, and
deletion is not implemented anywhere.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .index import Index
from .ollama_client import OllamaClient
from .search import hybrid_search
from .util import format_ts, strip_think

log = logging.getLogger("file_index.agent")

MAX_AGENT_TURNS = 12

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_index",
            "description": "Hybrid keyword+semantic search over the indexed file contents. "
            "Returns paths, snippets, and timestamps for audio/video hits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "limit": {"type": "integer", "description": "max results (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_content",
            "description": "Read the extracted/indexed content of a file (text, transcript, "
            "image analysis). Falls back to reading small text files directly from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute file path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories of a directory inside the whitelisted roots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute directory path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_info",
            "description": "Get indexed metadata for a file: size, mtime, type, hash, processing status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute file path"},
                },
                "required": ["path"],
            },
        },
    },
]


class AgentTools:
    def __init__(self, config: Config, index: Index):
        self.config = config
        self.index = index

    def dispatch(self, name: str, args: dict) -> str:
        try:
            if name == "search_index":
                return self.search_index(args.get("query", ""), int(args.get("limit", 10) or 10))
            if name == "read_file_content":
                return self.read_file_content(args.get("path", ""))
            if name == "list_directory":
                return self.list_directory(args.get("path", ""))
            if name == "get_file_info":
                return self.get_file_info(args.get("path", ""))
            return f"error: unknown tool {name}"
        except Exception as e:  # noqa: BLE001 — return errors to the model
            return f"error: {type(e).__name__}: {e}"

    def search_index(self, query: str, limit: int = 10) -> str:
        hits = hybrid_search(self.config, self.index, query, limit=limit)
        if not hits:
            return "no results"
        lines = []
        for h in hits:
            ts = f" @ {format_ts(h.ts_start)}" if h.ts_start is not None else ""
            lines.append(f"{h.path}{ts} [{h.stage}]\n  {h.snippet[:200]}")
        return "\n".join(lines)

    def _check_path(self, path_s: str) -> Path:
        p = Path(path_s).expanduser()
        if not self.config.is_within_roots(p):
            raise PermissionError(f"{p} is outside the whitelisted roots")
        return p

    def read_file_content(self, path_s: str) -> str:
        p = self._check_path(path_s)
        row = self.index.get_file_by_path(str(p.resolve()))
        if row:
            parts = []
            for c in self.index.get_content(row["id"]):
                if c["body"]:
                    parts.append(f"--- {c['stage']} ---\n{c['body'][:8000]}")
            if parts:
                return "\n\n".join(parts)[:24000]
        # not indexed (or nothing extracted): try reading directly if small text
        if p.is_file() and p.stat().st_size <= 200_000:
            from .extractors import text as text_ex

            try:
                return text_ex.extract(p, 24000)
            except (ValueError, UnicodeDecodeError):
                return "file exists but has no extractable text content"
        return "no indexed content for this file"

    def list_directory(self, path_s: str) -> str:
        p = self._check_path(path_s)
        if not p.is_dir():
            return f"error: {p} is not a directory"
        entries = []
        for child in sorted(p.iterdir()):
            try:
                if child.is_dir():
                    entries.append(f"{child.name}/")
                else:
                    entries.append(f"{child.name} ({child.stat().st_size} bytes)")
            except OSError:
                entries.append(f"{child.name} (unreadable)")
            if len(entries) >= 500:
                entries.append("… (truncated at 500 entries)")
                break
        return "\n".join(entries) or "(empty)"

    def get_file_info(self, path_s: str) -> str:
        p = self._check_path(path_s)
        row = self.index.get_file_by_path(str(p.resolve()))
        if not row:
            return "not in index"
        info = {
            "path": row["path"], "size": row["size"], "mtime": row["mtime"],
            "mime": row["mime"], "kind": row["kind"], "hash": row["hash"],
            "tier1_status": row["tier1_status"], "tier2_status": row["tier2_status"],
        }
        stages = [c["stage"] for c in self.index.get_content(row["id"])]
        info["extracted_stages"] = stages
        return json.dumps(info, indent=1)


ASK_SYSTEM = """You are a local file-search assistant. You answer questions about the
user's files using the provided tools. Everything runs locally; you may freely read
indexed content. Cite file paths in your answers. If a search returns nothing useful,
try rephrased queries before giving up. Be concise and concrete."""



def ask(config: Config, index: Index, question: str, on_tool=None) -> str:
    client = OllamaClient(config.models.ollama_url)
    client.require()
    tools_impl = AgentTools(config, index)
    messages = [
        {"role": "system", "content": ASK_SYSTEM},
        {"role": "user", "content": question},
    ]
    for _ in range(MAX_AGENT_TURNS):
        msg = client.chat(config.models.agent, messages, tools=TOOLS)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return strip_think(msg.get("content", "")) or "(no answer)"
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if on_tool:
                on_tool(name, args)
            result = tools_impl.dispatch(name, args)
            messages.append({"role": "tool", "content": result, "tool_name": name})
    return "(agent reached max turns without a final answer)"


# ---------------- organize ----------------

@dataclass
class PlanAction:
    action: str  # move | rename | mkdir | delete_candidate
    src: str | None
    dst: str | None
    reason: str


@dataclass
class OrganizePlan:
    summary: str
    actions: list[PlanAction]


ORGANIZE_SYSTEM = """You are a file-organization assistant. You will be given a directory
listing (with indexed content hints) and must propose a reorganization: logical
groupings into subdirectories, clearer file names, and moves.

Rules:
- Propose moves/renames ONLY within the given directory.
- You may flag files as deletion CANDIDATES (duplicates, junk) but deletion will
  never be performed; it is informational only.
- Return ONLY a JSON object, no other text:
{
  "summary": "one-paragraph description of the proposed organization",
  "actions": [
    {"action": "mkdir", "dst": "<absolute dir path>", "reason": "..."},
    {"action": "move", "src": "<absolute path>", "dst": "<absolute path>", "reason": "..."},
    {"action": "rename", "src": "<absolute path>", "dst": "<absolute path>", "reason": "..."},
    {"action": "delete_candidate", "src": "<absolute path>", "reason": "..."}
  ]
}"""


def propose_organization(config: Config, index: Index, directory: Path) -> OrganizePlan:
    client = OllamaClient(config.models.ollama_url)
    client.require()
    if not config.is_within_roots(directory):
        raise PermissionError(f"{directory} is outside the whitelisted roots")

    tools_impl = AgentTools(config, index)
    listing = tools_impl.list_directory(str(directory))

    # attach one-line content hints for indexed files
    hints = []
    for row in index.db.execute(
        "SELECT id, path, kind FROM files WHERE deleted=0 AND path LIKE ? LIMIT 200",
        (str(directory.resolve()) + "/%",),
    ):
        contents = index.get_content(row["id"])
        hint = ""
        for c in contents:
            if c["body"]:
                hint = c["body"][:120].replace("\n", " ")
                break
        hints.append(f"{row['path']} [{row['kind']}] {hint}")

    prompt = (
        f"Directory to organize: {directory}\n\nListing:\n{listing}\n\n"
        f"Indexed content hints:\n" + "\n".join(hints[:200])
    )
    raw = client.generate(
        config.models.agent,
        ORGANIZE_SYSTEM + "\n\n" + prompt + "\n\nReturn only the JSON object. /no_think",
        format_json=True,
    )
    raw = strip_think(raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1:
        raise ValueError(f"agent did not return JSON:\n{raw[:500]}")
    data = json.loads(raw[start : end + 1])

    actions = []
    for a in data.get("actions", []):
        act = PlanAction(
            action=a.get("action", ""),
            src=a.get("src"),
            dst=a.get("dst"),
            reason=a.get("reason", ""),
        )
        if act.action not in ("move", "rename", "mkdir", "delete_candidate"):
            continue
        # safety: every referenced path must stay inside the target directory
        ok = True
        for pth in (act.src, act.dst):
            if pth is None:
                continue
            try:
                Path(pth).resolve().relative_to(directory.resolve())
            except ValueError:
                ok = False
        if ok:
            actions.append(act)
    return OrganizePlan(summary=data.get("summary", ""), actions=actions)
