"""The browse UI: one self-contained page, no external assets.

Kept in its own module because it is a large literal. `web.py` serves it and
substitutes __CSRF_TOKEN__ and __WRITABLE__ per request; the page refuses to
show write controls when the server reports itself read-only.
"""

from __future__ import annotations

WEB_UI = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>file-index</title>
<style>
:root {
  --bg: #101014; --panel: #17171d; --card: #1c1c24; --border: #2a2a35;
  --text: #e8e8ee; --dim: #9a9aa8; --accent: #7aa2f7; --badge: #242430;
  --danger: #f7768e; --ok: #9ece6a;
}
* { box-sizing: border-box; margin: 0; }
body { background: var(--bg); color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; z-index: 10; background: var(--panel);
  border-bottom: 1px solid var(--border); padding: 10px 16px;
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
header h1 { font-size: 15px; font-weight: 600; margin-right: 6px; }
#q { flex: 1 1 220px; max-width: 420px; background: var(--card); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px; padding: 7px 12px; outline: none; }
#q:focus { border-color: var(--accent); }
.chip { background: var(--card); border: 1px solid var(--border); color: var(--dim);
  border-radius: 999px; padding: 4px 12px; cursor: pointer; font-size: 13px; }
.chip.on { color: var(--text); border-color: var(--accent); background: #20283e; }
label.cap { color: var(--dim); display: flex; gap: 6px; align-items: center;
  cursor: pointer; font-size: 13px; }
#count { color: var(--dim); font-size: 13px; margin-left: auto; }
/* tabs */
nav { display: flex; gap: 4px; }
nav button { background: none; border: 1px solid transparent; color: var(--dim);
  border-radius: 8px; padding: 5px 12px; cursor: pointer; font-size: 13.5px; }
nav button:hover { color: var(--text); }
nav button.on { color: var(--text); background: var(--card); border-color: var(--border); }
.view { display: none; }
.view.on { display: block; }
/* library */
#grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 14px; padding: 16px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden; cursor: pointer; display: flex; flex-direction: column; }
.card:hover { border-color: var(--accent); }
.thumb { aspect-ratio: 16/10; background: #0c0c10; display: flex;
  align-items: center; justify-content: center; overflow: hidden; }
.thumb img { width: 100%; height: 100%; object-fit: cover; }
.thumb .ph { font-size: 34px; opacity: .45; }
.card .body { padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 6px; }
.card .name { font-weight: 600; font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.card .cap-text { color: var(--dim); font-size: 12.5px; display: -webkit-box;
  -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.badges { display: flex; gap: 6px; flex-wrap: wrap; }
.badge { background: var(--badge); color: var(--dim); border-radius: 5px;
  font-size: 11px; padding: 2px 7px; }
#sentinel { height: 60px; }
#empty { color: var(--dim); text-align: center; padding: 60px 0; display: none; }
/* panels shared by settings/jobs/status */
.wrap { padding: 18px; max-width: 1100px; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px; margin-bottom: 16px; }
.panel > h2 { font-size: 14px; margin-bottom: 4px; }
.panel > .hint { color: var(--dim); font-size: 12.5px; margin-bottom: 12px; }
.row { display: flex; gap: 8px; align-items: center; padding: 7px 0;
  border-top: 1px solid var(--border); }
.row:first-of-type { border-top: none; }
.row .grow { flex: 1; word-break: break-all; font-size: 13px; }
button.act { background: var(--card); color: var(--text); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 12px; cursor: pointer; font-size: 13px; }
button.act:hover:not(:disabled) { border-color: var(--accent); }
button.act:disabled { opacity: .45; cursor: not-allowed; }
button.act.danger:hover:not(:disabled) { border-color: var(--danger); color: var(--danger); }
button.act.primary { border-color: var(--accent); color: var(--accent); }
input.txt, select.txt { background: var(--card); color: var(--text); font-size: 13px;
  border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; outline: none; }
input.txt:focus, select.txt:focus { border-color: var(--accent); }
.field { display: flex; gap: 10px; align-items: center; padding: 5px 0; }
.field label { flex: 0 0 210px; color: var(--dim); font-size: 12.5px; }
.field input.txt { flex: 1; max-width: 320px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: var(--dim); font-weight: 500; font-size: 12px;
  text-transform: uppercase; letter-spacing: .04em; padding: 6px 8px; }
td { padding: 6px 8px; border-top: 1px solid var(--border); word-break: break-word; }
td.num, th.num { text-align: right; }
pre.out { background: #0c0c10; border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 12px; font-size: 12px; max-height: 420px; overflow: auto;
  white-space: pre-wrap; word-break: break-word; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot.run { background: var(--ok); } .dot.idle { background: var(--dim); opacity: .5; }
.ro { color: var(--danger); font-size: 12.5px; }
#toast { position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%);
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 9px 16px; font-size: 13px; z-index: 40; display: none; max-width: 80vw; }
#toast.err { border-color: var(--danger); color: var(--danger); }
/* modal */
#overlay, #pick { position: fixed; inset: 0; background: rgba(0,0,0,.65); display: none;
  z-index: 20; align-items: center; justify-content: center; padding: 24px; }
#overlay.open, #pick.open { display: flex; }
#modal { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  width: min(1200px, 100%); max-height: 92vh; display: flex; overflow: hidden; }
#m-media { flex: 1.4; background: #000; display: flex; align-items: center;
  justify-content: center; min-width: 0; }
#m-media img, #m-media video { max-width: 100%; max-height: 92vh; object-fit: contain; }
#m-media .ph { font-size: 80px; opacity: .4; }
#m-side { flex: 1; min-width: 320px; max-width: 460px; overflow-y: auto; padding: 18px; }
#m-side h2 { font-size: 15px; word-break: break-all; margin-bottom: 4px; }
#m-side .path { color: var(--dim); font-size: 12px; word-break: break-all;
  cursor: pointer; margin-bottom: 10px; }
#m-side .path:hover { color: var(--accent); }
#m-side section { border-top: 1px solid var(--border); padding: 12px 0; }
#m-side section h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
  color: var(--accent); margin-bottom: 6px; }
#m-side p, #m-side pre { color: var(--text); font-size: 13px; white-space: pre-wrap;
  word-break: break-word; }
#m-side pre { max-height: 260px; overflow-y: auto; background: var(--card);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; }
#m-close { position: absolute; top: 14px; right: 18px; font-size: 26px; color: #fff;
  background: none; border: none; cursor: pointer; opacity: .7; }
#m-close:hover { opacity: 1; }
/* directory picker */
#pick-box { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  width: min(640px, 100%); max-height: 80vh; display: flex; flex-direction: column; }
#pick-box header { position: static; border-radius: 12px 12px 0 0; }
#pick-cur { font-size: 12.5px; color: var(--dim); word-break: break-all; }
#pick-list { overflow-y: auto; padding: 6px 0; flex: 1; }
#pick-list .row { padding: 7px 16px; cursor: pointer; }
#pick-list .row:hover { background: var(--card); }
#pick-foot { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--border); }
@media (max-width: 800px) { #modal { flex-direction: column; overflow-y: auto; }
  #m-side { max-width: none; } .field label { flex-basis: 130px; } }
</style>
</head>
<body>
<header>
  <h1>file-index</h1>
  <nav id="tabs"></nav>
  <input id="q" type="search" placeholder="Search captions, text, transcripts…">
  <div id="chips"></div>
  <label class="cap"><input id="captioned" type="checkbox"> captioned only</label>
  <span id="count"></span>
</header>

<div id="view-library" class="view on">
  <div id="grid"></div>
  <div id="empty">No files match.</div>
  <div id="sentinel"></div>
</div>

<div id="view-settings" class="view"><div class="wrap" id="settings-body"></div></div>
<div id="view-jobs" class="view"><div class="wrap" id="jobs-body"></div></div>
<div id="view-status" class="view"><div class="wrap" id="status-body"></div></div>

<div id="overlay"><button id="m-close">&times;</button><div id="modal">
  <div id="m-media"></div><div id="m-side"></div>
</div></div>

<div id="pick"><div id="pick-box">
  <header><h1>Choose a folder</h1><span id="pick-cur"></span></header>
  <div id="pick-list"></div>
  <div id="pick-foot">
    <button class="act" id="pick-up">↑ Parent</button>
    <span style="flex:1"></span>
    <button class="act" id="pick-cancel">Cancel</button>
    <button class="act primary" id="pick-ok">Index this folder</button>
  </div>
</div></div>

<div id="toast"></div>

<script>
"use strict";
const CSRF = "__CSRF_TOKEN__";
const WRITABLE = __WRITABLE__;
const state = { q: "", kind: "", captioned: false, offset: 0, total: 0, busy: false, done: false };
const PAGE = 60;
const ICONS = { image: "🖼", video: "🎬", audio: "🎧", pdf: "📄", text: "📄",
  code: "⌨", office: "📄", other: "📦" };
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function toast(msg, isErr) {
  const t = $("toast");
  t.textContent = msg;
  t.className = isErr ? "err" : "";
  t.style.display = "block";
  clearTimeout(t._h);
  t._h = setTimeout(() => { t.style.display = "none"; }, isErr ? 6000 : 2800);
}

async function getJSON(path) {
  const r = await fetch(path);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": CSRF },
    body: JSON.stringify(body || {}),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

function human(n) {
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i ? n.toFixed(1) : n.toFixed(0)) + " " + u[i];
}

// tabs ----------------------------------------------------------------
const VIEWS = [
  { id: "library", label: "Library" },
  { id: "settings", label: "Settings" },
  { id: "jobs", label: "Jobs" },
  { id: "status", label: "Status" },
];
let current = "library";
let poll = null;

function show(id) {
  current = id;
  for (const v of VIEWS) {
    $("view-" + v.id).classList.toggle("on", v.id === id);
  }
  for (const b of $("tabs").children) b.classList.toggle("on", b.dataset.id === id);
  // the search controls belong to the library view only
  for (const el of [$("q"), $("chips"), $("count")]) {
    el.style.display = id === "library" ? "" : "none";
  }
  document.querySelector("label.cap").style.display = id === "library" ? "" : "none";
  if (poll) { clearInterval(poll); poll = null; }
  if (id === "settings") renderSettings();
  if (id === "jobs") { renderJobs(); poll = setInterval(renderJobs, 1500); }
  if (id === "status") { renderStatus(); poll = setInterval(renderStatus, 5000); }
  location.hash = id;
}

function buildTabs() {
  const nav = $("tabs");
  for (const v of VIEWS) {
    const b = el("button", v.id === current ? "on" : "", v.label);
    b.dataset.id = v.id;
    b.onclick = () => show(v.id);
    nav.appendChild(b);
  }
}

// settings ------------------------------------------------------------
function panel(title, hint) {
  const p = el("div", "panel");
  p.appendChild(el("h2", "", title));
  if (hint) p.appendChild(el("div", "hint", hint));
  return p;
}

function listRow(text, onRemove, removeLabel) {
  const r = el("div", "row");
  r.appendChild(el("span", "grow", text));
  if (WRITABLE && onRemove) {
    const b = el("button", "act danger", removeLabel || "Remove");
    b.onclick = onRemove;
    r.appendChild(b);
  }
  return r;
}

async function renderSettings() {
  const body = $("settings-body");
  let s;
  try { s = await getJSON("api/settings"); }
  catch (e) { body.textContent = ""; body.appendChild(panel("Settings unavailable", e.message)); return; }
  body.textContent = "";

  if (!WRITABLE) {
    const p = panel("Read-only",
      "This server is bound to a non-loopback address, so settings and jobs are disabled. " +
      "Restart `file-index browse` without --host to enable them.");
    p.className = "panel";
    body.appendChild(p);
  }

  // roots
  const rp = panel("Indexed folders",
    "Only these directories are ever read. Changes are written to " + s.config_path +
    " — run a scan afterwards to pick up new files.");
  for (const r of s.roots) {
    rp.appendChild(listRow(r, async () => {
      if (!confirm("Stop indexing " + r + "?\n\nAlready-indexed files stay in the index " +
                   "until you exclude or purge them. Files on disk are never touched."))
        return;
      try { await post("api/roots/remove", { path: r }); toast("Removed " + r); renderSettings(); }
      catch (e) { toast(e.message, true); }
    }));
  }
  if (!s.roots.length) rp.appendChild(el("div", "hint", "No folders configured."));
  if (WRITABLE) {
    const add = el("div", "row");
    const inp = el("input", "txt grow");
    inp.placeholder = "/home/you/Documents";
    inp.onkeydown = (e) => { if (e.key === "Enter") addRoot(inp.value); };
    const browse = el("button", "act", "Browse…");
    browse.onclick = () => openPicker(s.roots[0] || null);
    const b = el("button", "act primary", "Add folder");
    b.onclick = () => addRoot(inp.value);
    add.append(inp, browse, b);
    rp.appendChild(add);
  }
  body.appendChild(rp);

  // excludes
  const ep = panel("Excluded patterns",
    "Glob patterns skipped during crawling. A directory path excludes everything beneath it.");
  for (const x of s.excludes) {
    ep.appendChild(listRow(x, async () => {
      try { await post("api/excludes/remove", { pattern: x }); toast("Removed " + x); renderSettings(); }
      catch (e) { toast(e.message, true); }
    }));
  }
  if (WRITABLE) {
    const add = el("div", "row");
    const inp = el("input", "txt grow");
    inp.placeholder = "**/node_modules/**  or  /path/to/folder";
    const b = el("button", "act primary", "Add pattern");
    const go = async () => {
      if (!inp.value.trim()) return;
      try { await post("api/excludes", { pattern: inp.value }); toast("Added"); renderSettings(); }
      catch (e) { toast(e.message, true); }
    };
    inp.onkeydown = (e) => { if (e.key === "Enter") go(); };
    b.onclick = go;
    add.append(inp, b);
    ep.appendChild(add);
  }
  body.appendChild(ep);

  body.appendChild(sectionForm("Models", "Ollama model names and server URL. After changing " +
    "one, run reindex so existing files are re-extracted with it.", "models", s.models, s.editable.models));
  body.appendChild(sectionForm("Deep processing", "Tuning for the tier-2 pass. Lower " +
    "video_frames_per_scene or raise video_dedup_distance to trade detail for speed.",
    "deep", s.deep, s.editable.deep));
  body.appendChild(sectionForm("Limits", "Crawl and chunking caps.", "limits", s.limits, s.editable.limits));
}

function sectionForm(title, hint, section, values, editable) {
  const p = panel(title, hint);
  const inputs = {};
  for (const k of editable) {
    const f = el("div", "field");
    f.appendChild(el("label", "", k));
    const v = values[k];
    let inp;
    if (typeof v === "boolean") {
      inp = el("input", "txt");
      inp.type = "checkbox";
      inp.checked = v;
      inp.style.flex = "0 0 auto";
    } else {
      inp = el("input", "txt");
      inp.value = v;
    }
    inp.disabled = !WRITABLE;
    inputs[k] = inp;
    f.appendChild(inp);
    p.appendChild(f);
  }
  if (WRITABLE) {
    const row = el("div", "row");
    const b = el("button", "act primary", "Save " + title.toLowerCase());
    b.onclick = async () => {
      const patch = {};
      for (const [k, inp] of Object.entries(inputs)) {
        patch[k] = inp.type === "checkbox" ? inp.checked : inp.value;
      }
      try {
        const r = await post("api/settings/" + section, { values: patch });
        const n = Object.keys(r.changed || {}).length;
        toast(n ? `Saved ${n} change${n > 1 ? "s" : ""}` : "No changes");
      } catch (e) { toast(e.message, true); }
    };
    row.appendChild(b);
    p.appendChild(row);
  }
  return p;
}

async function addRoot(path) {
  if (!path || !path.trim()) return;
  try { await post("api/roots", { path }); toast("Now indexing " + path); renderSettings(); }
  catch (e) { toast(e.message, true); }
}

// directory picker ----------------------------------------------------
let pickPath = null;
async function openPicker(start) {
  $("pick").classList.add("open");
  await loadPick(start);
}
async function loadPick(path) {
  let d;
  try { d = await getJSON("api/fs" + (path ? "?path=" + encodeURIComponent(path) : "")); }
  catch (e) { toast(e.message, true); return; }
  pickPath = d.path;
  $("pick-cur").textContent = d.path;
  const list = $("pick-list");
  list.textContent = "";
  $("pick-up").disabled = !d.parent;
  for (const name of d.dirs) {
    const r = el("div", "row");
    r.appendChild(el("span", "grow", "📁  " + name));
    r.onclick = () => loadPick(d.path.replace(/\/$/, "") + "/" + name);
    list.appendChild(r);
  }
  if (!d.dirs.length) list.appendChild(el("div", "hint", "No subfolders."));
}
$("pick-up").onclick = async () => {
  const d = await getJSON("api/fs?path=" + encodeURIComponent(pickPath));
  if (d.parent) loadPick(d.parent);
};
$("pick-cancel").onclick = () => $("pick").classList.remove("open");
$("pick-ok").onclick = async () => {
  $("pick").classList.remove("open");
  await addRoot(pickPath);
};
$("pick").onclick = (e) => { if (e.target.id === "pick") $("pick").classList.remove("open"); };

// jobs ----------------------------------------------------------------
const JOBS = [
  { name: "scan", label: "Scan", hint: "Crawl roots, extract text and metadata." },
  { name: "deep", label: "Deep", hint: "VLM captions, OCR, transcripts. Safe to stop; resumes." },
  { name: "reindex", label: "Reindex", hint: "Re-queue files whose extractor or model changed." },
  { name: "purge", label: "Purge", hint: "Erase indexed data of removed files. Cannot be undone." },
];
let selectedJob = null;

async function renderJobs() {
  const body = $("jobs-body");
  let d;
  try { d = await getJSON("api/jobs"); }
  catch (e) { return; }
  const running = d.jobs.find(j => j.running);

  if (!body._built) {
    body.textContent = "";
    const p = panel("Run a job", WRITABLE
      ? "One job runs at a time — scan and deep both write the index."
      : "Disabled: this server is bound to a non-loopback address.");
    const row = el("div", "row");
    for (const j of JOBS) {
      const b = el("button", "act" + (j.name === "purge" ? " danger" : ""), j.label);
      b.id = "run-" + j.name;
      b.title = j.hint;
      b.onclick = async () => {
        if (j.name === "purge" && !confirm(
          "Permanently erase indexed data for files removed from the index?\n\n" +
          "Files on disk are never touched. This cannot be undone.")) return;
        try { const r = await post("api/jobs", { name: j.name }); selectedJob = r.id; toast("Started " + j.label); renderJobs(); }
        catch (e) { toast(e.message, true); }
      };
      row.appendChild(b);
    }
    p.appendChild(row);
    body.appendChild(p);
    const lp = panel("Recent jobs", "");
    lp.id = "job-list-panel";
    body.appendChild(lp);
    const op = panel("Output", "");
    op.id = "job-out-panel";
    body.appendChild(op);
    body._built = true;
  }

  for (const j of JOBS) {
    const b = $("run-" + j.name);
    if (b) b.disabled = !WRITABLE || !!running;
  }

  const lp = $("job-list-panel");
  while (lp.children.length > 1) lp.removeChild(lp.lastChild);
  if (!d.jobs.length) lp.appendChild(el("div", "hint", "Nothing has run yet."));
  for (const j of d.jobs.slice(0, 8)) {
    const r = el("div", "row");
    r.style.cursor = "pointer";
    const dot = el("span", "dot " + (j.running ? "run" : "idle"));
    const label = j.running
      ? `${j.name} — running ${j.elapsed}s${j.stopping ? " (stopping)" : ""}`
      : `${j.name} — ${j.returncode === 0 ? "done" : "exit " + j.returncode} in ${j.elapsed}s`;
    const span = el("span", "grow", label);
    r.append(dot, span);
    if (j.running && WRITABLE) {
      const s = el("button", "act danger", j.stopping ? "Force stop" : "Stop");
      s.onclick = async (e) => {
        e.stopPropagation();
        try { await post("api/jobs/" + j.id + "/stop", { force: j.stopping }); renderJobs(); }
        catch (err) { toast(err.message, true); }
      };
      r.appendChild(s);
    }
    r.onclick = () => { selectedJob = j.id; renderJobs(); };
    lp.appendChild(r);
  }

  if (selectedJob == null && running) selectedJob = running.id;
  if (selectedJob == null && d.jobs.length) selectedJob = d.jobs[0].id;
  const op = $("job-out-panel");
  while (op.children.length > 1) op.removeChild(op.lastChild);
  if (selectedJob == null) { op.appendChild(el("div", "hint", "Select a job to see its output.")); return; }
  try {
    const j = await getJSON("api/jobs/" + selectedJob);
    op.appendChild(el("div", "hint", `${j.name} · ${j.running ? "running" : "finished"} · ${j.elapsed}s`));
    const pre = el("pre", "out", (j.output || []).join("\n") || "(no output yet)");
    op.appendChild(pre);
    if (j.running) pre.scrollTop = pre.scrollHeight;
  } catch (e) { op.appendChild(el("div", "hint", e.message)); }
}

// status --------------------------------------------------------------
async function renderStatus() {
  const body = $("status-body");
  let s;
  try { s = await getJSON("api/status"); }
  catch (e) { return; }
  body.textContent = "";

  const t1 = s.queue.tier1 || {}, t2 = s.queue.tier2 || {};
  const qp = panel("Queue", "Pending work by tier. Tier 1 is the fast text pass; tier 2 is the deep pass.");
  const qt = el("table");
  qt.innerHTML = "<tr><th>Tier</th><th class='num'>Pending</th><th class='num'>Done</th><th class='num'>Failed</th></tr>";
  const pend2 = (t2.pending_deep || 0) + (t2.pending_transcript || 0) + (t2.pending_summary || 0);
  for (const [name, pending, done, failed] of [
    ["1 (metadata)", t1.pending_metadata || 0, t1.done || 0, t1.failed || 0],
    ["2 (deep)", pend2, t2.done || 0, t2.failed || 0],
  ]) {
    const tr = el("tr");
    tr.append(el("td", "", name), el("td", "num", pending.toLocaleString()),
              el("td", "num", done.toLocaleString()), el("td", "num", failed.toLocaleString()));
    qt.appendChild(tr);
  }
  qp.appendChild(qt);
  body.appendChild(qp);

  const kp = panel("Files", `${s.totals.files.toLocaleString()} files · ` +
    `${s.totals.chunks.toLocaleString()} chunks (${s.totals.embedded.toLocaleString()} embedded) · ` +
    `index ${human(s.db_bytes)}`);
  const kt = el("table");
  kt.innerHTML = "<tr><th>Kind</th><th class='num'>Count</th><th class='num'>Size</th></tr>";
  for (const k of s.kinds) {
    const tr = el("tr");
    tr.append(el("td", "", k.kind), el("td", "num", k.n.toLocaleString()), el("td", "num", human(k.size)));
    kt.appendChild(tr);
  }
  kp.appendChild(kt);
  body.appendChild(kp);

  const fp = panel("Recent failures", s.failures.length ? "" : "None — nothing has failed.");
  if (s.failures.length) {
    const ft = el("table");
    ft.innerHTML = "<tr><th>Tier</th><th>File</th><th>Error</th></tr>";
    for (const f of s.failures) {
      const tr = el("tr");
      tr.append(el("td", "", String(f.tier)), el("td", "", f.path), el("td", "", f.error));
      ft.appendChild(tr);
    }
    fp.appendChild(ft);
  }
  body.appendChild(fp);
}

// library -------------------------------------------------------------
async function loadSummary() {
  const s = await (await fetch("api/summary")).json();
  const chips = $("chips");
  const mk = (label, kind, n) => {
    const c = el("button", "chip" + (state.kind === kind ? " on" : ""),
      n === undefined ? label : `${label} ${n.toLocaleString()}`);
    c.onclick = () => { state.kind = kind; refresh();
      [...chips.children].forEach(x => x.classList.remove("on")); c.classList.add("on"); };
    chips.appendChild(c);
  };
  mk("All", "", s.total);
  for (const k of Object.keys(s.kinds).sort((a, b) => s.kinds[b] - s.kinds[a]))
    mk(k, k, s.kinds[k]);
}

function card(f) {
  const c = el("div", "card");
  const t = el("div", "thumb");
  if (["image", "video", "pdf"].includes(f.kind)) {
    const img = el("img");
    img.loading = "lazy";
    img.src = "thumb/" + f.id;
    img.onerror = () => { t.textContent = ""; t.appendChild(el("span", "ph", ICONS[f.kind] || "📦")); };
    t.appendChild(img);
  } else {
    t.appendChild(el("span", "ph", ICONS[f.kind] || "📦"));
  }
  const body = el("div", "body");
  body.appendChild(el("div", "name", f.path.split("/").pop()));
  const badges = el("div", "badges");
  badges.appendChild(el("span", "badge", f.kind));
  if (f.vlm_type && f.vlm_type !== "other") badges.appendChild(el("span", "badge", f.vlm_type));
  if (f.degraded) badges.appendChild(el("span", "badge", "degraded"));
  body.appendChild(badges);
  body.appendChild(el("div", "cap-text",
    f.caption || (f.tier2_status ? "(no caption)" : "(not yet processed by deep)")));
  c.append(t, body);
  c.onclick = () => openModal(f.id);
  return c;
}

async function loadPage() {
  if (state.busy || state.done) return;
  state.busy = true;
  const p = new URLSearchParams({ q: state.q, kind: state.kind,
    captioned: state.captioned ? "1" : "", offset: state.offset, limit: PAGE });
  const r = await (await fetch("api/files?" + p)).json();
  state.total = r.total;
  for (const f of r.files) $("grid").appendChild(card(f));
  state.offset += r.files.length;
  state.done = state.offset >= r.total || r.files.length === 0;
  $("count").textContent = `${state.offset.toLocaleString()} of ${r.total.toLocaleString()}`;
  $("empty").style.display = r.total === 0 ? "block" : "none";
  state.busy = false;
}

function refresh() {
  state.offset = 0; state.done = false;
  $("grid").textContent = "";
  loadPage();
}

// modal ---------------------------------------------------------------
function section(title, contentEl) {
  const s = el("section");
  s.appendChild(el("h3", "", title));
  s.appendChild(contentEl);
  return s;
}

async function openModal(id) {
  const d = await (await fetch("api/file/" + id)).json();
  const media = $("m-media"), side = $("m-side");
  media.textContent = ""; side.textContent = "";
  if (d.kind === "image") {
    const img = el("img"); img.src = "media/" + d.id; media.appendChild(img);
  } else if (d.kind === "video") {
    const v = el("video"); v.controls = true; v.src = "media/" + d.id;
    media.appendChild(v);
  } else if (d.kind === "audio") {
    const a = el("audio"); a.controls = true; a.src = "media/" + d.id;
    media.appendChild(a);
  } else {
    media.appendChild(el("span", "ph", ICONS[d.kind] || "📦"));
  }
  side.appendChild(el("h2", "", d.path.split("/").pop()));
  const path = el("div", "path", d.path + "  (click to copy)");
  path.onclick = () => navigator.clipboard.writeText(d.path);
  side.appendChild(path);
  const info = el("div", "badges");
  info.appendChild(el("span", "badge", d.kind));
  info.appendChild(el("span", "badge", (d.size / 1048576).toFixed(2) + " MB"));
  info.appendChild(el("span", "badge", new Date(d.mtime * 1000).toLocaleString()));
  side.appendChild(info);
  if (d.error) side.appendChild(section("Processing error", el("p", "", d.error)));
  for (const c of d.content) side.appendChild(renderStage(c));
  $("overlay").classList.add("open");
}

function renderStage(c) {
  const m = typeof c.meta === "object" && c.meta !== null ? c.meta : null;
  if (c.stage === "vlm_image" && m) {
    const box = el("div");
    if (m.description) box.appendChild(el("p", "", m.description));
    if (m.ocr_text) {
      box.appendChild(el("h3", "", "Text in image"));
      box.appendChild(el("pre", "", m.ocr_text));
    }
    if (m.objects && m.objects.length) {
      const b = el("div", "badges");
      for (const o of m.objects) b.appendChild(el("span", "badge", o));
      box.appendChild(b);
    }
    if (m.inferred_context) box.appendChild(el("p", "", "Context: " + m.inferred_context));
    return section("Image analysis" + (c.degraded ? " (degraded)" : ""), box);
  }
  if (c.stage === "exif" && m) {
    const pre = el("pre", "", Object.entries(m)
      .map(([k, v]) => k + ": " + JSON.stringify(v)).join("\n"));
    return section("EXIF", pre);
  }
  const titles = { text: "Extracted text", pdf_text: "PDF text", pdf_scan_vlm: "Scanned PDF",
    whisper: "Transcript", audio_summary: "Audio summary", video_scenes: "Scenes",
    video_transcript: "Video transcript", video_summary: "Video summary" };
  const body = (c.body || "").slice(0, 20000);
  const node = ["video_summary", "audio_summary"].includes(c.stage)
    ? el("p", "", body)
    : el("pre", "", body || (c.degraded ? "(nothing could be extracted)" : ""));
  return section((titles[c.stage] || c.stage) + (c.degraded ? " (degraded)" : ""), node);
}

function closeModal() {
  $("overlay").classList.remove("open");
  $("m-media").textContent = "";  // stop any playing video
}
$("overlay").onclick = (e) => { if (e.target.id === "overlay") closeModal(); };
$("m-close").onclick = closeModal;
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if ($("pick").classList.contains("open")) $("pick").classList.remove("open");
  else closeModal();
});

// wiring --------------------------------------------------------------
let debounce;
$("q").oninput = () => {
  clearTimeout(debounce);
  debounce = setTimeout(() => { state.q = $("q").value; refresh(); }, 300);
};
$("captioned").onchange = () => { state.captioned = $("captioned").checked; refresh(); };
new IntersectionObserver((es) => { if (es[0].isIntersecting && current === "library") loadPage(); })
  .observe($("sentinel"));
buildTabs();
const initial = (location.hash || "").replace("#", "");
show(VIEWS.some(v => v.id === initial) ? initial : "library");
loadSummary();
loadPage();
</script>
</body>
</html>
"""
