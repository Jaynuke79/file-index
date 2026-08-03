browse server is readable via DNS rebinding and fully exposed when bound beyond localhost

**Labels:** `type:bug`, `type:security`, `priority:p1`  

---

## Problem
The browse server authenticates nothing and never validates the `Host` header
(`file_index/web.py:268`, `file_index/cli.py:408`). Two consequences:

1. **DNS rebinding against the default localhost bind.** A malicious website the user
   visits can rebind its hostname to `127.0.0.1` and issue same-origin requests to
   `http://127.0.0.1:8765/`. Because the handler ignores `Host`, the site can read
   `api/files`, `api/file/<id>`, and `media/<id>` — i.e. exfiltrate the extracted
   text, captions, transcripts, and raw bytes of everything indexed (the user's
   whitelisted home directories). This defeats the project's core "no data leaves
   the machine" guarantee without any misconfiguration by the user.
2. **`--host` exposes everything unauthenticated.** `browse --host 0.0.0.0` serves
   every indexed file's original bytes to anyone on the network, with no warning at
   startup.

## Proposed Solution
Validate the `Host` header against an allowlist (`127.0.0.1[:port]`,
`localhost[:port]`, and the explicitly bound host) and reject others with 403 — this
is the standard localhost-server rebinding defense and is a few lines in
`Handler._route`. For non-localhost binds, either print a prominent warning or
require a `--token` that must be present as a query parameter/cookie. Setting
`Access-Control-Allow-Origin` is not needed (its absence already blocks normal CORS
reads); rebinding is the gap.

## Acceptance Criteria
- [ ] Requests with a `Host` header not matching the bound/allowed hosts receive 403
      and no body (covered by a test).
- [ ] `browse --host <non-loopback>` prints an explicit exposure warning (or refuses
      without an auth token, if the token approach is chosen).
