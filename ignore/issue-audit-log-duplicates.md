Every organize run re-appends the last 200 audit rows to audit.log

**Labels:** `type:bug`, `priority:p2`  

---

## Problem
`_append_audit_file` (`file_index/cli.py:340-349`) appends the most recent 200
`audit_log` DB rows to `audit.log` on every `organize --apply`, with no memory of
what was already written. The second and every later run re-appends rows from
earlier runs, so the plain-text log accumulates duplicates and its ordering no
longer reflects when operations happened — undermining the file's purpose as a
trustworthy audit trail.

## Proposed Solution
Track the last mirrored `audit_log.id` in the `meta` table and append only rows with
a greater id; or simpler, write the audit line at the moment each operation is
recorded (inside `Index.audit`) instead of batch-mirroring afterwards.

## Acceptance Criteria
- [ ] Running `organize --apply` twice produces each audit entry exactly once in
      `audit.log`.
