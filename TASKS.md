# TASKS.md

## Branch and PR task log

See [AGENTS.md](AGENTS.md) for the research goals, model contract, and verification
commands. Start a section when branch work begins, link its PR once opened, and
append dated entries for decisions, results, and blockers. Check an item once its
acceptance evidence is recorded. Keep failed checks and their reproduction links
in the log when later runs produce a different outcome.

### Branch map as of 2026-11-22

| Branch | Head | Base and status |
| --- | --- | --- |
| `main` | `1e6f360` | GitHub default. Original dynamic-token-pooling research baseline. |
| `feat/first-kv-cache` | `dafb96d` | Based on `main`. Open draft [PR #1](https://github.com/phdpersoncode-beep/dynamic-pooling/pull/1). |
| `review/pr1-cache-correctness` | `29bc361` | Based on `feat/first-kv-cache`. Open draft [PR #2](https://github.com/phdpersoncode-beep/dynamic-pooling/pull/2). Latest implementation and experiment evidence. |
| `debug/fp32-cache-numerics` | `c4ed9a2` | Branched from review commit `232f2f5`. Its head is an ancestor of PR #2's head. |
| `work` | `1e6f360` | Local checkout at the `main` commit. Contains these editable documentation drafts. |

### Template for a new branch or PR

```markdown
## `<branch>` · <PR link or pending>

- Base branch and starting commit
- Scope and acceptance checks
  - [ ] <check and evidence link>
- YYYY-MM-DD · <decision, result or blocker> · <commit, command or artifact>
- Next action and any dependency
```
