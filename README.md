# chat-conductor
Life is a symphony, your chat history is the rehearsal.

## v1 CLI

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m chat_conductor index input\data-...
.\.venv\Scripts\python.exe -m chat_conductor search "MCP" --limit 5
.\.venv\Scripts\python.exe -m chat_conductor rehearse "MCP"
.\.venv\Scripts\python.exe -m chat_conductor eval path\to\cases.json --k 10
.\.venv\Scripts\python.exe -m chat_conductor serve
.\.venv\Scripts\python.exe -m chat_conductor status
```

`index` accepts either a Claude export `.zip` or an already-unzipped export
directory. By default, the SQLite cache lives outside the repo at the OS user
data path. Override it with `--index <path>` or `CHAT_CONDUCTOR_INDEX`.

The real `input/` export and `.chat-conductor/` cache are gitignored because
both contain sensitive chat data or derived sensitive text.

`rehearse` expands each hit by one neighboring turn on either side and packs a
few coherent windows under a shared 2,400-token default budget. On the current
real index, that budget fits roughly 1-2 full recall windows for common technical
queries without truncation.

`eval` accepts JSON or JSONL cases shaped like
`{"query": "...", "expected_turn_id": "turn_...", "k": 10}`. Real eval files can
contain private derived identifiers or revealing query choices; keep them local
with `eval/*.private.json`, `eval/*.local.json`, or `.eval/`.

