# chat-conductor
Life is a symphony, your chat history is the rehearsal.

## v1 CLI

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m chat_conductor index input\data-...
.\.venv\Scripts\python.exe -m chat_conductor search "MCP" --limit 5
.\.venv\Scripts\python.exe -m chat_conductor status
```

`index` accepts either a Claude export `.zip` or an already-unzipped export
directory. By default, the SQLite cache lives outside the repo at the OS user
data path. Override it with `--index <path>` or `CHAT_CONDUCTOR_INDEX`.

The real `input/` export and `.chat-conductor/` cache are gitignored because
both contain sensitive chat data or derived sensitive text.

