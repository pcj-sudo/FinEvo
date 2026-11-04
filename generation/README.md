# Task generation

The author and the independent solver use the same model. Configure an OpenAI-compatible Qwen3.6-27B endpoint with `OPENAI_BASE_URL` and `OPENAI_API_KEY`; both model entries in `configs/generation.json` must identify the same served model. OpenHands SDK is pinned to 1.47.0.

From the repository root, register the included prepared workbook and generate a new task:

```bash
python scripts/import_prepared_source.py
rsi-generate --root generation --harness openhands run financialstatements1-new-modeling-001 \
  --source financialstatements1-example --type modeling --reuse-source
```

Use `--type debugging` for formula repair. `--reuse-source` skips cleaning for a prepared source; omitting it starts a new cleaning and independent audit loop. For an unprepared workbook directory:

```bash
rsi-generate --root generation ingest my-source /path/to/workbooks
rsi-generate --root generation --harness openhands run my-task \
  --source my-source --type modeling
```

Construction produces a Harbor task and private reference/target artifacts. Fixed checks run before a fresh-context solver. Successful solving triggers trajectory-guided hardening; normal failure triggers author review and either revision or acceptance. Any content change is reconstructed and checked again. Reference workbooks, private targets and author histories are not mounted into the solver environment.

```bash
rsi-generate --root generation inspect my-task
rsi-generate --root generation resume my-task
rsi-generate --root generation export my-task --to /path/to/exported-task
```

Model requests, task execution and generated evidence are written only when these commands are run. API credentials come from the environment; no historical sessions or credentials are included.
