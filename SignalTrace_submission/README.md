# SignalTrace V2

SignalTrace is a Python 3.10+ command-line investigator for the supplied payment,
shipment, and complaint data. It uses an LLM only to classify complaint meaning.
All extraction validation, record correlation, evidence selection, and verdicts are
deterministic Python code.

The implementation uses only the Python standard library, so there are no packages
to install.

## Run it

Offline smoke test (guaranteed to make no network requests):

```bash
python3 main.py \
  --payments data/smoke/payments.csv \
  --shipments data/smoke/shipments.jsonl \
  --complaints data/smoke/complaints.jsonl \
  --extractions data/smoke/extractions.json \
  --output smoke_result.json
```

Required challenge result:

```bash
python3 main.py \
  --payments data/challenge/payments.csv \
  --shipments data/challenge/shipments.jsonl \
  --complaints data/challenge/complaints.jsonl \
  --extractions data/challenge/extractions.json \
  --output result_offline.json
```

Live mode automatically loads `.env` from the project folder. Values already
exported by the shell take precedence. When both keys are available, automatic
selection uses the verified Groq integration; choose either provider explicitly
with `--provider`.

```bash
python3 main.py \
  --payments data/sample/payments.csv \
  --shipments data/sample/shipments.jsonl \
  --complaints data/sample/complaints.jsonl \
  --output result_live.json
```

Select a provider with `--provider gemini` or `--provider groq`. Use `--env-file`
to select a different dotenv file. Override the provider's
default model with `--model MODEL_ID`, `GEMINI_MODEL`, or `GROQ_MODEL`. `--timeout`
sets the per-request timeout in seconds. For low-quota accounts, use
`--request-delay 7`; transient failures are retried with bounded exponential
backoff. API keys are read only from environment variables and must never be
committed.

When a live request fails, that complaint receives the required safe fallback and
processing continues. Offline extraction objects are also treated as untrusted and
validated against the original complaint.

## Verify it

```bash
python3 -m unittest -v
```

The suite compares the complete smoke output and exercises validation,
deduplication, extraction sanitization, evidence tie-breaking, time boundaries, and
verdict precedence.

## Optional SQLite audit history

Add `--database` to any offline or live command when you want to retain an
append-only investigation history:

```bash
python3 main.py \
  --payments data/challenge/payments.csv \
  --shipments data/challenge/shipments.jsonl \
  --complaints data/challenge/complaints.jsonl \
  --extractions data/challenge/extractions.json \
  --output result_offline.json \
  --database signaltrace_history.db
```

The database contains three related tables:

- `runs`: execution mode, provider/model metadata, file paths, error count, and a
  complete JSON snapshot.
- `source_stats`: validated counts for each source.
- `cases`: queryable ticket outcomes and evidence IDs.

Each execution creates a new run rather than overwriting history. The JSON output
remains the authoritative submission artifact; SQLite is an optional audit and
analysis layer. Inspect recent runs with:

```bash
sqlite3 signaltrace_history.db \
  "SELECT run_id, created_at_utc, mode, provider, llm_errors FROM runs;"
```
