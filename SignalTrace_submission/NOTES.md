# Implementation notes

**Implementation approach:** Standard-library Python validates every source, keeps the first valid occurrence of each ID, sorts events chronologically, sanitizes LLM/fixture extractions against the original complaint, and applies deterministic evidence rules. Offline mode never initializes a network client. Gemini and Groq provide structured extraction only; neither chooses verdicts.

**Tests performed:** The complete smoke result matches the supplied expected JSON. Sixteen unit tests cover dotenv precedence, provider selection, offline isolation, live failure fallbacks, timestamp and row validation, duplicate accounting, chronological tie-breaking, order-ID boundaries and ambiguity, every signal boundary, verdict precedence, and SQLite round trips. Challenge statistics and all 46 cases passed an independent contract audit.

**Optional enhancement:** `--database` appends each run, source statistics, cases, evidence, and a JSON snapshot to normalized SQLite tables. JSON behavior is unchanged. No keys or complaint messages are stored.

**Known limitations:** Live classification depends on provider/model availability. Transient requests use bounded retry/backoff; terminal failures are isolated per complaint. SQLite is local and has no multi-user service layer.

**Relevant risk:** Complaint text is adversarial. It is delimited as data, while Python independently constrains issue labels, order IDs, evidence, and verdicts.

**Live result:** Groq `openai/gpt-oss-20b` processed all 15 accepted sample complaints with zero API errors, matching validated fixtures. Gemini `gemini-3.5-flash-lite` also passed a real structured extraction check. No secret is submitted.
