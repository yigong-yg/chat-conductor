-- Chat Evidence dataset schema, version 1 (packaged runtime copy).
--
-- Derived from the approved design artifact docs/prototypes/chat-evidence/schema.sql
-- and kept to the same four domain tables and columns. Differences:
--   * No PRAGMA/BEGIN/COMMIT lines: the application applies these statements
--     inside its own BEGIN IMMEDIATE transaction, and every connection enables
--     foreign_keys and recursive_triggers itself.
--   * messages adds CHECKs for the canonical UTC format, a nonempty original
--     timestamp and zone, and order_basis/source_order consistency.
--   * The per-insert batch-size guards bound a batch's row count by its rowid
--     span instead of COUNT(*). Rows of one batch receive contiguous rowids
--     inside the single import transaction and are never deleted, so the span
--     is never below the true count: the guard can only abort earlier. COUNT(*)
--     on every insert made each import quadratic in its size. The added
--     origins_by_revision index serves that lookup.
-- The writer still checks exact batch totals before COMMIT, and
-- prior_import_is_complete still checks exact totals of the previous batch.

CREATE TABLE dataset (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    dataset_id TEXT NOT NULL UNIQUE CHECK (length(dataset_id) > 0),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    created_at TEXT NOT NULL
);

CREATE TABLE import_batches (
    revision INTEGER PRIMARY KEY CHECK (revision > 0),
    import_id TEXT NOT NULL UNIQUE,
    export_sha256 TEXT NOT NULL CHECK (
        length(export_sha256) = 64 AND export_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    adapter TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    profile_sha256 TEXT NOT NULL CHECK (
        length(profile_sha256) = 64 AND profile_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    raw_manifest_path TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    record_count INTEGER NOT NULL CHECK (record_count >= 0),
    appended_count INTEGER NOT NULL CHECK (
        appended_count >= 0 AND appended_count <= record_count
    ),
    UNIQUE (export_sha256, adapter, adapter_version, profile_sha256)
);

CREATE TABLE messages (
    -- m_ + full SHA-256 of the versioned, namespaced native identity tuple.
    id TEXT PRIMARY KEY NOT NULL CHECK (
        length(id) = 66 AND substr(id, 1, 2) = 'm_'
        AND substr(id, 3) NOT GLOB '*[^0-9a-f]*'
    ),
    source_namespace TEXT NOT NULL CHECK (length(source_namespace) > 0),
    conversation_id TEXT NOT NULL CHECK (length(conversation_id) > 0),
    native_message_id TEXT NOT NULL CHECK (length(native_message_id) > 0),
    sender_id TEXT NOT NULL CHECK (length(sender_id) > 0),
    ts_utc TEXT NOT NULL CHECK (
        length(ts_utc) = 27
        AND ts_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
    ),
    ts_original TEXT NOT NULL CHECK (length(ts_original) > 0),
    source_timezone TEXT NOT NULL CHECK (length(source_timezone) > 0),
    source_order TEXT NOT NULL,
    order_basis TEXT NOT NULL CHECK (order_basis IN ('native', 'timestamp_only')),
    kind TEXT NOT NULL CHECK (kind IN (
        'text', 'voice', 'image', 'video', 'file', 'transfer', 'call',
        'recall', 'system', 'other'
    )),
    text_origin TEXT NOT NULL CHECK (
        text_origin IN ('export_text', 'export_transcript', 'placeholder')
    ),
    text TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL CHECK (
        length(payload_sha256) = 64 AND payload_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    first_seen_revision INTEGER NOT NULL REFERENCES import_batches(revision),
    UNIQUE (source_namespace, conversation_id, native_message_id),
    CHECK (
        (order_basis = 'native' AND length(source_order) > 0)
        OR (order_basis = 'timestamp_only' AND source_order = '')
    )
);

CREATE TABLE message_origins (
    import_revision INTEGER NOT NULL REFERENCES import_batches(revision),
    -- Relative file path + JSON pointer, line number, or equivalent stable locator.
    raw_locator TEXT NOT NULL CHECK (length(raw_locator) > 0),
    message_id TEXT NOT NULL REFERENCES messages(id),
    PRIMARY KEY (import_revision, raw_locator)
);

CREATE INDEX messages_by_conversation_time ON messages (
    source_namespace, conversation_id, ts_utc, source_order, id
);
CREATE INDEX messages_by_sender_time ON messages (
    source_namespace, sender_id, ts_utc, id
);
CREATE INDEX messages_by_revision ON messages (first_seen_revision);
CREATE INDEX origins_by_message ON message_origins (message_id, import_revision);
CREATE INDEX origins_by_revision ON message_origins (import_revision);
-- Deliberately no UNIQUE constraint or index on payload_sha256: equal text and
-- timestamps may belong to different real message occurrences.

CREATE TRIGGER dataset_no_update BEFORE UPDATE ON dataset
BEGIN SELECT RAISE(ABORT, 'dataset identity is immutable'); END;
CREATE TRIGGER dataset_no_delete BEFORE DELETE ON dataset
BEGIN SELECT RAISE(ABORT, 'dataset identity is immutable'); END;
CREATE TRIGGER import_batches_no_update BEFORE UPDATE ON import_batches
BEGIN SELECT RAISE(ABORT, 'import history is append-only'); END;
CREATE TRIGGER import_batches_no_delete BEFORE DELETE ON import_batches
BEGIN SELECT RAISE(ABORT, 'import history is append-only'); END;
CREATE TRIGGER messages_no_update BEFORE UPDATE ON messages
BEGIN SELECT RAISE(ABORT, 'message evidence is immutable'); END;
CREATE TRIGGER messages_no_delete BEFORE DELETE ON messages
BEGIN SELECT RAISE(ABORT, 'message evidence is immutable'); END;
CREATE TRIGGER message_origins_no_update BEFORE UPDATE ON message_origins
BEGIN SELECT RAISE(ABORT, 'origin evidence is append-only'); END;
CREATE TRIGGER message_origins_no_delete BEFORE DELETE ON message_origins
BEGIN SELECT RAISE(ABORT, 'origin evidence is append-only'); END;

-- A committed import is one monotonic snapshot revision. Writers use BEGIN
-- IMMEDIATE and allocate the next revision inside the same transaction.
CREATE TRIGGER import_revision_is_next BEFORE INSERT ON import_batches
WHEN NEW.revision != COALESCE((SELECT MAX(revision) FROM import_batches), 0) + 1
BEGIN SELECT RAISE(ABORT, 'import revision must be consecutive'); END;

CREATE TRIGGER import_requires_dataset BEFORE INSERT ON import_batches
WHEN (SELECT COUNT(*) FROM dataset) != 1
BEGIN SELECT RAISE(ABORT, 'initialize dataset identity before import'); END;

-- Only the new revision may receive rows. Fixed batch totals seal a completed
-- revision against further inserts even while it remains the latest revision.
-- The writer must check exact totals before COMMIT; partial batches roll back.
CREATE TRIGGER message_insert_is_current BEFORE INSERT ON messages
WHEN NEW.first_seen_revision != COALESCE((SELECT MAX(revision) FROM import_batches), 0)
  OR COALESCE(
       (SELECT MAX(rowid) FROM messages WHERE first_seen_revision = NEW.first_seen_revision)
       - (SELECT MIN(rowid) FROM messages WHERE first_seen_revision = NEW.first_seen_revision) + 1,
       0)
     >= (SELECT appended_count FROM import_batches WHERE revision = NEW.first_seen_revision)
BEGIN SELECT RAISE(ABORT, 'message insert exceeds the current import'); END;

CREATE TRIGGER origin_insert_is_current BEFORE INSERT ON message_origins
WHEN NEW.import_revision != COALESCE((SELECT MAX(revision) FROM import_batches), 0)
  OR NEW.import_revision < (SELECT first_seen_revision FROM messages WHERE id = NEW.message_id)
  OR COALESCE(
       (SELECT MAX(rowid) FROM message_origins WHERE import_revision = NEW.import_revision)
       - (SELECT MIN(rowid) FROM message_origins WHERE import_revision = NEW.import_revision) + 1,
       0)
     >= (SELECT record_count FROM import_batches WHERE revision = NEW.import_revision)
BEGIN SELECT RAISE(ABORT, 'origin insert exceeds the current import'); END;

CREATE TRIGGER prior_import_is_complete BEFORE INSERT ON import_batches
WHEN EXISTS (
    SELECT 1 FROM import_batches b
    WHERE b.revision = (SELECT MAX(revision) FROM import_batches)
      AND (
        b.appended_count != (SELECT COUNT(*) FROM messages m WHERE m.first_seen_revision = b.revision)
        OR b.record_count != (SELECT COUNT(*) FROM message_origins o WHERE o.import_revision = b.revision)
      )
)
BEGIN SELECT RAISE(ABORT, 'prior import totals are incomplete'); END;
