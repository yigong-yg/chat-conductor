-- Messages by one sender in one conversation containing a literal term,
-- within a half-open UTC window; placeholders excluded.
SELECT COUNT(*) AS message_count
FROM visible_messages
WHERE source_namespace = :source
  AND conversation_id = :conversation
  AND sender_id = :sender
  AND ts_utc >= :from_utc AND ts_utc < :to_utc
  AND text_origin != 'placeholder'
  AND text LIKE :pattern ESCAPE '\'
