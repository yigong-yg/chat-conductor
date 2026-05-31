from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Iterable


def normalize_text(value: str) -> str:
    """Normalize text for stable hashing without flattening meaningful lines."""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    return "\n".join(lines).strip()


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def turn_text(human_text: str, assistant_text: str) -> str:
    # No "Human:" / "Assistant:" scaffolding: those literal words would otherwise
    # match every indexed turn (bm25 distortion) and ride along in returned text,
    # wasting rehearse budget. Role is preserved via the separate human_text /
    # assistant_text FTS columns. content_hash is over this text, so dropping the
    # constant labels is neutral for supersede (within a source_key) but cleaner.
    parts = [part for part in (human_text, assistant_text) if part]
    return normalize_text("\n\n".join(parts))


# FTS5 trigram cannot tokenize a term shorter than 3 codepoints, and Claude
# exports are bilingual: most Chinese words are 2 characters (信仰, 因果), as are
# "AI" / "Yi". Those terms are routed to a LIKE substring clause instead (spec
# §10 broad-net, pulled forward for short terms only); long terms still use FTS5
# MATCH / bm25. Splitting the two also stops a short term from AND-zeroing the
# whole query.
TRIGRAM_MIN = 3


def split_query_terms(query: str) -> tuple[list[str], list[str]]:
    """Split a query into (fts_terms >= 3 codepoints, like_terms < 3 codepoints)."""
    fts_terms: list[str] = []
    like_terms: list[str] = []
    for term in re.split(r"\s+", query.strip()):
        if not term:
            continue
        if len(term) >= TRIGRAM_MIN:
            fts_terms.append(term)
        else:
            like_terms.append(term)
    return fts_terms, like_terms


def build_fts_match(terms: list[str], column: str | None = None) -> str:
    """AND the (already long-enough) terms into an FTS5 MATCH expression."""
    body = " AND ".join(_quote_fts_term(term) for term in terms)
    if column:
        return f"{column} : ({body})"
    return body


def _quote_fts_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return ""
