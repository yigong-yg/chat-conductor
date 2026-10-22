"""Deterministic Markdown rendering of a successfully verified report.

Only the verifier calls this, with claims it has checked against the pinned
evidence. All source-derived strings are rendered inside code spans or fences
(with JSON-escaped forms that expose invisible characters), so exported text
cannot inject Markdown, links or HTML into the report.
"""

from __future__ import annotations

import json
import re
import unicodedata

from . import TOOL_NAME, TOOL_VERSION
from .runs import Run

LIMITATIONS = (
    "Verification checks citation and query-result integrity against the pinned evidence only. "
    "It does not establish that any statement is true, representative, relevant or correctly "
    "interpreted, and it says nothing about motive, meaning, personality or relationship outcomes.",
    "The corpus is what was imported through the pinned revision. Records missing from a later export "
    "are not deletion evidence, and corpus edges are not conversation boundaries.",
    "Statistics count exported message records under the recorded SQL definition and scope; they are "
    "not counts of real-world events.",
    "Absence findings cover only the listed queries and filters; no synonym, spelling-variant, "
    "media-content or cross-archive coverage is implied.",
    "Exported transcripts are labelled and were neither produced nor checked by this tool; media "
    "placeholders are never quoted as speech.",
    "Messages that share a timestamp and have timestamp-only ordering have uncertain relative order.",
    "Receipt hashes are unkeyed; verification replayed every referenced receipt against the canonical "
    "pinned evidence instead of trusting it.",
    "This report counts as verified only while its bytes match report_sha256 in verification.json. "
    "Any edit requires a new structured draft and a new verification.",
)
_TEXT_ORIGIN_LABELS = {
    "export_text": "exported message text",
    "export_transcript": "exported transcript (labelled; not transcribed or checked by this tool)",
}
_ESCAPED_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cn", "Cs", "Zl", "Zp"})


def render_report(run: Run, draft: dict, claims: list[dict]) -> str:
    counts = {kind: sum(1 for claim in claims if claim["type"] == kind) for kind in ("quote", "statistic", "not_found")}
    lines = [
        "# Chat Evidence report",
        "",
        "**Question** (unverified input, displayed as asked; not a finding):",
        "",
        *_fenced(draft["question"], "text"),
        "",
        f"- Question, JSON-escaped: {_code(_escaped(draft['question']))}",
        f"- Dataset: {_code(run.dataset_id)}",
        f"- Run: {_code(run.run_id)}, created {_code(run.created_at)}",
        f"- Corpus revision: {run.corpus_revision} (only imports up to and including revision "
        f"{run.corpus_revision} are in scope)",
        f"- Verified claims: {len(claims)} ({counts['quote']} quote, {counts['statistic']} statistic, "
        f"{counts['not_found']} not found)",
        f"- Rendered by: {TOOL_NAME} {TOOL_VERSION}",
        "",
        "Verified means that every citation and query result below matched the pinned evidence when "
        "replayed. It does not establish truth, motive, meaning, relevance or archive completeness.",
        "",
        "## Claims",
    ]
    for claim in claims:
        lines.append("")
        if claim["type"] == "quote":
            lines.extend(_quote(claim, run))
        elif claim["type"] == "statistic":
            lines.extend(_statistic(claim, run))
        else:
            lines.extend(_not_found(claim, run))
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in LIMITATIONS)
    return "\n".join(lines) + "\n"


def _quote(claim: dict, run: Run) -> list[str]:
    message = claim["message"]
    window = claim["window"]
    origins = "; ".join(f"revision {item['revision']} {_code(item['raw_locator'])}" for item in claim["origins"])
    lines = [
        f"### {claim['id']} · Quote",
        "",
        *_fenced(claim["text"], "text"),
        "",
        f"- Exact text, JSON-escaped: {_code(_escaped(claim['text']))}",
        f"- Message: {_code(message['id'])}",
        f"- Sender: {_code(message['sender_id'])}",
        f"- Time: {_code(message['ts_utc'])} UTC; original {_code(message['ts_original'])} in declared zone "
        f"{_code(message['source_timezone'])}",
        f"- Span: Unicode code points [{claim['start']}, {claim['end']}) of {len(message['text'])}",
        f"- Text origin: {_TEXT_ORIGIN_LABELS[message['text_origin']]}",
        f"- Kind: {_code(message['kind'])}",
        f"- Conversation: {_code(message['source_namespace'])} / {_code(message['conversation_id'])}",
        f"- Provenance (imports up to revision {run.corpus_revision}): {origins}",
        f"- Context: receipt {_code(claim['context_receipt_id'])} delivered {window['before_count']} earlier and "
        f"{window['after_count']} later messages of all kinds (radius {window['radius']})",
        f"- Corpus edges: earlier side {'reached' if window['edge_before'] else 'not reached'}, later side "
        f"{'reached' if window['edge_after'] else 'not reached'} (edges of the imported corpus only)",
        f"- Ordering: {_ordering(message, window)}",
    ]
    return lines


def _ordering(message: dict, window: dict) -> str:
    basis = "native source order" if message["order_basis"] == "native" else "timestamp only"
    if window["equal_time_order_uncertain"]:
        return f"{basis}; some messages in this context window share a timestamp and their order is uncertain"
    return basis


def _statistic(claim: dict, run: Run) -> list[str]:
    receipt = claim["receipt"]
    arguments = receipt["arguments"]
    scope = arguments["scope"]
    value = claim["value"]
    lines = [
        f"### {claim['id']} · Statistic",
        "",
        f"- Value: {_code(json.dumps(value, ensure_ascii=False))} ({_json_type(value)}) at row {claim['row']}, "
        f"column {_code(claim['column'])}",
        f"- Definition (as recorded; not verified to measure the intended concept): {_code(scope['definition'])}",
        f"- Unit: {_code(scope['unit'])}",
        f"- Filters: {_code(scope['filters'])}",
    ]
    for name, label in (("denominator", "Denominator"), ("period", "Period"), ("timezone", "Time zone")):
        if name in scope:
            lines.append(f"- {label}: {_code(scope[name])}")
    lines.append(f"- Missing-data limits: {_code(scope['missing_data'])}")
    for item in arguments["utc_parameters"]:
        lines.append(
            f"- Parameter {_code(item['name'])}: local {_code(item['local'])} in {_code(item['zone'])} = "
            f"{_code(item['utc'])} UTC"
        )
    snapshot = receipt["snapshot"]
    lines.extend([
        f"- Receipt: {_code(receipt['receipt_id'])}; replayed over a fresh snapshot of {snapshot['rows']} messages "
        f"at revision {run.corpus_revision}",
        "- SQL:",
        "",
        *_fenced(arguments["sql"], "sql"),
        "",
        "- Parameters:",
        "",
        *_fenced(json.dumps(arguments["parameters"], ensure_ascii=False, sort_keys=True, indent=2), "json"),
        "",
        "- Message counts are counts of exported message records, not of real-world events.",
    ])
    return lines


def _not_found(claim: dict, run: Run) -> list[str]:
    lines = [
        f"### {claim['id']} · Not found",
        "",
        f"No matching records were found for these queries and filters at corpus revision {run.corpus_revision}. "
        "This does not show that the event did not occur, that the archive is complete, or that synonyms, "
        "other spellings or other archives were covered.",
        "",
    ]
    for receipt in claim["receipts"]:
        arguments = receipt["arguments"]
        scope = receipt["scope"]
        filters = scope["filters"]
        window = scope["time_window"]
        lines.append(
            f"- Search {_code(receipt['receipt_id'])}: terms {_code(json.dumps(arguments['terms'], ensure_ascii=False))} "
            f"(all must match), case policy {_code(scope['case'])}: {scope['case_policy']}; "
            f"source {_code(_describe(filters['source_namespace']))}, conversation "
            f"{_code(_describe(filters['conversation_id']))}, sender {_code(_describe(filters['sender_id']))}, "
            f"kinds {_code(_describe(filters['kinds']))}, text origins {_code(_describe(filters['text_origins']))}; "
            f"time window {_window(window)}; 0 matches, complete, no continuation"
        )
    return lines


def _window(window: dict) -> str:
    parts = []
    for name in ("from", "to"):
        bound = window[name]
        if bound is None:
            parts.append(f"{name} unbounded")
        else:
            parts.append(f"{name} {_code(bound['local'])} = {_code(bound['utc'])} UTC")
    zone = f" in {_code(window['zone'])}" if window["zone"] else ""
    return f"half-open [from, to){zone}: " + ", ".join(parts)


def _describe(value: object) -> str:
    if value is None:
        return "any"
    if isinstance(value, list):
        return ", ".join(value)
    return str(value)


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "real"
    return "text"


def _escaped(text: str) -> str:
    """JSON string literal that also escapes invisible and format characters."""
    return "".join(
        json.dumps(char)[1:-1] if unicodedata.category(char) in _ESCAPED_CATEGORIES else char
        for char in json.dumps(text, ensure_ascii=False)
    )


def _code(text: str) -> str:
    """Inline code span that cannot be broken out of by backticks or newlines."""
    flat = text.replace("\r", "\\r").replace("\n", "\\n")
    longest = max((len(run) for run in re.findall(r"`+", flat)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if flat.startswith("`") or flat.endswith("`") or not flat.strip() else ""
    return f"{fence}{pad}{flat}{pad}{fence}"


def _fenced(text: str, language: str) -> list[str]:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{language}", text, fence]
