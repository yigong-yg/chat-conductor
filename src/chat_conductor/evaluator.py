from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import sqlite3

from .store import search


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_turn_ids: tuple[str, ...]
    k: int | None = None


@dataclass(frozen=True)
class EvalCaseResult:
    query: str
    k: int
    hit: bool
    expected_turn_ids: tuple[str, ...]
    returned_turn_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvalReport:
    cases: tuple[EvalCaseResult, ...]

    @property
    def recall_at_k(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for case in self.cases if case.hit) / len(self.cases)

    @property
    def misses(self) -> tuple[EvalCaseResult, ...]:
        return tuple(case for case in self.cases if not case.hit)


def load_eval_cases(path: Path) -> list[EvalCase]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        raw_cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw_cases = json.loads(text)
    if not isinstance(raw_cases, list):
        raise ValueError("eval file must contain a list of cases")
    return [_parse_case(raw) for raw in raw_cases]


def run_eval(connection: sqlite3.Connection, cases: list[EvalCase], *, default_k: int = 10) -> EvalReport:
    results: list[EvalCaseResult] = []
    for case in cases:
        k = case.k or default_k
        returned = tuple(result.turn_id for result in search(connection, case.query, limit=k))
        expected = set(case.expected_turn_ids)
        results.append(
            EvalCaseResult(
                query=case.query,
                k=k,
                hit=bool(expected.intersection(returned)),
                expected_turn_ids=case.expected_turn_ids,
                returned_turn_ids=returned,
            )
        )
    return EvalReport(cases=tuple(results))


def report_to_dict(report: EvalReport) -> dict:
    return {
        "cases": len(report.cases),
        "hits": sum(1 for case in report.cases if case.hit),
        "recall_at_k": report.recall_at_k,
        "misses": [
            {
                "query": case.query,
                "k": case.k,
                "expected_turn_ids": list(case.expected_turn_ids),
                "returned_turn_ids": list(case.returned_turn_ids),
            }
            for case in report.misses
        ],
    }


def _parse_case(raw: object) -> EvalCase:
    if not isinstance(raw, dict):
        raise ValueError("each eval case must be an object")
    query = raw.get("query")
    expected = raw.get("expected_turn_ids", raw.get("expected_turn_id"))
    if isinstance(expected, str):
        expected_turn_ids = (expected,)
    elif isinstance(expected, list) and all(isinstance(item, str) for item in expected):
        expected_turn_ids = tuple(expected)
    else:
        raise ValueError(f"eval case {query!r} needs expected_turn_id(s)")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("eval case needs a non-empty query")
    k = raw.get("k")
    if k is not None and (not isinstance(k, int) or k <= 0):
        raise ValueError(f"eval case {query!r} has invalid k")
    return EvalCase(query=query, expected_turn_ids=expected_turn_ids, k=k)
