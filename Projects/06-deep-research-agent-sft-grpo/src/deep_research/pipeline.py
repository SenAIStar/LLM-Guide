from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol


@dataclass(frozen=True)
class EvidenceCard:
    card_id: str
    url: str
    title: str
    published_at: str | None
    fetched_at: str
    page_hash: str
    claim: str
    evidence_text: str
    source_type: str
    conflict_group: str | None = None


@dataclass
class ResearchTrace:
    question: str
    subquestions: list[str]
    actions: list[dict] = field(default_factory=list)
    evidence_cards: list[EvidenceCard] = field(default_factory=list)
    report: str = ""
    stop_reason: str = ""


class SearchSnapshot(Protocol):
    def search(self, query: str, top_k: int) -> list[dict]: ...
    def fetch(self, url: str) -> dict: ...


def duplicate_query_ratio(queries: Iterable[str]) -> float:
    normalized = [" ".join(query.lower().split()) for query in queries]
    if not normalized:
        return 0.0
    return 1.0 - len(set(normalized)) / len(normalized)


def run_research(
    question: str,
    planner,
    policy,
    snapshot: SearchSnapshot,
    *,
    max_steps: int = 20,
) -> ResearchTrace:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    subquestions = planner(question)
    if not isinstance(subquestions, list) or not all(
        isinstance(item, str) and item.strip() for item in subquestions
    ):
        raise ValueError("planner must return a list of non-empty subquestions")
    trace = ResearchTrace(question=question, subquestions=subquestions)
    for step in range(max_steps):
        action = policy(trace)
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            trace.actions.append(
                {
                    "step": step,
                    "action": action,
                    "observation": {"ok": False, "error": "invalid_action"},
                }
            )
            trace.stop_reason = "invalid_action"
            break
        if action["type"] == "finish":
            trace.report = str(action.get("report", ""))
            trace.stop_reason = "finished"
            break
        if action["type"] == "search":
            query = str(action.get("query", "")).strip()
            observation = (
                snapshot.search(query, top_k=5)
                if query
                else {"ok": False, "error": "empty_query"}
            )
        elif action["type"] == "fetch":
            url = str(action.get("url", "")).strip()
            observation = (
                snapshot.fetch(url) if url else {"ok": False, "error": "empty_url"}
            )
        elif action["type"] == "save_evidence":
            try:
                card = EvidenceCard(**action["card"])
            except (KeyError, TypeError):
                observation = {"ok": False, "error": "invalid_evidence_card"}
            else:
                if any(existing.card_id == card.card_id for existing in trace.evidence_cards):
                    observation = {"ok": False, "error": "duplicate_evidence_card"}
                else:
                    trace.evidence_cards.append(card)
                    observation = {"ok": True}
        else:
            observation = {"ok": False, "error": "unknown_action"}
        trace.actions.append({"step": step, "action": action, "observation": observation})
    else:
        trace.stop_reason = "step_budget_exhausted"
    return trace


def audit_citations(claim_to_card_ids: dict[str, list[str]], cards: list[EvidenceCard]) -> dict:
    available = {card.card_id for card in cards}
    cited = {card_id for ids in claim_to_card_ids.values() for card_id in ids}
    return {
        "invalid_card_ids": sorted(cited - available),
        "claims_without_citations": sorted(
            claim for claim, ids in claim_to_card_ids.items() if not ids
        ),
    }
