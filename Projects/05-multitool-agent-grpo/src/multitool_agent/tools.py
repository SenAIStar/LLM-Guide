from __future__ import annotations

from dataclasses import dataclass, field
from operator import add, mul, sub, truediv
from typing import Any, Callable


@dataclass
class MockState:
    calendar: dict[str, str] = field(default_factory=dict)
    drafts: list[dict[str, str]] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    web_index: dict[str, list[dict[str, str]]] = field(default_factory=dict)


def calculator_compute(left: float, operator: str, right: float) -> dict[str, Any]:
    operations = {"add": add, "subtract": sub, "multiply": mul, "divide": truediv}
    if operator not in operations:
        return {"ok": False, "error_code": "UNSUPPORTED_OPERATOR"}
    if operator == "divide" and right == 0:
        return {"ok": False, "error_code": "DIVISION_BY_ZERO"}
    return {"ok": True, "data": {"result": operations[operator](left, right)}}


def file_read(state: MockState, path: str) -> dict[str, Any]:
    if path not in state.files:
        return {"ok": False, "error_code": "FILE_NOT_FOUND"}
    return {"ok": True, "data": {"path": path, "content": state.files[path]}}


def calendar_query(state: MockState, start: str) -> dict[str, Any]:
    status = state.calendar.get(start, "unknown")
    return {"ok": True, "data": {"start": start, "status": status}}


def create_mail_draft(
    state: MockState, to: str, subject: str, body: str
) -> dict[str, Any]:
    draft = {"to": to, "subject": subject, "body": body}
    state.drafts.append(draft)
    return {"ok": True, "data": {"draft_id": len(state.drafts) - 1}}


def create_todo(state: MockState, title: str) -> dict[str, Any]:
    if not title.strip():
        return {"ok": False, "error_code": "EMPTY_TITLE"}
    state.todos.append(title.strip())
    return {"ok": True, "data": {"todo_id": len(state.todos) - 1}}


def web_search(state: MockState, query: str, top_k: int = 5) -> dict[str, Any]:
    if top_k <= 0:
        return {"ok": False, "error_code": "INVALID_TOP_K"}
    return {"ok": True, "data": {"results": state.web_index.get(query, [])[:top_k]}}


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "calculator.compute": calculator_compute,
    "file.read": file_read,
    "calendar.query": calendar_query,
    "mail.create_draft": create_mail_draft,
    "todo.create": create_todo,
    "web.search": web_search,
}

TOOL_ARGUMENTS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "calculator.compute": {"left": (int, float), "operator": str, "right": (int, float)},
    "file.read": {"path": str},
    "calendar.query": {"start": str},
    "mail.create_draft": {"to": str, "subject": str, "body": str},
    "todo.create": {"title": str},
    "web.search": {"query": str},
}


def _arguments_match_schema(name: str, arguments: dict[str, Any]) -> bool:
    schema = TOOL_ARGUMENTS[name]
    allowed_fields = set(schema)
    if name == "web.search":
        allowed_fields.add("top_k")
    if not set(schema).issubset(arguments) or not set(arguments).issubset(allowed_fields):
        return False
    for field, expected_type in schema.items():
        value = arguments[field]
        if isinstance(value, bool) and expected_type == (int, float):
            return False
        if not isinstance(value, expected_type):
            return False
    return not (
        "top_k" in arguments
        and (
            isinstance(arguments["top_k"], bool)
            or not isinstance(arguments["top_k"], int)
        )
    )


def execute_action(
    state: MockState, action: dict[str, Any], allowed_tools: set[str]
) -> dict[str, Any]:
    name = action.get("tool_name")
    arguments = action.get("arguments")
    if name not in allowed_tools or name not in TOOLS:
        return {"ok": False, "error_code": "TOOL_NOT_ALLOWED"}
    if not isinstance(arguments, dict):
        return {"ok": False, "error_code": "INVALID_ARGUMENTS"}
    if not _arguments_match_schema(name, arguments):
        return {"ok": False, "error_code": "SCHEMA_ERROR"}
    try:
        if name == "calculator.compute":
            return TOOLS[name](**arguments)
        return TOOLS[name](state, **arguments)
    except TypeError as exc:
        return {"ok": False, "error_code": "SCHEMA_ERROR", "detail": str(exc)}
