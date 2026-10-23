import json
import re
from typing import Any


def log_with_prefix(prefix: str, message: str) -> None:
    """Write a prefixed log line for local debugging."""
    print(f"[{prefix}] {message}")


def truncate_text(text: str, limit: int = 180) -> str:
    """Collapse whitespace and shorten long text for compact logs."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def summarize_log_value(value: Any) -> str:
    """Convert nested values into short, human-readable log previews."""
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            preview = []
            for item in value[:3]:
                code = item.get("code") or item.get("normalized_code") or item.get("input_code")
                name = item.get("name") or item.get("entity_type_label")
                if code and name:
                    preview.append(f"{code}:{name}")
                elif code:
                    preview.append(str(code))
                else:
                    preview.append(f"dict_keys={list(item.keys())[:4]}")
            return f"list(len={len(value)}, preview={preview})"
        preview = [truncate_text(str(item), 60) for item in value[:3]]
        return f"list(len={len(value)}, preview={preview})"
    if isinstance(value, dict):
        preferred = {}
        for key in [
            "question",
            "query_text",
            "entity_codes",
            "entity_types",
            "normalized_code",
            "match_type",
            "result_count",
            "found",
            "model",
        ]:
            if key in value:
                preferred[key] = value[key]
        if preferred:
            return json.dumps(preferred, ensure_ascii=False, default=str)
        return f"dict(keys={list(value.keys())[:6]})"
    return repr(truncate_text(str(value), 140))


def summarize_arguments(arguments: dict[str, Any]) -> str:
    """Format tool arguments into a single-line log summary."""
    if not arguments:
        return "{}"
    return ", ".join(
        f"{key}={summarize_log_value(value)}" for key, value in arguments.items()
    )


def log_conversation_start(
    conversation_id: int,
    user_message: str,
    user_context: str,
    model: str,
    *,
    prefix: str = "orchestrator",
) -> None:
    """Print the start of a new conversation with the key request metadata."""
    log_with_prefix(prefix, "=" * 72)
    log_with_prefix(prefix, f"Conversation #{conversation_id} started | model={model}")
    log_with_prefix(prefix, f"User message: {truncate_text(user_message)}")
    if user_context.strip():
        log_with_prefix(prefix, f"User context: {truncate_text(user_context)}")
    log_with_prefix(prefix, "-" * 72)


def tool_log(tool_name: str, message: str) -> None:
    """Write a prefixed tool log line for local debugging."""
    log_with_prefix(f"scor_tool:{tool_name}", message)


def log_tool_start(tool_name: str, **kwargs: Any) -> None:
    """Log a tool invocation together with a compact argument preview."""
    if not kwargs:
        tool_log(tool_name, "called without arguments")
        return
    argument_summary = ", ".join(
        f"{key}={summarize_log_value(value)}" for key, value in kwargs.items()
    )
    tool_log(tool_name, f"called with {argument_summary}")


def log_tool_end(tool_name: str, result: Any) -> None:
    """Log the returned value of a tool in summarized form."""
    tool_log(tool_name, f"returning {summarize_log_value(result)}")
