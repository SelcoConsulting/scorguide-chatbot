import json
import re
from typing import Any

from .supply_chain_specialist.supply_chain_specialist import ask_supply_chain_specialist
from ..tools import available_functions, tools as scor_tools
from ..utils.logging import log_with_prefix, summarize_arguments, summarize_log_value


MAX_TOOL_ITEMS = 3
MAX_TEXT_LENGTH = 600
COMPLETE_LIST_TOOLS = {
    "get_process_details",
    "get_practice_details",
    "get_metric_details",
    "get_skill_details",
}
LEVEL_THREE_PROCESS_CODE_PATTERN = re.compile(r"^[A-Z]{1,3}\d+\.\d+$", re.IGNORECASE)
LEVEL_THREE_DESCENDANT_REQUEST_PATTERN = re.compile(
    r"\blevel\s*[- ]?\s*3\b|\bchild\s+process(?:es)?\b|"
    r"\bsub[- ]?process(?:es)?\b|\bprocess(?:es)?\s+(?:under|within)\b",
    re.IGNORECASE,
)

GENERAL_LLM_TOOL = {
    "type": "function",
    "name": "ask_supply_chain_specialist",
    "description": (
        "Use this only for broad supply-chain concepts, methods, implementation guidance, or "
        "document-backed questions that are not already answered by the embedded SCOR-DS "
        "reference and are not primarily asking for an exact SCOR entity or a root-cause "
        "diagnosis. It searches the private supply-chain info packs before answering."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The user's complete broad supply-chain question.",
            },
            "user_context": {
                "type": "string",
                "description": "Optional recent chat context that may help answer the question.",
            },
        },
        "required": ["question", "user_context"],
        "additionalProperties": False,
    },
}


BASE_SCOR_AGENT_TOOLS = [
    tool for tool in scor_tools if tool.get("name") != "search_supply_chain_knowledge"
] + [GENERAL_LLM_TOOL]


def _slim_related_entities(values: Any) -> list[Any]:
    if not isinstance(values, list):
        return []
    slimmed: list[Any] = []
    allowed_keys = {
        "code",
        "name",
        "definition",
        "category",
        "attribute",
        "categories",
        "pillars",
        "level",
    }
    for value in values:
        if isinstance(value, dict):
            slimmed.append({key: value[key] for key in allowed_keys if key in value})
        else:
            slimmed.append(value)
    return slimmed


def build_focused_source_record(item: dict[str, Any]) -> dict[str, Any]:
    """Keep exact facts visible without burying them under unrelated nested relationships."""
    entity_type = str(item.get("entity_type", ""))
    details = dict(item.get("details", {}))
    full_definition = str(item.get("summary", {}).get("definition", "")).strip()
    if full_definition:
        details["definition"] = full_definition
    hierarchy = item.get("hierarchy", {})
    related = item.get("related", {})

    if entity_type == "practices":
        return {
            "details": details,
            "direct_relationships": {
                "supported_processes": _slim_related_entities(related.get("processes", [])),
                "impacted_metrics": _slim_related_entities(related.get("metrics", [])),
                "required_people_skills": _slim_related_entities(
                    related.get("people_skills", [])
                ),
            },
        }
    if entity_type == "skills":
        return {
            "details": details,
            "direct_relationships": {
                "applicable_processes": _slim_related_entities(
                    related.get("processes", []),
                ),
                "supported_best_practices": _slim_related_entities(
                    related.get("best_practices", [])
                ),
                "experiences": related.get("experiences", []),
                "trainings": related.get("trainings", []),
            },
        }
    if entity_type == "metrics":
        improvement = item.get("improvement_options", {})
        return {
            "details": details,
            "hierarchy": hierarchy,
            "direct_relationships": {
                "measured_processes": details.get("measured_processes", []),
                "improved_by_best_practices": details.get(
                    "improved_by_best_practices", []
                ),
            },
            "improvement_relationships": {
                "related_best_practices": _slim_related_entities(
                    improvement.get("related_best_practices", [])
                ),
                "related_processes_across_metric_scope": _slim_related_entities(
                    improvement.get("related_processes", [])
                ),
                "related_people_skills": _slim_related_entities(
                    improvement.get("related_people_skills", [])
                ),
            },
        }
    if entity_type == "processes":
        return {
            "details": details,
            "hierarchy": hierarchy,
            "direct_relationships": {
                "best_practices": _slim_related_entities(related.get("best_practices", [])),
                "metrics": _slim_related_entities(related.get("metrics", [])),
                "people_skills": _slim_related_entities(related.get("people_skills", [])),
                "inputs": related.get("inputs", []),
                "outputs": related.get("outputs", []),
            },
        }
    return {"details": details, "hierarchy": hierarchy}


def _requested_process_relationship_fields(user_message: str) -> list[str]:
    """Return requested direct process fields without tying the lookup to any SCOR code."""
    lowered = user_message.lower()
    requested: list[str] = []
    if re.search(r"\bmetrics?\b|\bmeasure(?:s|d|ment|ments)?\b.*\bperformance\b", lowered):
        requested.append("performance_metrics")
    if re.search(r"\b(?:best\s+)?practices?\b", lowered):
        requested.append("supporting_best_practices")
    if re.search(r"\b(?:people\s+)?skills?\b", lowered):
        requested.append("associated_skills")
    if re.search(r"\binputs?\b", lowered):
        requested.append("input_processes")
    if re.search(r"\boutputs?\b", lowered):
        requested.append("output_processes")
    return requested


def _resolve_entity_identities(codes: list[str]) -> dict[str, dict[str, str]]:
    if not codes:
        return {}
    response = available_functions["get_scor_entity_details"](
        entity_codes=codes,
        query_text="",
        entity_types=[],
        limit=max(3, len(codes)),
    )
    identities: dict[str, dict[str, str]] = {}
    for item in response.get("results", []):
        if not item.get("found"):
            continue
        code = str(item.get("normalized_code", ""))
        summary = item.get("summary") or {}
        identities[code.upper()] = {
            "code": code,
            "name": str(summary.get("name", "")),
        }
    return identities


def _build_level_three_descendant_context(
    user_message: str,
    detected: dict[str, Any],
) -> str:
    """Expand requested fields from Level-3 descendants using only structured hierarchy data."""
    if not LEVEL_THREE_DESCENDANT_REQUEST_PATTERN.search(user_message):
        return ""

    requested_fields = _requested_process_relationship_fields(user_message)
    normalized_field_map = {
        "performance_metrics": "related_metrics",
        "supporting_best_practices": "related_best_practices",
        "associated_skills": "related_people_skills",
        "input_processes": "process_inputs",
        "output_processes": "process_outputs",
    }
    blocks: list[dict[str, Any]] = []

    for match in detected.get("matches", []):
        if match.get("entity_type") != "processes":
            continue
        parent_code = str(match.get("normalized_code", ""))
        if not parent_code or LEVEL_THREE_PROCESS_CODE_PATTERN.fullmatch(parent_code):
            continue

        descendants = [
            process
            for process in available_functions["get_process_details"]([parent_code])
            if LEVEL_THREE_PROCESS_CODE_PATTERN.fullmatch(str(process.get("code", "")))
        ]
        if not descendants:
            continue

        related_codes: list[str] = []
        seen_related_codes: set[str] = set()
        for process in descendants:
            for requested_field in requested_fields:
                source_field = normalized_field_map[requested_field]
                for code in process.get(source_field, []):
                    normalized = str(code).upper()
                    if normalized and normalized not in seen_related_codes:
                        seen_related_codes.add(normalized)
                        related_codes.append(str(code))
        identities = _resolve_entity_identities(related_codes)

        child_records: list[dict[str, Any]] = []
        aggregate: dict[str, list[dict[str, str]]] = {
            field: [] for field in requested_fields
        }
        aggregate_seen: dict[str, set[str]] = {
            field: set() for field in requested_fields
        }
        for process in descendants:
            child_record: dict[str, Any] = {
                "process_element_code": process.get("code", ""),
                "process_element_name": process.get("name", ""),
            }
            for requested_field in requested_fields:
                source_field = normalized_field_map[requested_field]
                values: list[dict[str, str]] = []
                for code_value in process.get(source_field, []):
                    code = str(code_value)
                    identity = identities.get(code.upper(), {"code": code, "name": ""})
                    values.append(identity)
                    if code.upper() not in aggregate_seen[requested_field]:
                        aggregate_seen[requested_field].add(code.upper())
                        aggregate[requested_field].append(identity)
                child_record[requested_field] = values
            child_records.append(child_record)

        blocks.append(
            {
                "parent_process_code": parent_code,
                "parent_process_name": str((match.get("summary") or {}).get("name", "")),
                "level_3_processes": child_records,
                "distinct_requested_relationships": aggregate,
            }
        )

    if not blocks:
        return ""
    return (
        "INTERNAL LEVEL-3 RELATIONSHIP CONTEXT — use the facts, but never quote, cite, or "
        "describe this context, its field names, or its storage format. "
        "Requested Level-3 descendant process records from approved local structured data. "
        "The descendants and every listed relationship below were derived dynamically from "
        "the stored process hierarchy and each descendant's own direct fields:\n"
        + json.dumps(blocks, ensure_ascii=False, indent=2, default=str)
        + "\nUse the Level-3 records above for the requested scope; do not substitute the "
        "parent process record's relationship list."
    )


def build_detected_code_source_context(
    user_message: str,
    detected_codes: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Resolve codes in a message to exact local records before an LLM answers."""
    detected = detected_codes or available_functions["identify_scor_codes_in_text"](user_message)
    if not detected.get("found"):
        return detected, ""

    normalized_codes = [
        str(match.get("normalized_code", ""))
        for match in detected.get("matches", [])
        if match.get("normalized_code")
    ]
    details_response = available_functions["get_scor_entity_details"](
        entity_codes=normalized_codes,
        query_text=user_message,
        entity_types=[],
        limit=max(3, len(normalized_codes)),
    )
    details_by_code = {
        str(item.get("normalized_code", "")).upper(): item
        for item in details_response.get("results", [])
        if item.get("found")
    }

    lines = [
        "INTERNAL ENTITY CONTEXT — use the facts, but never quote, cite, or describe this "
        "context, its field names, or its storage format.",
        "Detected SCOR code context: exact records from approved local structured data:",
        "Use the normalized identity and the stored fields below as source-of-truth.",
    ]
    for match_index, match in enumerate(detected.get("matches", [])):
        normalized_code = str(match.get("normalized_code", ""))
        summary = match.get("summary") or {}
        name = str(summary.get("name", ""))
        identity = (
            f"- input fragment `{match.get('input_fragment', '')}` -> "
            f"`{normalized_code}` ({match.get('entity_type_label', '')})"
        )
        if name:
            identity += f": {name}"
        lines.append(identity)

        item = details_by_code.get(normalized_code.upper())
        if not item:
            continue
        source_record = build_focused_source_record(item)
        lines.append(
            "  Approved structured record:\n"
            + json.dumps(source_record, ensure_ascii=False, indent=2, default=str)
        )

    descendant_context = _build_level_three_descendant_context(user_message, detected)
    if descendant_context:
        lines.append(descendant_context)

    lines.append(
        "Relationship discipline: report a direct link only when it appears in a stored "
        "relationship field. A hierarchy parent may be explained as hierarchy context, but it "
        "must not be added to a list of directly linked entities unless the record lists it. "
        "For a metric's measured processes, use the metric details `measured_processes` field; "
        "do not substitute processes aggregated from child metrics or a linked practice. Do not "
        "claim a field is unavailable when it is present above. Present the resulting facts "
        "directly to the user without mentioning records, fields, inventories, or paths."
    )
    return detected, "\n".join(lines)


def summarize_base_tool_result(tool_name: str, tool_result: Any) -> dict[str, Any]:
    if tool_name == "get_scor_entity_details" and isinstance(tool_result, dict):
        summarized_results = []
        for item in tool_result.get("results", []):
            summarized_results.append(
                {
                    "found": item.get("found", False),
                    "input_code": item.get("input_code", ""),
                    "normalized_code": item.get("normalized_code", ""),
                    "entity_type_label": item.get("entity_type_label", ""),
                    "match_type": item.get("match_type", ""),
                    "match_score": item.get("match_score"),
                    "matched_name": item.get("matched_name", ""),
                    "summary": item.get("summary", {}),
                    "details": item.get("details", {}),
                    "hierarchy": item.get("hierarchy", {}),
                    "related": item.get("related", {}),
                    "improvement_options": item.get("improvement_options", {}),
                }
            )
        return {
            "tool_name": tool_name,
            "requested_entity_codes": tool_result.get("requested_entity_codes", []),
            "query_text": tool_result.get("query_text", ""),
            "requested_entity_types": tool_result.get("requested_entity_types", []),
            "result_count": tool_result.get("result_count", 0),
            "results": summarized_results,
            "dictionary_match_count": tool_result.get("dictionary_match_count", 0),
            "dictionary_matches": tool_result.get("dictionary_matches", [])[:MAX_TOOL_ITEMS],
            "truncated": tool_result.get("result_count", 0) > len(summarized_results),
            "guidance": (
                "Use the embedded SCOR-DS reference before this result for hierarchy. Tool "
                "hierarchy only supplements details absent from the embedded reference. "
                "Interpret match_type as follows: direct_code and exact_name establish entity "
                "identity; fuzzy_name and text_search are candidates only. A candidate match, "
                "including the top-scoring candidate, does not by itself prove that the entity "
                "applies to the user's scenario and must not be presented as an exact mapping. "
                "If truncated is true, do not imply that the returned list is complete."
            ),
        }

    if tool_name == "ask_supply_chain_specialist" and isinstance(tool_result, dict):
        answer = str(tool_result.get("answer", ""))
        return {
            "tool_name": tool_name,
            "preview": {
                "answer": answer,
                "model": tool_result.get("model", ""),
                "used_supply_chain_knowledge": tool_result.get(
                    "used_supply_chain_knowledge", False
                ),
                "knowledge_result_count": tool_result.get("knowledge_result_count", 0),
            },
            "truncated": False,
            "guidance": (
                "Use this source-grounded specialist answer only when it is relevant. The "
                "embedded SCOR-DS reference remains authoritative for hierarchy and for facts it "
                "states explicitly."
            ),
        }

    if tool_name == "search_supply_chain_knowledge" and isinstance(tool_result, dict):
        results = []
        for item in tool_result.get("results", [])[:MAX_TOOL_ITEMS]:
            results.append(
                {
                    "id": item.get("id"),
                    "score": item.get("score"),
                    "title": item.get("title", ""),
                    "text": str(item.get("text", ""))[:MAX_TEXT_LENGTH],
                    "source": item.get("source", ""),
                    "metadata": item.get("metadata", {}),
                }
            )
        return {
            "tool_name": tool_name,
            "query": tool_result.get("query", ""),
            "result_count": tool_result.get("result_count", 0),
            "results": results,
            "truncated": tool_result.get("result_count", 0) > len(results),
            "guidance": "Base the answer on these passages and cite title/source when available.",
        }

    if isinstance(tool_result, list):
        # Relationship lookups such as "practices for metric X" are exact list
        # requests. Truncating these to the first three records makes a correct
        # tool result incomplete before it reaches the model.
        returned_items = (
            tool_result
            if tool_name in COMPLETE_LIST_TOOLS
            else tool_result[:MAX_TOOL_ITEMS]
        )
        return {
            "tool_name": tool_name,
            "item_count": len(tool_result),
            "preview": returned_items,
            "truncated": len(tool_result) > len(returned_items),
        }

    if isinstance(tool_result, dict):
        return {"tool_name": tool_name, "preview": tool_result, "truncated": False}

    text = str(tool_result)
    return {
        "tool_name": tool_name,
        "preview": text[:MAX_TEXT_LENGTH],
        "truncated": len(text) > MAX_TEXT_LENGTH,
    }


def run_base_tool(tool_name: str, arguments_json: str, *, log_prefix: str) -> Any:
    arguments = json.loads(arguments_json)
    log_with_prefix(
        log_prefix,
        f"Tool call started: {tool_name} | args: {summarize_arguments(arguments)}",
    )
    if tool_name == "ask_supply_chain_specialist":
        result = ask_supply_chain_specialist(**arguments)
    else:
        tool_fn = available_functions.get(tool_name)
        if not tool_fn:
            raise RuntimeError(f"Bilinmeyen tool cagrisi: {tool_name}")
        result = tool_fn(**arguments)
    log_with_prefix(
        log_prefix,
        f"Tool call finished: {tool_name} | result: {summarize_log_value(result)}",
    )
    return result
