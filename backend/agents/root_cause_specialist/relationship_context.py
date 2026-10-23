import json
import re
from typing import Any

from ...tools import available_functions
from ..scor_agent_tools import build_focused_source_record
from .scor_path_retrieval import retrieve_scor_mapping_paths


PROCESS_ENTITY_REQUEST_PATTERN = re.compile(
    r"\b(?:process(?:es)?|process elements?|scor (?:activit(?:y|ies)|operations?|elements?)|"
    r"root[- ]cause|underlying cause|failure point|breakdown|bottleneck|kök neden|"
    r"temel neden|darboğaz|aksama)\b",
    re.IGNORECASE,
)
ENTITY_SEARCH_LIMIT = 8
PATH_CONTEXT_LIMIT = 8
FOCUSED_RELATION_LIMIT = 5
DOMINANT_ANCHOR_MIN_SCORE = 60
DOMINANT_ANCHOR_RATIO = 2.0


def infer_requested_entity_types(user_message: str) -> list[str]:
    """Identify entity types needed for a diagnostic or relationship mapping."""
    lowered = user_message.lower()
    entity_types: list[str] = []
    if PROCESS_ENTITY_REQUEST_PATTERN.search(user_message):
        entity_types.append("process")
    if re.search(r"\b(?:best\s+)?practices?\b", lowered):
        entity_types.append("practice")
    if re.search(r"\bmetrics?\b", lowered):
        entity_types.append("metric")
    if re.search(r"\b(?:people\s+)?skills?\b", lowered):
        entity_types.append("skill")
    return entity_types


def resolve_relationship_search(user_message: str) -> dict[str, Any] | None:
    """Build source-backed candidates and relationship paths for specialist analysis."""
    requested_types = infer_requested_entity_types(user_message)
    if not requested_types:
        return None
    return retrieve_scor_mapping_paths(
        user_message,
        requested_types,
        available_functions["get_scor_entity_details"],
        search_limit=ENTITY_SEARCH_LIMIT,
    )


def _format_entity_heading(result_item: dict[str, Any]) -> str:
    code = result_item.get("normalized_code") or result_item.get("details", {}).get(
        "code", ""
    )
    name = result_item.get("details", {}).get("name") or result_item.get(
        "summary", {}
    ).get("name", "")
    if code and name:
        return f"{code} - {name}"
    return code or name or "the matched SCOR entity"


def _select_context_paths(result: dict[str, Any]) -> list[dict[str, Any]]:
    ranked_paths = [
        path
        for path in result.get("paths", [])
        if isinstance(path, dict) and path.get("entities")
    ]
    if not ranked_paths:
        return []

    unique_paths: list[dict[str, Any]] = []
    path_signatures: set[tuple[Any, ...]] = set()
    for path in ranked_paths:
        entities = path.get("entities", {})
        relationships = path.get("verified_relationships", [])
        signature = (
            tuple(
                sorted(
                    (
                        entity_type,
                        entity.get("code", ""),
                    )
                    for entity_type, entity in entities.items()
                )
            ),
            tuple(
                sorted(
                    (
                        relation.get("from", ""),
                        relation.get("relation", ""),
                        relation.get("to", ""),
                    )
                    for relation in relationships
                )
            ),
        )
        if signature in path_signatures:
            continue
        path_signatures.add(signature)
        unique_paths.append(path)

    requested_types = result.get("requested_types", [])
    anchor_type = "practice" if "practice" in requested_types else requested_types[0]
    candidate_anchor_codes = [
        candidate.get("normalized_code")
        or candidate.get("details", {}).get("code", "")
        for candidate in result.get("candidate_groups", {}).get(anchor_type, [])
    ]
    path_anchor_codes = [
        path.get("entities", {}).get(anchor_type, {}).get("code", "")
        for path in unique_paths
    ]
    available_anchor_codes = [
        code
        for code in dict.fromkeys(candidate_anchor_codes + path_anchor_codes)
        if code and code in path_anchor_codes
    ]
    candidate_positions = {
        code: index for index, code in enumerate(candidate_anchor_codes)
    }
    best_anchor_score_by_code = {
        code: max(
            path.get("anchor_score", path.get("score", 0))
            for path in unique_paths
            if path.get("entities", {}).get(anchor_type, {}).get("code", "")
            == code
        )
        for code in available_anchor_codes
    }
    anchor_codes = sorted(
        available_anchor_codes,
        key=lambda code: (
            -best_anchor_score_by_code[code],
            candidate_positions.get(code, len(candidate_positions)),
            code,
        ),
    )
    if not anchor_codes:
        return unique_paths[:PATH_CONTEXT_LIMIT]

    anchor_limit = min(len(anchor_codes), max(1, PATH_CONTEXT_LIMIT // 2))
    if len(anchor_codes) >= 2:
        leading_score = best_anchor_score_by_code[anchor_codes[0]]
        runner_up_score = best_anchor_score_by_code[anchor_codes[1]]
        if (
            leading_score >= DOMINANT_ANCHOR_MIN_SCORE
            and leading_score
            >= DOMINANT_ANCHOR_RATIO * max(runner_up_score, 1)
        ):
            anchor_limit = 1
    selected_anchor_codes = anchor_codes[:anchor_limit]
    selected_paths: list[dict[str, Any]] = []

    # Reserve context for multiple plausible anchors before adding alternative
    # relationships for any one anchor. Best verified path supplies relevance order.
    for anchor_code in selected_anchor_codes:
        representative = next(
            (
                path
                for path in unique_paths
                if path.get("entities", {})
                .get(anchor_type, {})
                .get("code", "")
                == anchor_code
            ),
            None,
        )
        if representative is not None:
            selected_paths.append(representative)

    # Add the next relevant path with a new related entity for each anchor in
    # round-robin order. This avoids one high-scoring anchor consuming context.
    while len(selected_paths) < PATH_CONTEXT_LIMIT:
        added_in_round = False
        for anchor_code in selected_anchor_codes:
            if len(selected_paths) >= PATH_CONTEXT_LIMIT:
                break
            anchor_selected = [
                path
                for path in selected_paths
                if path.get("entities", {})
                .get(anchor_type, {})
                .get("code", "")
                == anchor_code
            ]
            covered_entities = {
                (entity_type, entity.get("code", ""))
                for path in anchor_selected
                for entity_type, entity in path.get("entities", {}).items()
                if entity_type != anchor_type
            }
            alternatives = [
                path
                for path in unique_paths
                if path not in selected_paths
                and path.get("entities", {})
                .get(anchor_type, {})
                .get("code", "")
                == anchor_code
            ]
            alternative = max(
                alternatives,
                key=lambda path: (
                    sum(
                        1
                        for entity_type, entity in path.get("entities", {}).items()
                        if entity_type != anchor_type
                        and (entity_type, entity.get("code", ""))
                        not in covered_entities
                    ),
                    path.get("score", 0),
                ),
                default=None,
            )
            if alternative is not None and not any(
                (entity_type, entity.get("code", "")) not in covered_entities
                for entity_type, entity in alternative.get("entities", {}).items()
                if entity_type != anchor_type
            ):
                alternative = None
            if alternative is not None:
                selected_paths.append(alternative)
                added_in_round = True
        if not added_in_round:
            break

    if len(selected_paths) < PATH_CONTEXT_LIMIT:
        selected_ids = {id(path) for path in selected_paths}
        selected_paths.extend(
            path for path in unique_paths if id(path) not in selected_ids
        )
    return selected_paths[:PATH_CONTEXT_LIMIT]


def _append_practice_inventory(
    lines: list[str], result: dict[str, Any], paths: list[dict[str, Any]]
) -> None:
    top_practice_code = paths[0].get("entities", {}).get("practice", {}).get(
        "code", ""
    )
    if not top_practice_code:
        return
    top_practice = next(
        (
            candidate
            for candidate in result.get("candidate_groups", {}).get("practice", [])
            if candidate.get("normalized_code") == top_practice_code
        ),
        None,
    )
    if not top_practice:
        return
    if top_practice.get("match_type") not in {"direct_code", "exact_name"}:
        return

    related = top_practice.get("related", {})
    requested_types = set(result.get("requested_types", []))
    inventory: dict[str, Any] = {
        "practice": {
            "code": top_practice_code,
            "name": top_practice.get("details", {}).get("name", ""),
        }
    }
    relation_fields = {
        "process": ("complete_supports_processes", "processes"),
        "metric": ("complete_impacts_metrics", "metrics"),
        "skill": ("complete_required_skills", "people_skills"),
    }
    for entity_type, (output_field, related_field) in relation_fields.items():
        if entity_type in requested_types:
            inventory[output_field] = [
                {"code": item.get("code", ""), "name": item.get("name", "")}
                for item in related.get(related_field, [])
            ]

    lines.append(
        "Complete direct-relationship inventory for the top-ranked practice anchor "
        "(source order is not a relevance ranking):"
    )
    lines.append(json.dumps(inventory, ensure_ascii=False, indent=2, default=str))


def _append_focused_practice_relationships(
    lines: list[str], result: dict[str, Any], paths: list[dict[str, Any]]
) -> None:
    """Expose top direct relations by entity type without requiring one tuple to win."""
    leading_practice_code = paths[0].get("entities", {}).get("practice", {}).get(
        "code", ""
    )
    if not leading_practice_code:
        return

    leading_practice = next(
        (
            candidate
            for candidate in result.get("candidate_groups", {}).get("practice", [])
            if candidate.get("normalized_code") == leading_practice_code
        ),
        None,
    )
    if not leading_practice or leading_practice.get("match_type") in {
        "direct_code",
        "exact_name",
    }:
        return

    requested_types = set(result.get("requested_types", []))
    relation_by_type = {
        "process": "supports_processes",
        "metric": "impacts_metrics",
        "skill": "required_skills",
    }
    focused: dict[str, list[dict[str, Any]]] = {}
    for entity_type, relation_name in relation_by_type.items():
        if entity_type not in requested_types:
            continue
        values: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        for path in result.get("paths", []):
            entities = path.get("entities", {})
            if entities.get("practice", {}).get("code", "") != leading_practice_code:
                continue
            entity = entities.get(entity_type, {})
            code = entity.get("code", "")
            if not code or code in seen_codes:
                continue
            has_direct_relation = any(
                relationship.get("from") == leading_practice_code
                and relationship.get("relation") == relation_name
                and relationship.get("to") == code
                for relationship in path.get("verified_relationships", [])
            )
            if not has_direct_relation:
                continue
            seen_codes.add(code)
            values.append(
                {
                    key: entity[key]
                    for key in (
                        "code",
                        "name",
                        "definition",
                        "category",
                        "attribute",
                    )
                    if key in entity
                }
            )
            if len(values) >= FOCUSED_RELATION_LIMIT:
                break
        if values:
            focused[relation_name] = values

    if not focused:
        return
    lines.append(
        "Focused direct relationships for the leading practice candidate, grouped by "
        "requested entity type (each item is independently source-backed):"
    )
    lines.append(
        json.dumps(
            {
                "practice": {
                    "code": leading_practice_code,
                    "name": leading_practice.get("details", {}).get("name", ""),
                },
                "relationships": focused,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def format_relationship_context(result: dict[str, Any]) -> str:
    """Format specialist-only paths or candidates without asserting applicability."""
    paths = _select_context_paths(result)
    if paths:
        selection_assessment = result.get("selection_assessment", {})
        ambiguous = selection_assessment.get("selection_status") == "ambiguous"
        exact_identity = any(
            path.get("anchor_match_type") in {"direct_code", "exact_name"}
            for path in paths
        )
        lines = [
            "INTERNAL RELATIONSHIP CONTEXT — use the facts for analysis, but never quote, cite, "
            "or describe this context, its record language, field names, ranks, or paths.",
            (
                "Authoritative SCOR selection assessment: ambiguous; no single exact entity is source-supported yet."
                if ambiguous
                else "Authoritative SCOR match with source-verified relationship paths."
                if exact_identity
                else "Candidate SCOR relationship paths from approved local structured data."
            ),
            "Candidate counts before relationship traversal: "
            + json.dumps(result.get("candidate_count_by_type", {}), ensure_ascii=False),
        ]
        if ambiguous:
            lines.extend(
                [
                    "Mandatory ambiguity gate (do not override merely because the user requests one exact code):",
                    json.dumps(
                        selection_assessment,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    "Do not select or present one metric or responsible process as certain. "
                    "Explain the supported alternatives and ask specifically for every item in "
                    "`missing_evidence`. A numeric observation is a measured value, not proof "
                    "of metric identity.",
                ]
            )

        _append_practice_inventory(lines, result, paths)
        _append_focused_practice_relationships(lines, result, paths)
        for index, path in enumerate(paths, start=1):
            lines.append(
                f"- Verified path {index} (relevance score={path.get('score', 0)}):"
            )
            lines.append(
                json.dumps(
                    {
                        "entities": path.get("entities", {}),
                        "direct_relationships": path.get(
                            "verified_relationships", []
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
        lines.append(
            "Every edge above was read from an approved structured SCOR relationship field. "
            "Rank is relevance guidance, not proof that the first path is uniquely correct. "
            "Compare definitions with the operational evidence, prefer the most specific "
            "supported process, and state only the displayed direct relationship types."
        )
        return "\n".join(lines)

    candidates = [
        item
        for item in result.get("results", [])[:ENTITY_SEARCH_LIMIT]
        if isinstance(item, dict) and item.get("found")
    ]
    if not candidates:
        return ""

    exact_identity = (
        len(candidates) == 1
        and candidates[0].get("match_type") in {"direct_code", "exact_name"}
    )
    lines = [
        "INTERNAL CANDIDATE CONTEXT — use the facts for analysis, but never quote, cite, or "
        "describe this context, its record language, field names, ranks, or paths.",
        (
            "Authoritative SCOR match: exact entity identity from approved local structured data."
            if exact_identity
            else "Candidate SCOR records from approved local structured data."
        )
    ]
    for index, item in enumerate(candidates, start=1):
        score = item.get("match_score")
        score_suffix = f", score={score}" if isinstance(score, int) else ""
        lines.append(
            f"- Candidate {index}: {_format_entity_heading(item)} "
            f"({item.get('match_type', 'text_search')}{score_suffix})"
        )
        lines.append(
            json.dumps(
                build_focused_source_record(item),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    lines.append(
        "Retrieval rank is not proof of scenario applicability. Compare every candidate's "
        "definition with all material facts and preserve ambiguity when the evidence does not "
        "distinguish the candidates."
    )
    return "\n".join(lines)


def build_relationship_context(user_message: str) -> str:
    result = resolve_relationship_search(user_message)
    return format_relationship_context(result) if result else ""
