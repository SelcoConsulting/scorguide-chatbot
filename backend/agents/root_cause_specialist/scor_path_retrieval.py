import difflib
import re
from itertools import product
from typing import Any, Callable

from ...tools.dictionary import _normalize_search_token, _score_text_match
from ...utils.shared import _tokenize_text


SearchFunction = Callable[..., dict[str, Any]]

ENTITY_TYPE_KEYS = {
    "process": "processes",
    "practice": "practices",
    "metric": "metrics",
    "skill": "skills",
}
LEVEL_THREE_PROCESS_PATTERN = re.compile(r"^[A-Z]{1,3}\d+\.\d+$", re.IGNORECASE)
HIGH_LEVEL_PROCESS_REQUEST_PATTERN = re.compile(
    r"\b(?:high[- ]level|parent|process family|level[- ]?[012]|strategic overview)\b",
    re.IGNORECASE,
)
PRACTICE_METRIC_RELATION_PATTERN = re.compile(
    r"\b(?:impacted|affected|improved)\s+(?:performance\s+)?metrics?\b|"
    r"\bmetrics?\s+(?:impacted|affected|improved)\b",
    re.IGNORECASE,
)
PROCESS_METRIC_RELATION_PATTERN = re.compile(
    r"\b(?:process[- ](?:applicable|specific)|measured[- ]process|process performance)\s+metrics?\b|"
    r"\bmetrics?\s+(?:measured by|for|of)\s+(?:the\s+)?process\b",
    re.IGNORECASE,
)
PRACTICE_SKILL_RELATION_PATTERN = re.compile(
    r"\b(?:directly\s+)?required\s+(?:people\s+)?skills?\b|"
    r"\bpractice[- ]required\s+(?:people\s+)?skills?\b",
    re.IGNORECASE,
)
PROCESS_SKILL_RELATION_PATTERN = re.compile(
    r"\bprocess[- ](?:applicable|specific|associated)\s+"
    r"(?:[a-z][a-z-]*\s+){0,3}(?:people\s+)?skills?\b|"
    r"\b(?:people\s+)?skills?\s+applicable\s+to\s+(?:the\s+)?(?:relevant\s+)?"
    r"(?:process|planning scope)\b",
    re.IGNORECASE,
)
ENTITY_SELECTION_REQUEST_PATTERN = re.compile(
    r"\b(?:give|identify|select|choose|map|which|exact|responsible|corresponding|fit)\b",
    re.IGNORECASE,
)
ENTITY_SCOPE_PATTERNS = {
    "process": re.compile(r"\bprocess(?:es)?\b|\bprocess elements?\b", re.IGNORECASE),
    "practice": re.compile(r"\b(?:best\s+)?practices?\b", re.IGNORECASE),
    "metric": re.compile(r"\bmetrics?\b|\bmeasures?\b", re.IGNORECASE),
    "skill": re.compile(r"\b(?:people\s+)?skills?\b", re.IGNORECASE),
}
ENTITY_SCOPE_GENERIC_TOKENS = {
    "a",
    "an",
    "and",
    "approved",
    "choose",
    "corresponding",
    "direct",
    "directly",
    "exact",
    "give",
    "identify",
    "impacted",
    "map",
    "metric",
    "most",
    "one",
    "or",
    "practice",
    "process",
    "required",
    "select",
    "skill",
    "specific",
    "stored",
    "the",
    "two",
    "which",
}
RELATED_ENTITY_OPTION_LIMIT = 6
EXPLICIT_SCOR_ENTITY_CODE_PATTERN = re.compile(
    r"\b(?:BP|HS|AM|AG|CO|EV|RL|RS|SC|OE|P|O|S|T|F|R)\.??\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
NUMERIC_MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|%|percent(?:age)?)\b",
    re.IGNORECASE,
)
MEASUREMENT_BOUNDARY_PATTERN = re.compile(
    r"\bfrom\b.+\bto\b|\bbetween\b.+\band\b|"
    r"\bstart(?:s|ed|ing)?\b.+\bend(?:s|ed|ing)?\b|"
    r"\bbegin(?:s|ning)?\b.+\b(?:finish|complete|end)(?:es|ed|ing)?\b",
    re.IGNORECASE,
)
NON_SCOPE_MEASUREMENT_TOKENS = {
    "average",
    "current",
    "metric",
    "measure",
    "measured",
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "percent",
    "percentage",
}


def _entity_code(value: dict[str, Any]) -> str:
    return str(
        value.get("normalized_code")
        or value.get("code")
        or value.get("details", {}).get("code")
        or ""
    ).strip()


def _compact_entity(value: dict[str, Any]) -> dict[str, Any]:
    details = value.get("details", {}) if isinstance(value.get("details"), dict) else {}
    summary = value.get("summary", {}) if isinstance(value.get("summary"), dict) else {}
    code = _entity_code(value)
    name = str(value.get("name") or details.get("name") or summary.get("name") or "").strip()
    definition = str(
        value.get("_match_definition")
        or value.get("definition")
        or details.get("definition")
        or summary.get("definition")
        or ""
    ).strip()
    compact = {"code": code, "name": name, "definition": definition}
    for metadata_key in ("category", "attribute", "calculation", "unit"):
        metadata_value = value.get(metadata_key) or details.get(metadata_key)
        if metadata_value:
            compact[metadata_key] = metadata_value
    level = value.get("level") or details.get("level")
    if level is not None:
        compact["level"] = level
    parent_process = value.get("parent_process") or details.get("parent_process")
    if isinstance(parent_process, dict) and parent_process.get("code"):
        compact["parent_process"] = {
            key: parent_process[key]
            for key in ("code", "name", "definition")
            if key in parent_process
        }
    return compact


def _deduplicate_entities(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
    for value in values:
        compact = _compact_entity(value)
        code = compact["code"].upper()
        if code and code not in deduplicated:
            deduplicated[code] = compact
    return list(deduplicated.values())


def _scenario_evidence_text(query_text: str) -> str:
    """Remove a trailing mapping instruction when earlier sentences contain the evidence."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", query_text)
        if sentence.strip()
    ]
    if len(sentences) < 2:
        return query_text

    evidence_sentences: list[str] = []
    removed_instruction = False
    for sentence in sentences:
        mentions_entity_type = any(
            pattern.search(sentence) for pattern in ENTITY_SCOPE_PATTERNS.values()
        )
        if mentions_entity_type and ENTITY_SELECTION_REQUEST_PATTERN.search(sentence):
            removed_instruction = True
            continue
        evidence_sentences.append(sentence)

    evidence_text = " ".join(evidence_sentences).strip()
    if removed_instruction and len(_tokenize_text(evidence_text)) >= 3:
        return evidence_text
    return query_text


def _entity_scope_text(query_text: str, entity_type: str) -> str:
    """Extract nearby descriptors for an requested entity type without scenario rules."""
    pattern = ENTITY_SCOPE_PATTERNS.get(entity_type)
    if pattern is None:
        return ""

    scoped_phrases: list[str] = []
    for match in pattern.finditer(query_text):
        prefix = query_text[: match.start()]
        boundary = max(
            prefix.rfind(","),
            prefix.rfind(";"),
            prefix.rfind(":"),
            prefix.rfind("."),
            prefix.rfind("?"),
            prefix.rfind("!"),
        )
        prefix_tokens = _tokenize_text(prefix[boundary + 1 :])[-8:]
        suffix = query_text[match.end() :]
        suffix_boundaries = [
            position
            for position in (
                suffix.find(","),
                suffix.find(";"),
                suffix.find(":"),
                suffix.find("."),
                suffix.find("?"),
                suffix.find("!"),
            )
            if position >= 0
        ]
        suffix_boundaries.extend(
            next_match.start()
            for other_pattern in ENTITY_SCOPE_PATTERNS.values()
            for next_match in [other_pattern.search(suffix)]
            if next_match is not None
        )
        suffix_end = min(suffix_boundaries) if suffix_boundaries else len(suffix)
        suffix_tokens = _tokenize_text(suffix[:suffix_end])[:8]
        tokens = prefix_tokens + suffix_tokens
        meaningful_tokens = [
            token for token in tokens if token not in ENTITY_SCOPE_GENERIC_TOKENS
        ]
        if meaningful_tokens:
            scoped_phrases.append(" ".join(tokens))
    return " ".join(scoped_phrases)


def _requested_entity_level(query_text: str, entity_type: str) -> int | None:
    """Read an explicit level qualifier located next to an entity-type request."""
    pattern = ENTITY_SCOPE_PATTERNS.get(entity_type)
    if pattern is None:
        return None
    for match in pattern.finditer(query_text):
        nearby_text = query_text[max(0, match.start() - 80) : match.end() + 80]
        level_match = re.search(
            r"\blevel[-\s]*([0-3])\b",
            nearby_text,
            re.IGNORECASE,
        )
        if level_match:
            return int(level_match.group(1))
    return None


def _practice_anchor_score(practice: dict[str, Any], query_text: str) -> int:
    """Favor scenario-to-definition fit while preserving explicit identity matches."""
    match_score = int(practice.get("match_score") or 0)
    if practice.get("match_type") in {"direct_code", "exact_name"}:
        return match_score

    entity = _compact_entity(practice)
    evidence_text = _scenario_evidence_text(query_text)
    identity_and_definition_score = _score_text_match(
        evidence_text,
        f"{entity['code']} {entity['name']}",
        entity["definition"],
    )
    definition_evidence_score = _score_text_match(
        evidence_text,
        "",
        entity["definition"],
    )
    scoped_practice_text = _entity_scope_text(query_text, "practice")
    scoped_practice_score = (
        _score_text_match(
            scoped_practice_text,
            f"{entity['code']} {entity['name']}",
            entity["definition"],
        )
        if scoped_practice_text
        else 0
    )
    # Scenario mappings are primarily applicability tests against definitions.
    # Names still contribute, but cannot outweigh several definition-level clues.
    return (
        identity_and_definition_score
        + (3 * definition_evidence_score)
        + (2 * scoped_practice_score)
        + _soft_definition_evidence_score(evidence_text, entity["definition"])
    )


def _soft_definition_evidence_score(query_text: str, definition: str) -> int:
    """Give small credit to ordinary inflection and nominalization variants."""
    query_tokens = {
        _normalize_search_token(token)
        for token in _tokenize_text(query_text)
        if len(token) >= 5
    }
    definition_tokens = {
        _normalize_search_token(token)
        for token in _tokenize_text(definition)
        if len(token) >= 5
    }
    score = 0
    for query_token in query_tokens - definition_tokens:
        for definition_token in definition_tokens:
            common_prefix_length = 0
            for left, right in zip(query_token, definition_token):
                if left != right:
                    break
                common_prefix_length += 1
            if common_prefix_length < 4:
                continue
            if (
                difflib.SequenceMatcher(
                    None,
                    query_token,
                    definition_token,
                ).ratio()
                >= 0.74
            ):
                score += 3
                break
    return score


def _candidate_score_lookup(
    candidate_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    scores: dict[str, int] = {}
    for candidates in candidate_groups.values():
        for candidate in candidates:
            code = _entity_code(candidate).upper()
            score = candidate.get("match_score")
            if code and isinstance(score, int):
                scores[code] = max(score, scores.get(code, 0))
    return scores


def _rank_entities(
    values: list[dict[str, Any]],
    query_text: str,
    candidate_scores: dict[str, int],
    *,
    entity_type: str,
    prefer_level_three: bool,
) -> list[tuple[int, dict[str, Any]]]:
    ranked: list[tuple[int, dict[str, Any]]] = []
    scoped_query_text = _entity_scope_text(query_text, entity_type)
    requested_entity_level = _requested_entity_level(query_text, entity_type)
    for value in _deduplicate_entities(values):
        parent_process = value.get("parent_process", {})
        entity_name = " ".join(
            filter(
                None,
                [
                    value["code"],
                    value["name"],
                    parent_process.get("code", ""),
                    parent_process.get("name", ""),
                ],
            )
        )
        entity_definition = " ".join(
            filter(
                None,
                [
                    value["definition"],
                    str(value.get("category", "")),
                    str(value.get("attribute", "")),
                    str(value.get("calculation", "")),
                    str(value.get("unit", "")),
                    parent_process.get("definition", ""),
                ],
            )
        )
        score = _score_text_match(
            query_text,
            entity_name,
            entity_definition,
        )
        if scoped_query_text:
            score += 3 * _score_text_match(
                scoped_query_text,
                entity_name,
                entity_definition,
            )
        if entity_type == "metric" and requested_entity_level is not None:
            metric_code_parts = value["code"].split(".")
            metric_level = (
                int(metric_code_parts[1])
                if len(metric_code_parts) >= 3
                and metric_code_parts[1].isdigit()
                else None
            )
            if metric_level == requested_entity_level:
                score += 36
        score += candidate_scores.get(value["code"].upper(), 0) // 3
        if (
            entity_type == "process"
            and prefer_level_three
            and LEVEL_THREE_PROCESS_PATTERN.fullmatch(value["code"])
        ):
            score += 18
        ranked.append((score, value))
    return sorted(ranked, key=lambda item: (-item[0], item[1]["code"]))


def _exact_entity_result(
    code: str,
    search_function: SearchFunction,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    normalized = code.upper()
    if normalized in cache:
        return cache[normalized]
    result = search_function(
        entity_codes=[code],
        query_text="",
        entity_types=[],
        limit=1,
    )
    matches = result.get("results", []) if isinstance(result, dict) else []
    exact = next(
        (
            item
            for item in matches
            if isinstance(item, dict) and item.get("found") and _entity_code(item)
        ),
        None,
    )
    if exact:
        cache[normalized] = exact
    return exact


def _relationship_codes(item: dict[str, Any], relation_name: str) -> set[str]:
    related = item.get("related", {}) if isinstance(item.get("related"), dict) else {}
    values = related.get(relation_name, [])
    if not isinstance(values, list):
        return set()
    return {_entity_code(value).upper() for value in values if isinstance(value, dict) and _entity_code(value)}


def _build_relationships(
    practice: dict[str, Any] | None,
    process_item: dict[str, Any] | None,
    process_entity: dict[str, Any] | None,
    metric: dict[str, Any] | None,
    skill: dict[str, Any] | None,
) -> list[dict[str, str]]:
    relationships: list[dict[str, str]] = []
    practice_code = _entity_code(practice or {})
    process_code = _entity_code(process_entity or {})
    metric_code = _entity_code(metric or {})
    skill_code = _entity_code(skill or {})

    practice_related = (
        practice.get("related", {})
        if practice and isinstance(practice.get("related"), dict)
        else {}
    )
    process_metric_codes = _relationship_codes(process_item or {}, "metrics")
    process_skill_codes = _relationship_codes(process_item or {}, "people_skills")

    if practice_code and process_code:
        relationships.append(
            {
                "from": practice_code,
                "relation": "supports_processes",
                "to": process_code,
            }
        )
    if metric_code:
        practice_metric_codes = {
            _entity_code(value).upper()
            for value in practice_related.get("metrics", [])
            if isinstance(value, dict) and _entity_code(value)
        }
        if practice_code and metric_code.upper() in practice_metric_codes:
            relationships.append(
                {
                    "from": practice_code,
                    "relation": "impacts_metrics",
                    "to": metric_code,
                }
            )
        if process_code and metric_code.upper() in process_metric_codes:
            relationships.append(
                {
                    "from": process_code,
                    "relation": "performance_metrics",
                    "to": metric_code,
                }
            )
    if skill_code:
        practice_skill_codes = {
            _entity_code(value).upper()
            for value in practice_related.get("people_skills", [])
            if isinstance(value, dict) and _entity_code(value)
        }
        if practice_code and skill_code.upper() in practice_skill_codes:
            relationships.append(
                {
                    "from": practice_code,
                    "relation": "required_skills",
                    "to": skill_code,
                }
            )
        if process_code and skill_code.upper() in process_skill_codes:
            relationships.append(
                {
                    "from": process_code,
                    "relation": "associated_skills",
                    "to": skill_code,
                }
            )
    return relationships


def _path_key(path: dict[str, Any], requested_types: list[str]) -> tuple[str, ...]:
    entities = path.get("entities", {})
    return tuple(
        str(entities.get(entity_type, {}).get("code", "")).upper()
        for entity_type in requested_types
    )


def _build_practice_paths(
    query_text: str,
    requested_types: list[str],
    candidate_groups: dict[str, list[dict[str, Any]]],
    search_function: SearchFunction,
) -> list[dict[str, Any]]:
    practices = candidate_groups.get("practice", [])
    if not practices:
        return []

    candidate_scores = _candidate_score_lookup(candidate_groups)
    prefer_level_three = not HIGH_LEVEL_PROCESS_REQUEST_PATTERN.search(query_text)
    prefer_practice_metric = bool(PRACTICE_METRIC_RELATION_PATTERN.search(query_text))
    prefer_process_metric = bool(PROCESS_METRIC_RELATION_PATTERN.search(query_text))
    prefer_practice_skill = bool(PRACTICE_SKILL_RELATION_PATTERN.search(query_text))
    prefer_process_skill = bool(PROCESS_SKILL_RELATION_PATTERN.search(query_text))
    exact_cache: dict[str, dict[str, Any]] = {}
    paths: list[dict[str, Any]] = []

    for practice in practices:
        practice_entity = _compact_entity(practice)
        practice_score = _practice_anchor_score(practice, query_text)
        related = practice.get("related", {}) if isinstance(practice.get("related"), dict) else {}
        process_values = related.get("processes", []) if "process" in requested_types else []
        ranked_processes = _rank_entities(
            process_values,
            query_text,
            candidate_scores,
            entity_type="process",
            prefer_level_three=prefer_level_three,
        )
        process_options: list[tuple[int, dict[str, Any] | None]] = (
            [
                (score, value)
                for score, value in ranked_processes[:RELATED_ENTITY_OPTION_LIMIT]
            ]
            if "process" in requested_types
            else [(0, None)]
        )
        if "process" in requested_types and not process_options:
            continue

        for process_score, process_entity in process_options:
            process_item = (
                _exact_entity_result(process_entity["code"], search_function, exact_cache)
                if process_entity
                else None
            )
            process_related = (
                process_item.get("related", {})
                if process_item and isinstance(process_item.get("related"), dict)
                else {}
            )

            metric_values: list[dict[str, Any]] = []
            if "metric" in requested_types:
                metric_values.extend(related.get("metrics", []))
                metric_values.extend(process_related.get("metrics", []))
            ranked_metrics = _rank_entities(
                metric_values,
                query_text,
                candidate_scores,
                entity_type="metric",
                prefer_level_three=False,
            )
            metric_options: list[tuple[int, dict[str, Any] | None]] = (
            [
                (score, value)
                for score, value in ranked_metrics[:RELATED_ENTITY_OPTION_LIMIT]
            ]
                if "metric" in requested_types
                else [(0, None)]
            )
            if "metric" in requested_types and not metric_options:
                continue

            skill_values: list[dict[str, Any]] = []
            if "skill" in requested_types:
                skill_values.extend(related.get("people_skills", []))
                skill_values.extend(process_related.get("people_skills", []))
            ranked_skills = _rank_entities(
                skill_values,
                query_text,
                candidate_scores,
                entity_type="skill",
                prefer_level_three=False,
            )
            skill_options: list[tuple[int, dict[str, Any] | None]] = (
            [
                (score, value)
                for score, value in ranked_skills[:RELATED_ENTITY_OPTION_LIMIT]
            ]
                if "skill" in requested_types
                else [(0, None)]
            )
            if "skill" in requested_types and not skill_options:
                continue

            for (metric_score, metric), (skill_score, skill) in product(
                metric_options,
                skill_options,
            ):
                entities: dict[str, dict[str, Any]] = {"practice": practice_entity}
                if process_entity:
                    entities["process"] = process_entity
                if metric:
                    entities["metric"] = metric
                if skill:
                    entities["skill"] = skill
                relationships = _build_relationships(
                    practice,
                    process_item,
                    process_entity,
                    metric,
                    skill,
                )
                connected_entity_codes = {
                    relation.get("to", "").upper()
                    for relation in relationships
                }
                relationship_bonus = 14 * sum(
                    1
                    for entity_type, entity in entities.items()
                    if entity_type != "practice"
                    and entity.get("code", "").upper() in connected_entity_codes
                )
                metric_relations = {
                    relation["relation"]
                    for relation in relationships
                    if metric and relation.get("to") == metric.get("code")
                }
                skill_relations = {
                    relation["relation"]
                    for relation in relationships
                    if skill and relation.get("to") == skill.get("code")
                }
                if "impacts_metrics" in metric_relations and not prefer_process_metric:
                    relationship_bonus += 55 if prefer_practice_metric else 48
                if "performance_metrics" in metric_relations and prefer_process_metric:
                    relationship_bonus += 55
                if "required_skills" in skill_relations and not prefer_process_skill:
                    relationship_bonus += 45 if prefer_practice_skill else 34
                if "associated_skills" in skill_relations and prefer_process_skill:
                    relationship_bonus += 55
                path_score = (
                    (practice_score * 2)
                    + process_score
                    + metric_score
                    + skill_score
                    + relationship_bonus
                )
                paths.append(
                    {
                        "score": path_score,
                        "anchor_score": practice_score,
                        "entities": entities,
                        "verified_relationships": relationships,
                        "anchor_match_type": practice.get("match_type", "text_search"),
                    }
                )

    unique_paths: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in sorted(
        paths,
        key=lambda item: (
            -item.get("anchor_score", item["score"]),
            -item["score"],
        ),
    ):
        key = _path_key(path, requested_types)
        if all(key) and key not in unique_paths:
            unique_paths[key] = path
    return list(unique_paths.values())


def _build_process_paths(
    query_text: str,
    requested_types: list[str],
    candidate_groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidate_scores = _candidate_score_lookup(candidate_groups)
    prefer_level_three = not HIGH_LEVEL_PROCESS_REQUEST_PATTERN.search(query_text)
    paths: list[dict[str, Any]] = []

    for process_item in candidate_groups.get("process", []):
        process_entity = _compact_entity(process_item)
        process_score = int(process_item.get("match_score") or 0)
        if prefer_level_three and LEVEL_THREE_PROCESS_PATTERN.fullmatch(process_entity["code"]):
            process_score += 18
        related = process_item.get("related", {}) if isinstance(process_item.get("related"), dict) else {}

        relation_options: dict[str, list[tuple[int, dict[str, Any] | None]]] = {}
        relation_names = {
            "practice": "best_practices",
            "metric": "metrics",
            "skill": "people_skills",
        }
        for entity_type in requested_types:
            if entity_type == "process":
                continue
            ranked = _rank_entities(
                related.get(relation_names[entity_type], []),
                query_text,
                candidate_scores,
                entity_type=entity_type,
                prefer_level_three=False,
            )
            if not ranked:
                relation_options[entity_type] = []
                continue
            relation_options[entity_type] = [(score, value) for score, value in ranked[:4]]
        if any(not values for values in relation_options.values()):
            continue

        ordered_types = [entity_type for entity_type in requested_types if entity_type != "process"]
        option_product = product(*(relation_options[entity_type] for entity_type in ordered_types))
        for choices in option_product:
            entities = {"process": process_entity}
            relationships: list[dict[str, str]] = []
            score = process_score
            for entity_type, (entity_score, entity) in zip(ordered_types, choices):
                if not entity:
                    continue
                entities[entity_type] = entity
                score += entity_score
                relation = {
                    "practice": "supporting_best_practices",
                    "metric": "performance_metrics",
                    "skill": "associated_skills",
                }[entity_type]
                relationships.append(
                    {
                        "from": process_entity["code"],
                        "relation": relation,
                        "to": entity["code"],
                    }
                )
            score += len(relationships) * 14
            paths.append(
                {
                    "score": score,
                    "anchor_score": process_score,
                    "entities": entities,
                    "verified_relationships": relationships,
                    "anchor_match_type": process_item.get("match_type", "text_search"),
                }
            )

    unique_paths: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in sorted(
        paths,
        key=lambda item: (
            -item.get("anchor_score", item["score"]),
            -item["score"],
        ),
    ):
        key = _path_key(path, requested_types)
        if all(key) and key not in unique_paths:
            unique_paths[key] = path
    return list(unique_paths.values())


def _build_metric_paths(
    query_text: str,
    requested_types: list[str],
    candidate_groups: dict[str, list[dict[str, Any]]],
    search_function: SearchFunction,
) -> list[dict[str, Any]]:
    """Expand metric candidates through their authoritative measured-process field."""
    if "process" not in requested_types:
        return []

    candidate_scores = _candidate_score_lookup(candidate_groups)
    prefer_level_three = not HIGH_LEVEL_PROCESS_REQUEST_PATTERN.search(query_text)
    exact_cache: dict[str, dict[str, Any]] = {}
    paths: list[dict[str, Any]] = []

    for metric_item in candidate_groups.get("metric", []):
        metric_entity = _compact_entity(metric_item)
        metric_score = int(metric_item.get("match_score") or 0)
        details = metric_item.get("details", {})
        measured_process_codes = (
            details.get("measured_processes", [])
            if isinstance(details, dict)
            else []
        )
        process_values: list[dict[str, Any]] = []
        for process_code in measured_process_codes:
            process_item = _exact_entity_result(
                str(process_code),
                search_function,
                exact_cache,
            )
            if process_item:
                process_values.append(process_item)
        ranked_processes = _rank_entities(
            process_values,
            query_text,
            candidate_scores,
            entity_type="process",
            prefer_level_three=prefer_level_three,
        )
        for process_score, process_entity in ranked_processes[:12]:
            paths.append(
                {
                    "score": metric_score + process_score + 14,
                    "anchor_score": metric_score,
                    "entities": {
                        "metric": metric_entity,
                        "process": process_entity,
                    },
                    "verified_relationships": [
                        {
                            "from": metric_entity["code"],
                            "relation": "measures_process_performance",
                            "to": process_entity["code"],
                        }
                    ],
                    "anchor_match_type": metric_item.get(
                        "match_type",
                        "text_search",
                    ),
                }
            )

    unique_paths: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in sorted(paths, key=lambda item: -item["score"]):
        key = _path_key(path, requested_types)
        if all(key) and key not in unique_paths:
            unique_paths[key] = path
    return list(unique_paths.values())


def _build_metric_selection_assessment(
    query_text: str,
    requested_types: list[str],
    candidate_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Detect when a numeric observation cannot uniquely identify a metric scope."""
    selection_request = bool(ENTITY_SELECTION_REQUEST_PATTERN.search(query_text))
    numeric_observation = bool(NUMERIC_MEASUREMENT_PATTERN.search(query_text))
    explicit_code = bool(EXPLICIT_SCOR_ENTITY_CODE_PATTERN.search(query_text))
    if (
        "metric" not in requested_types
        or not selection_request
        or not numeric_observation
        or explicit_code
    ):
        return {
            "request_mode": "entity_lookup" if explicit_code else "entity_selection",
            "selection_status": "not_assessed",
        }

    metric_candidates = candidate_groups.get("metric", [])
    if len(metric_candidates) < 2:
        return {
            "request_mode": "entity_selection",
            "selection_status": "insufficient_candidates",
        }

    top_metric = _compact_entity(metric_candidates[0])
    top_details = metric_candidates[0].get("details", {})
    top_attribute = (
        str(top_details.get("attribute", "")).strip().lower()
        if isinstance(top_details, dict)
        else ""
    )
    top_name_tokens = set(_tokenize_text(top_metric.get("name", "")))
    relevant_candidates: list[dict[str, Any]] = []
    for candidate in metric_candidates:
        compact = _compact_entity(candidate)
        details = candidate.get("details", {})
        candidate_attribute = (
            str(details.get("attribute", "")).strip().lower()
            if isinstance(details, dict)
            else ""
        )
        name_tokens = set(_tokenize_text(compact.get("name", "")))
        shared_name_tokens = top_name_tokens.intersection(name_tokens)
        same_performance_attribute = (
            not top_attribute
            or not candidate_attribute
            or candidate_attribute == top_attribute
        )
        if same_performance_attribute and (
            candidate.get("match_type") in {"exact_name", "fuzzy_name"}
            or len(shared_name_tokens) >= 2
        ):
            relevant_candidates.append(
                {
                    "code": compact["code"],
                    "name": compact["name"],
                    "definition": compact["definition"],
                    "unit": details.get("unit", "") if isinstance(details, dict) else "",
                    "attribute": details.get("attribute", "") if isinstance(details, dict) else "",
                    "measured_processes": (
                        details.get("measured_processes", [])
                        if isinstance(details, dict)
                        else []
                    ),
                    "match_type": candidate.get("match_type", "text_search"),
                }
            )

    if len(relevant_candidates) < 2:
        return {
            "request_mode": "entity_selection",
            "selection_status": "single_supported_candidate",
        }

    name_token_sets = [
        set(_tokenize_text(candidate["name"]))
        for candidate in relevant_candidates
    ]
    common_name_tokens = set.intersection(*name_token_sets) if name_token_sets else set()
    discriminating_tokens = (
        set().union(*name_token_sets)
        - common_name_tokens
        - NON_SCOPE_MEASUREMENT_TOKENS
    )
    query_tokens = set(_tokenize_text(query_text))
    has_scope_evidence = bool(query_tokens.intersection(discriminating_tokens))
    has_boundary_evidence = bool(MEASUREMENT_BOUNDARY_PATTERN.search(query_text))

    if has_scope_evidence:
        return {
            "request_mode": "entity_selection",
            "selection_status": "scope_evidence_present",
            "candidate_metrics": relevant_candidates,
        }

    missing_evidence = ["operational scope or metric subtype"]
    if not has_boundary_evidence:
        missing_evidence.extend(["measurement start event", "measurement end event"])
    if any("order" in set(_tokenize_text(candidate["name"])) for candidate in relevant_candidates):
        missing_evidence.append("order or transaction type")
    return {
        "request_mode": "entity_selection",
        "selection_status": "ambiguous",
        "candidate_metrics": relevant_candidates,
        "missing_evidence": missing_evidence,
        "reason": (
            "A numeric value and unit do not distinguish among the retrieved metric scopes. "
            "An exact code requires the missing operational and measurement-boundary evidence."
        ),
    }


def retrieve_scor_mapping_paths(
    query_text: str,
    requested_types: list[str],
    search_function: SearchFunction,
    *,
    search_limit: int = 8,
) -> dict[str, Any] | None:
    """Retrieve every requested entity type and assemble only source-backed relation paths."""
    selection_mode = len(requested_types) > 1
    candidate_groups: dict[str, list[dict[str, Any]]] = {}
    for entity_type in requested_types:
        result = search_function(
            entity_codes=[],
            query_text=query_text,
            entity_types=[entity_type],
            limit=search_limit,
            include_alternatives=selection_mode,
        )
        candidates = [
            item
            for item in result.get("results", [])
            if isinstance(item, dict) and item.get("found")
        ] if isinstance(result, dict) else []
        candidate_groups[entity_type] = candidates

    if not any(candidate_groups.values()):
        return None

    if len(requested_types) == 1:
        entity_type = requested_types[0]
        return {
            "mode": "entity_candidates",
            "requested_types": requested_types,
            "candidate_groups": candidate_groups,
            "results": candidate_groups.get(entity_type, []),
            "result_count": len(candidate_groups.get(entity_type, [])),
        }

    paths: list[dict[str, Any]] = []
    if "practice" in requested_types:
        paths.extend(
            _build_practice_paths(
                query_text,
                requested_types,
                candidate_groups,
                search_function,
            )
        )
    if "process" in requested_types:
        paths.extend(_build_process_paths(query_text, requested_types, candidate_groups))
    if set(requested_types).issubset({"metric", "process"}):
        paths.extend(
            _build_metric_paths(
                query_text,
                requested_types,
                candidate_groups,
                search_function,
            )
        )

    unique_paths: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in sorted(
        paths,
        key=lambda item: (
            -item.get("anchor_score", item["score"]),
            -item["score"],
        ),
    ):
        key = _path_key(path, requested_types)
        if all(key) and key not in unique_paths:
            unique_paths[key] = path
    paths = list(unique_paths.values())
    selection_assessment = _build_metric_selection_assessment(
        query_text,
        requested_types,
        candidate_groups,
    )

    flattened_results = [
        item
        for entity_type in requested_types
        for item in candidate_groups.get(entity_type, [])
    ]
    return {
        "mode": "relationship_paths" if paths else "entity_candidates",
        "requested_types": requested_types,
        "candidate_groups": candidate_groups,
        "candidate_count_by_type": {
            entity_type: len(candidates)
            for entity_type, candidates in candidate_groups.items()
        },
        "selection_assessment": selection_assessment,
        "paths": paths,
        "results": flattened_results,
        "result_count": len(flattened_results),
    }
