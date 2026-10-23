import difflib
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from .dictionary import _score_text_match, _search_dictionary_terms
from .supply_chain_knowledge import SUPPLY_CHAIN_KNOWLEDGE_TOOL, search_supply_chain_knowledge
from ..utils.shared import (
        CODE_CHUNK_PATTERN,
        ENTITY_TYPE_LABELS,
        KNOWN_SECTION_NAMES,
        KNOWN_TOP_LEVEL_PROCESSES,
        SCOR_CODE_PATTERN,
        SLASH_PROCESS_PATTERN,
        _canonical_code_token,
        _extract_quoted_phrases,
        _get_first_present,
        _get_scalar_text,
        _load_json,
        _load_scor_entity_code_lists,
        _log_tool_end,
        _log_tool_start,
        _normalize_name_for_match,
        _summarize_for_log,
        _tokenize_text,
        _tool_log,
        _truncate_text,
)


# Record normalization
HIGHER_LEVEL_PROCESS_QUERY_PATTERN = re.compile(
    r"\b(?:level[- ]?[012]|higher[- ]level|high[- ]level|top[- ]level|parent (?:process|level)|"
    r"strategic (?:view|overview|level|process landscape)|end[- ]to[- ]end(?: process| view| map)?|"
    r"process (?:area|category|group|family))\b",
    re.IGNORECASE,
)
LEVEL_THREE_PROCESS_CODE_PATTERN = re.compile(r"^[A-Z]+\d+\.\d+$", re.IGNORECASE)


def _match_any(values: list[str], candidates: list[str]) -> bool:
    """Return True when at least one filter value appears in the candidate list."""
    if not values:
        return True
    candidate_set = {candidate.lower() for candidate in candidates}
    return any(value.lower() in candidate_set for value in values)


def _contains_code_prefix(code_filters: list[str], code_value: str) -> bool:
    """Check whether a code starts with any of the requested code prefixes."""
    if not code_filters:
        return True
    code_upper = code_value.upper()
    return any(code_upper.startswith(code_filter.upper()) for code_filter in code_filters)


def _matches_metric_hierarchy(metric: dict[str, Any], metric_filters: list[str]) -> bool:
    """Check whether a metric matches requested hierarchy levels or parents."""
    if not metric_filters:
        return True

    code_upper = metric["code"].upper()
    parent_codes = {
        value.upper()
        for value in [
            metric["level1_metric"],
            metric["level2_metric"],
            metric["level3_metric"],
        ]
        if value
    }

    for metric_filter in metric_filters:
        filter_upper = metric_filter.upper()
        if code_upper.startswith(filter_upper):
            return True
        if filter_upper in parent_codes:
            return True
    return False


def _normalize_process(process: dict[str, Any]) -> dict[str, Any]:
    """Map one raw process record into the internal normalized shape."""
    return {
        "code": process["process_element_code"],
        "name": process["process_element_name"],
        "definition": process["process_element_definition"],
        "related_best_practices": _get_first_present(process, "supporting_best_practices", "related_best_practices", default=[]) or [],
        "related_metrics": _get_first_present(process, "performance_metrics", "related_metrics", default=[]) or [],
        "related_people_skills": _get_first_present(process, "associated_skills", "related_people_skills", default=[]) or [],
        "process_inputs": _get_first_present(process, "input_processes", "process_inputs", default=[]) or [],
        "process_outputs": _get_first_present(process, "output_processes", "process_outputs", default=[]) or [],
    }


def _normalize_practice(practice: dict[str, Any]) -> dict[str, Any]:
    """Map one raw best-practice record into the internal normalized shape."""
    return {
        "code": practice["best_practice_code"],
        "name": practice["best_practice_name"],
        "definition": practice["best_practice_definition"],
        "categories": practice.get("categories") or [],
        "pillars": practice.get("pillars") or [],
        "related_metrics": _get_first_present(practice, "impacts_metrics", "related_metrics", default=[]) or [],
        "related_people_skills": _get_first_present(practice, "required_skills", "related_people_skills", default=[]) or [],
        "related_processes": _get_first_present(practice, "supports_processes", default=[]) or [],
    }


def _normalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    """Map one raw metric record into the internal normalized shape."""
    return {
        "code": metric["metric_code"],
        "name": metric["metric_name"],
        "definition": metric.get("metric_definition") or "",
        "category": _get_scalar_text(metric.get("metric_category")),
        "attribute": _get_scalar_text(metric.get("metric_attribute")),
        "calculation": metric.get("metric_calculation") or "",
        "unit": metric.get("metric_unit") or "",
        "level1_metric": metric.get("level1_metric") or "",
        "level2_metric": metric.get("level2_metric") or "",
        "level3_metric": metric.get("level3_metric") or "",
        "related_processes": _get_first_present(metric, "measures_process_performance", default=[]) or [],
        "related_best_practices": _get_first_present(metric, "improved_by_best_practice", default=[]) or [],
    }


def _normalize_skill(skill: dict[str, Any]) -> dict[str, Any]:
    """Map one raw skill record into the internal normalized shape."""
    return {
        "code": skill["people_skill_code"],
        "name": skill["people_skill_name"],
        "definition": skill["people_skill_definition"],
        "related_experiences": skill.get("related_experiences") or [],
        "related_trainings": skill.get("related_trainings") or [],
        "related_processes": _get_first_present(skill, "applicable_processes", default=[]) or [],
        "related_best_practices": _get_first_present(skill, "supports_best_practices", "applicable_best_practices", default=[]) or [],
    }


def _extract_entity_name_candidates(query_text: str, entity_types: set[str]) -> list[str]:
    """Extract likely entity-name fragments from a natural-language query."""
    candidates: list[str] = []
    lowered_query = query_text.lower()
    entity_markers = []
    if "process" in entity_types:
        entity_markers.append("process")
    if "practice" in entity_types:
        entity_markers.extend(["best practice", "practice"])
    if "metric" in entity_types:
        entity_markers.append("metric")
    if "skill" in entity_types:
        entity_markers.extend(["people skill", "skill"])

    for marker in entity_markers:
        pattern = re.compile(
            rf"\b{re.escape(marker)}\s+(.+?)(?:,|\?|$)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(query_text):
            candidate = match.group(1).strip().strip("\"' ")
            candidate = re.sub(
                r"\b(list|show|only|all|directly|related|that|which|include|includes|including|have|has|with|the)\b",
                " ",
                candidate,
                flags=re.IGNORECASE,
            )
            candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;")
            normalized_candidate = _normalize_name_for_match(candidate)
            if normalized_candidate and normalized_candidate not in {
                _normalize_name_for_match(existing) for existing in candidates
            }:
                candidates.append(candidate)

    if not candidates and "process" in entity_types and "process" in lowered_query:
        tail = query_text.lower().split("process", 1)[1].strip()
        tail = re.split(r"[,?]", tail, maxsplit=1)[0].strip().strip("\"' ")
        if tail:
            candidates.append(tail)
    return candidates


def _iter_entities_for_search(entity_type: str) -> list[dict[str, Any]]:
    """Return normalized entities for the requested search type."""
    if entity_type == "process":
        return list(_get_processes())
    if entity_type == "practice":
        return [_normalize_practice(item) for item in _load_json("practices.json")]
    if entity_type == "metric":
        return list(_get_metrics())
    if entity_type == "skill":
        return list(_get_skills())
    return []


def _find_exact_name_matches(
    query_text: str,
    entity_types: set[str],
) -> list[tuple[int, str, str, str]]:
    """Find entities whose names exactly match a phrase in the query."""
    exact_phrases = [_normalize_name_for_match(value) for value in _extract_quoted_phrases(query_text)]
    normalized_query = _normalize_name_for_match(query_text)
    is_direct_lookup_query = bool(
        re.search(
            r"\b(?:define|what is|what does|explain|details (?:for|of)|tell me about|"
            r"look up|lookup)\b",
            normalized_query,
        )
    )

    raw_matches: list[tuple[int, str, str, str, str]] = []
    seen_codes: set[tuple[str, str]] = set()
    for entity_type in sorted(entity_types):
        for entity in _iter_entities_for_search(entity_type):
            raw_name = str(entity.get("name", ""))
            normalized_name = _normalize_name_for_match(raw_name)
            is_exact_match = normalized_name in exact_phrases
            acronym_aliases = [
                _normalize_name_for_match(alias)
                for alias in re.findall(r"\(([A-Z][A-Z0-9&-]{1,})\)", raw_name)
            ]
            if not is_exact_match:
                is_exact_match = any(
                    alias
                    and f" {alias} " in f" {normalized_query} "
                    and (
                        is_direct_lookup_query
                        or bool(
                            re.search(
                                rf"\b(?:recommend|implement|adopt|use|assess|evaluate)\w*\b"
                                rf".{{0,60}}\b{re.escape(alias)}\b",
                                normalized_query,
                            )
                        )
                    )
                    for alias in acronym_aliases
                )
            if not is_exact_match and normalized_name:
                name_is_present = f" {normalized_name} " in f" {normalized_query} "
                name_token_count = len(normalized_name.split())
                # A single activity word such as "receiving" may occur incidentally in an
                # operational scenario. Treat it as an exact entity name only when the user is
                # explicitly looking up that name; otherwise let relevance search evaluate it as
                # a candidate alongside the described process evidence.
                is_exact_match = name_is_present and (
                    name_token_count > 1 or is_direct_lookup_query
                )
            if is_exact_match:
                dedupe_key = (entity_type, entity["code"])
                if dedupe_key in seen_codes:
                    continue
                seen_codes.add(dedupe_key)
                raw_matches.append((10_000, entity["code"], entity_type, entity["name"], normalized_name))

    filtered_matches: list[tuple[int, str, str, str]] = []
    for score, code, entity_type, name, normalized_name in sorted(
        raw_matches,
        key=lambda item: (-len(item[4]), item[1]),
    ):
        if normalized_name and any(
            normalized_name != other_normalized_name and normalized_name in other_normalized_name
            for _, _, _, _, other_normalized_name in raw_matches
        ):
            continue
        filtered_matches.append((score, code, entity_type, name))
    return filtered_matches


def _find_fuzzy_name_matches(
    query_text: str,
    entity_types: set[str],
) -> list[tuple[int, str, str, str]]:
    """Find close name matches when the query contains minor spelling errors."""
    candidates = _extract_entity_name_candidates(query_text, entity_types)
    if not candidates:
        return []

    matches: list[tuple[int, str, str, str]] = []
    seen_codes: set[tuple[str, str]] = set()
    for entity_type in sorted(entity_types):
        for entity in _iter_entities_for_search(entity_type):
            normalized_name = _normalize_name_for_match(entity.get("name", ""))
            if not normalized_name:
                continue
            best_ratio = 0.0
            for candidate in candidates:
                normalized_candidate = _normalize_name_for_match(candidate)
                if not normalized_candidate:
                    continue
                ratio = difflib.SequenceMatcher(None, normalized_candidate, normalized_name).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
            if best_ratio < 0.84:
                continue
            dedupe_key = (entity_type, entity["code"])
            if dedupe_key in seen_codes:
                continue
            seen_codes.add(dedupe_key)
            matches.append((9_000 + int(best_ratio * 100), entity["code"], entity_type, entity["name"]))

    return sorted(matches, key=lambda item: (-item[0], item[1]))


def _compact_process(process: dict[str, Any]) -> dict[str, Any]:
    """Return the compact process shape used inside aggregated responses."""
    result = {
        "code": process["code"],
        "name": process["name"],
        "definition": _truncate_text(process.get("definition") or ""),
    }
    parent_code = process["code"].rsplit(".", 1)[0] if "." in process["code"] else ""
    if parent_code:
        parent = next(
            (candidate for candidate in _get_processes() if candidate["code"] == parent_code),
            None,
        )
        if parent:
            result["parent_process"] = {
                "code": parent["code"],
                "name": parent["name"],
                "definition": _truncate_text(parent.get("definition") or ""),
            }
    return result


def _get_processes_by_exact_codes(process_codes: list[str]) -> list[dict[str, Any]]:
    """Fetch normalized processes by exact code after code normalization."""
    if not process_codes:
        return []

    normalized_codes = {code.upper() for code in _normalize_code_list(process_codes)}
    return [
        process
        for process in _get_processes()
        if process["code"].upper() in normalized_codes
    ]


def _compact_practice(practice: dict[str, Any]) -> dict[str, Any]:
    """Return the compact best-practice shape used inside aggregated responses."""
    return {
        "code": practice["code"],
        "name": practice["name"],
        "definition": _truncate_text(practice.get("definition") or ""),
        "categories": practice.get("categories", []),
        "pillars": practice.get("pillars", []),
    }


def _get_practices_by_exact_codes(practice_codes: list[str]) -> list[dict[str, Any]]:
    """Fetch normalized best practices by exact code after code normalization."""
    if not practice_codes:
        return []

    normalized_codes = {code.upper() for code in _normalize_code_list(practice_codes)}
    practices = [_normalize_practice(item) for item in _load_json("practices.json")]
    return [
        practice
        for practice in practices
        if practice["code"].upper() in normalized_codes
    ]


def _get_metrics_by_exact_codes(metric_codes: list[str]) -> list[dict[str, Any]]:
    """Fetch normalized metrics by exact code after code normalization."""
    if not metric_codes:
        return []

    normalized_codes = {code.upper() for code in _normalize_code_list(metric_codes)}
    return [
        metric
        for metric in _get_metrics()
        if metric["code"].upper() in normalized_codes
    ]


def _compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    """Return the compact metric shape used inside aggregated responses."""
    return {
        "code": metric["code"],
        "name": metric["name"],
        "definition": _truncate_text(metric.get("definition") or ""),
        "category": metric.get("category") or "",
        "attribute": metric.get("attribute") or "",
        "calculation": metric.get("calculation") or "",
        "unit": metric.get("unit") or "",
        "measured_processes": metric.get("related_processes") or [],
        "improved_by_best_practices": metric.get("related_best_practices") or [],
        "level1_metric": metric.get("level1_metric") or "",
        "level2_metric": metric.get("level2_metric") or "",
        "level3_metric": metric.get("level3_metric") or "",
    }


def _compact_skill(skill: dict[str, Any]) -> dict[str, Any]:
    """Return the compact skill shape used inside aggregated responses."""
    return {
        "code": skill["code"],
        "name": skill["name"],
        "definition": _truncate_text(skill.get("definition") or ""),
    }


@lru_cache(maxsize=1)
def _get_skills() -> tuple[dict[str, Any], ...]:
    """Load and cache normalized skill records."""
    return tuple(_normalize_skill(item) for item in _load_json("skills.json"))


def _get_skills_by_exact_codes(skill_codes: list[str]) -> list[dict[str, Any]]:
    """Fetch normalized skills by exact code after code normalization."""
    if not skill_codes:
        return []

    normalized_codes = {code.upper() for code in _normalize_code_list(skill_codes)}
    return [
        skill
        for skill in _get_skills()
        if skill["code"].upper() in normalized_codes
    ]


@lru_cache(maxsize=1)
def _get_distinct_metric_attributes() -> tuple[str, ...]:
    """Return the distinct metric attributes available in the knowledge base."""
    return tuple(
        sorted(
            {
                metric["attribute"]
                for metric in _get_metrics()
                if metric.get("attribute")
            }
        )
    )


@lru_cache(maxsize=1)
def _get_distinct_metric_categories() -> tuple[str, ...]:
    """Return the distinct metric categories available in the knowledge base."""
    return tuple(
        sorted(
            {
                metric["category"]
                for metric in _get_metrics()
                if metric.get("category")
            }
        )
    )


@lru_cache(maxsize=1)
def _get_distinct_practice_pillars() -> tuple[str, ...]:
    """Return the distinct best-practice pillars available in the knowledge base."""
    return tuple(
        sorted(
            {
                pillar
                for practice in (_normalize_practice(item) for item in _load_json("practices.json"))
                for pillar in practice.get("pillars", [])
                if pillar
            }
        )
    )


@lru_cache(maxsize=1)
def _get_distinct_practice_categories() -> tuple[str, ...]:
    """Return the distinct best-practice categories available in the knowledge base."""
    return tuple(
        sorted(
            {
                category
                for practice in (_normalize_practice(item) for item in _load_json("practices.json"))
                for category in practice.get("categories", [])
                if category
            }
        )
    )


# Query filter extraction
def _extract_contextual_named_matches(
    query_text: str,
    options: tuple[str, ...],
    suffix_patterns: tuple[str, ...],
) -> list[str]:
    """Extract filter names such as pillars or categories from query context."""
    normalized_options = {
        _normalize_name_for_match(option): option
        for option in options
    }
    matches: list[str] = []
    lowered_query = query_text.lower()
    for suffix_pattern in suffix_patterns:
        pattern = re.compile(
            rf"([a-z][a-z0-9/&,\- ]{{0,80}}?)\s+{suffix_pattern}",
            re.IGNORECASE,
        )
        for match in pattern.finditer(lowered_query):
            candidate_text = _normalize_name_for_match(match.group(1))
            candidate_text = re.sub(
                r"^(?:only|related|the|a|an|in|with|for|of|list|show)\s+",
                "",
                candidate_text,
            ).strip()
            exact_option = normalized_options.get(candidate_text)
            if exact_option and exact_option not in matches:
                matches.append(exact_option)
                continue
            for normalized_option, option in normalized_options.items():
                if normalized_option and normalized_option in candidate_text and option not in matches:
                    matches.append(option)
    return matches


def _build_relation_query_filters(query_text: str) -> dict[str, Any]:
    """Parse relation-specific filters such as metric attributes or BP pillars."""
    if not query_text.strip():
        return {
            "focus_relations": set(),
            "metric_attributes": [],
            "metric_categories": [],
            "practice_pillars": [],
            "practice_categories": [],
        }

    normalized_query = _normalize_name_for_match(query_text)
    query_terms = set(normalized_query.split())
    metric_attributes = _extract_contextual_named_matches(
        query_text,
        _get_distinct_metric_attributes(),
        ("attribute", "attributes"),
    )
    metric_categories = _extract_contextual_named_matches(
        query_text,
        _get_distinct_metric_categories(),
        ("category", "categories"),
    )
    practice_pillars = _extract_contextual_named_matches(
        query_text,
        _get_distinct_practice_pillars(),
        ("pillar", "pillars"),
    )
    practice_categories = _extract_contextual_named_matches(
        query_text,
        _get_distinct_practice_categories(),
        ("category", "categories"),
    )

    focus_relations: set[str] = set()
    if query_terms.intersection({"process", "processes", "activity", "activities"}):
        focus_relations.add("processes")
    if query_terms.intersection({"metric", "metrics", "kpi", "kpis", "attribute", "attributes"}):
        focus_relations.add("metrics")
    if (
        query_terms.intersection({"practice", "practices", "pillar", "pillars", "category", "categories"})
        or "best practice" in normalized_query
        or "best practices" in normalized_query
    ):
        focus_relations.add("best_practices")
    if (
        query_terms.intersection({"skill", "skills", "training", "trainings", "experience", "experiences"})
        or "people skill" in normalized_query
        or "people skills" in normalized_query
    ):
        focus_relations.add("people_skills")
    if query_terms.intersection({"input", "inputs"}):
        focus_relations.add("inputs")
    if query_terms.intersection({"output", "outputs"}):
        focus_relations.add("outputs")

    return {
        "focus_relations": focus_relations,
        "metric_attributes": metric_attributes,
        "metric_categories": metric_categories,
        "practice_pillars": practice_pillars,
        "practice_categories": practice_categories,
    }


def _filter_metric_details(
    metric_details: list[dict[str, Any]],
    query_filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter related metrics using attribute and category constraints."""
    required_attributes = {value.lower() for value in query_filters["metric_attributes"]}
    required_categories = {value.lower() for value in query_filters["metric_categories"]}

    filtered = metric_details
    if required_attributes:
        filtered = [
            metric for metric in filtered
            if metric.get("attribute", "").lower() in required_attributes
        ]
    if required_categories:
        filtered = [
            metric for metric in filtered
            if metric.get("category", "").lower() in required_categories
        ]
    return filtered


def _filter_practice_details(
    practice_details: list[dict[str, Any]],
    query_filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter related best practices using pillar and category constraints."""
    required_pillars = {value.lower() for value in query_filters["practice_pillars"]}
    required_categories = {value.lower() for value in query_filters["practice_categories"]}

    filtered = practice_details
    if required_pillars:
        filtered = [
            practice
            for practice in filtered
            if required_pillars.intersection(
                pillar.lower() for pillar in practice.get("pillars", [])
            )
        ]
    if required_categories:
        filtered = [
            practice
            for practice in filtered
            if required_categories.intersection(
                category.lower() for category in practice.get("categories", [])
            )
        ]
    return filtered


@lru_cache(maxsize=1)
def _get_processes() -> tuple[dict[str, Any], ...]:
    """Load and cache normalized process records."""
    return tuple(_normalize_process(item) for item in _load_json("processes.json"))


@lru_cache(maxsize=1)
def _get_metrics() -> tuple[dict[str, Any], ...]:
    """Load and cache normalized metric records."""
    return tuple(_normalize_metric(item) for item in _load_json("metrics.json"))


# Code and name normalization for SCOR matching
@lru_cache(maxsize=1)
def _get_known_scor_entities() -> dict[str, set[str]]:
    """Build cached lookup sets for known SCOR codes and normalized names."""
    processes = [_normalize_process(item) for item in _load_json("processes.json")]
    practices = [_normalize_practice(item) for item in _load_json("practices.json")]
    metrics = list(_get_metrics())
    skills = [_normalize_skill(item) for item in _load_json("skills.json")]

    codes = {
        entity["code"].upper()
        for entity in [*processes, *practices, *metrics, *skills]
        if entity.get("code")
    }
    for entity_codes in _load_scor_entity_code_lists().values():
        codes.update(code.upper() for code in entity_codes)
    names = {
        _normalize_name_for_match(entity["name"])
        for entity in [*processes, *practices, *metrics, *skills]
        if entity.get("name")
    }
    names.update(KNOWN_TOP_LEVEL_PROCESSES)
    names.update(KNOWN_SECTION_NAMES)

    return {
        "codes": codes,
        "names": names,
        "top_level_processes": set(KNOWN_TOP_LEVEL_PROCESSES),
    }


@lru_cache(maxsize=1)
def _get_known_exact_code_map() -> dict[str, str]:
    """Map exact uppercase code strings to their canonical stored form."""
    known_codes = _get_known_scor_entities()["codes"]
    return {code.upper(): code for code in known_codes}


@lru_cache(maxsize=1)
def _get_known_codes_by_canonical() -> dict[str, tuple[str, ...]]:
    """Group codes by separator-free canonical token for normalization."""
    grouped: dict[str, list[str]] = {}
    for code in _get_known_scor_entities()["codes"]:
        grouped.setdefault(_canonical_code_token(code), []).append(code)
    return {
        canonical: tuple(sorted(codes, key=lambda item: (("." in item), len(item), item)))
        for canonical, codes in grouped.items()
    }


@lru_cache(maxsize=1)
def _get_code_entity_type_map() -> dict[str, str]:
    """Map canonical SCOR code tokens to their entity type names."""
    code_entity_type_map: dict[str, str] = {}
    for entity_type, codes in _load_scor_entity_code_lists().items():
        for code in codes:
            code_entity_type_map[_canonical_code_token(code)] = entity_type
    return code_entity_type_map


@lru_cache(maxsize=1)
def _get_known_canonical_codes_by_length() -> tuple[str, ...]:
    """Return canonical code tokens sorted longest-first for greedy matching."""
    return tuple(
        sorted(
            _get_known_codes_by_canonical().keys(),
            key=lambda code: (-len(code), code),
        )
    )


def _pick_best_known_code(input_value: str, candidates: tuple[str, ...]) -> str:
    """Choose the best canonical SCOR code variant for a user-provided input."""
    input_upper = input_value.upper().strip()
    if input_upper in _get_known_exact_code_map():
        return _get_known_exact_code_map()[input_upper]

    has_separator = any(separator in input_upper for separator in ".-,")
    if has_separator:
        separated_candidates = [candidate for candidate in candidates if any(sep in candidate for sep in ".-,")]
        if separated_candidates:
            return sorted(separated_candidates, key=lambda item: (len(item), item))[0]
    else:
        plain_candidates = [candidate for candidate in candidates if not any(sep in candidate for sep in ".-,")]
        if plain_candidates:
            return sorted(plain_candidates, key=lambda item: (len(item), item))[0]

    return candidates[0]


def _normalize_code_input(value: str) -> str:
    """Normalize a code-like input into the canonical stored SCOR code."""
    input_upper = value.upper().strip()
    exact_match = _get_known_exact_code_map().get(input_upper)
    if exact_match:
        return exact_match

    canonical = _canonical_code_token(value)
    candidates = _get_known_codes_by_canonical().get(canonical)
    if candidates:
        return _pick_best_known_code(value, candidates)
    return input_upper


def _segment_canonical_code_chunk(canonical_chunk: str) -> list[str]:
    """Split glued-together code text into known canonical SCOR code tokens."""
    matches: list[str] = []
    index = 0
    known_codes = _get_known_canonical_codes_by_length()
    while index < len(canonical_chunk):
        matched_code = next(
            (
                code
                for code in known_codes
                if canonical_chunk.startswith(code, index)
            ),
            None,
        )
        if matched_code is None:
            # Do not silently discard an unknown prefix or infix. That previously turned
            # nonexistent inputs such as ``DS.1`` into the known code ``S1`` by skipping the
            # leading D. Glued known codes are still supported, but the whole chunk must be
            # segmentable into known canonical tokens.
            return []
        matches.append(matched_code)
        index += len(matched_code)
    return matches


def _extract_scor_code_matches(text: str) -> list[dict[str, Any]]:
    """Extract and normalize all SCOR code matches found in free text."""
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk_match in CODE_CHUNK_PATTERN.finditer(text):
        raw_chunk = chunk_match.group(0)
        if not any(char.isdigit() for char in raw_chunk):
            continue

        canonical_chunk = _canonical_code_token(raw_chunk)
        for canonical_code in _segment_canonical_code_chunk(canonical_chunk):
            if canonical_code in seen:
                continue
            seen.add(canonical_code)

            normalized_code = _pick_best_known_code(
                raw_chunk,
                _get_known_codes_by_canonical()[canonical_code],
            )
            entity_type = _get_code_entity_type_map()[canonical_code]
            matches.append(
                {
                    "input_fragment": raw_chunk,
                    "normalized_code": normalized_code,
                    "entity_type": entity_type,
                    "entity_type_label": ENTITY_TYPE_LABELS.get(entity_type, entity_type),
                    "summary": _get_entity_summary(entity_type, normalized_code),
                }
            )

    return matches


def _get_entity_summary(entity_type: str, normalized_code: str) -> dict[str, Any]:
    """Return a lightweight summary for one normalized SCOR entity."""
    code_upper = normalized_code.upper()

    if entity_type == "processes":
        process = next(
            (item for item in _get_processes() if item["code"].upper() == code_upper),
            None,
        )
        if process:
            return {"name": process["name"], "definition": process["definition"]}

    if entity_type == "practices":
        practices = [_normalize_practice(item) for item in _load_json("practices.json")]
        practice = next(
            (item for item in practices if item["code"].upper() == code_upper),
            None,
        )
        if practice:
            return {"name": practice["name"], "definition": practice["definition"]}

    if entity_type == "metrics":
        metric = next(
            (item for item in _get_metrics() if item["code"].upper() == code_upper),
            None,
        )
        if metric:
            return {
                "name": metric["name"],
                "definition": metric["definition"],
                "category": metric["category"],
                "attribute": metric["attribute"],
            }

    if entity_type == "skills":
        skills = [_normalize_skill(item) for item in _load_json("skills.json")]
        skill = next((item for item in skills if item["code"].upper() == code_upper), None)
        if skill:
            return {"name": skill["name"], "definition": skill["definition"]}

    return {}


def _normalize_code_list(values: list[str]) -> list[str]:
    """Normalize a list of code strings while skipping empty values."""
    return [_normalize_code_input(value) for value in values if str(value).strip()]


def _find_metric(metrics: tuple[dict[str, Any], ...], metric_code: str) -> dict[str, Any] | None:
    """Find one metric by normalized code inside a metric collection."""
    metric_code_upper = _normalize_code_input(metric_code).upper()
    return next((metric for metric in metrics if metric["code"].upper() == metric_code_upper), None)


def _find_process(processes: tuple[dict[str, Any], ...], process_reference: str) -> dict[str, Any] | None:
    """Find one process by normalized code or exact normalized name."""
    normalized_reference = _normalize_name_for_match(process_reference)
    reference_upper = _normalize_code_input(process_reference).upper()
    for process in processes:
        if process["code"].upper() == reference_upper:
            return process
        if _normalize_name_for_match(process["name"]) == normalized_reference:
            return process
    return None


def _get_metric_level(metrics: tuple[dict[str, Any], ...], metric_code: str) -> int | None:
    """Infer whether a metric is level 1, 2, or 3 within the hierarchy."""
    metric_code_upper = _normalize_code_input(metric_code).upper()

    if any(metric["level1_metric"].upper() == metric_code_upper for metric in metrics if metric["level1_metric"]):
        return 1
    if any(metric["level2_metric"].upper() == metric_code_upper for metric in metrics if metric["level2_metric"]):
        return 2
    if any(metric["level3_metric"].upper() == metric_code_upper for metric in metrics if metric["level3_metric"]):
        return 3

    metric = _find_metric(metrics, metric_code)
    if metric is None:
        return None
    if metric["level3_metric"].upper() == metric_code_upper:
        return 3
    if metric["level2_metric"].upper() == metric_code_upper:
        return 2
    if metric["level1_metric"].upper() == metric_code_upper:
        return 1
    return None


def _build_metric_hierarchy(metric_code: str) -> dict[str, Any]:
    """Build parent and child hierarchy details for one metric."""
    metrics = _get_metrics()
    metric = _find_metric(metrics, metric_code)
    if metric is None:
        return {"metric_code": metric_code, "found": False}

    level = _get_metric_level(metrics, metric_code)
    code_upper = metric["code"].upper()

    parent_codes: list[str] = []
    if level == 2 and metric["level1_metric"]:
        parent_codes.append(metric["level1_metric"])
    elif level == 3:
        if metric["level1_metric"]:
            parent_codes.append(metric["level1_metric"])
        if metric["level2_metric"]:
            parent_codes.append(metric["level2_metric"])

    child_metrics: list[dict[str, str]] = []
    if level == 1:
        child_metrics = [
            {"code": child["code"], "name": child["name"], "level": "level_2"}
            for child in metrics
            if child["code"].upper() != code_upper
            and child["level1_metric"].upper() == code_upper
            and child["level2_metric"].upper() == child["code"].upper()
        ]
    elif level == 2:
        child_metrics = [
            {"code": child["code"], "name": child["name"], "level": "level_3"}
            for child in metrics
            if child["code"].upper() != code_upper
            and child["level2_metric"].upper() == code_upper
            and child["level3_metric"].upper() == child["code"].upper()
        ]

    parent_metrics = [
        {"code": parent["code"], "name": parent["name"]}
        for parent_code in parent_codes
        for parent in [ _find_metric(metrics, parent_code) ]
        if parent is not None
    ]

    return {
        "metric_code": metric["code"],
        "metric_name": metric["name"],
        "metric_level": level,
        "upper_metrics": parent_metrics,
        "child_metrics": child_metrics,
    }


# Public data access helpers
def get_process_details(process_codes: list[str]) -> list[dict[str, Any]]:
    """Return process details filtered by code prefix, or all processes if empty."""
    _log_tool_start("get_process_details", process_codes=process_codes)
    processes = _get_processes()
    normalized_process_codes = _normalize_code_list(process_codes)
    result = [
        process
        for process in processes
        if _contains_code_prefix(normalized_process_codes, process["code"])
    ]
    _log_tool_end("get_process_details", result)
    return result


def get_process_hierarchy(process_reference: str) -> dict[str, Any]:
    """Return one process together with its direct child processes."""
    _log_tool_start("get_process_hierarchy", process_reference=process_reference)
    processes = _get_processes()
    process = _find_process(processes, process_reference)
    if process is None:
        result = {"process_reference": process_reference, "found": False}
        _log_tool_end("get_process_hierarchy", result)
        return result

    process_code = process["code"]
    is_top_level_process = "." not in process_code and not any(
        char.isdigit() for char in process_code
    )
    child_processes = [
        {
            "code": candidate["code"],
            "name": candidate["name"],
            "definition": candidate["definition"],
        }
        for candidate in processes
        if candidate["code"] != process_code
        and (
            (
                is_top_level_process
                and candidate["code"].startswith(process_code)
                and "." not in candidate["code"][len(process_code):]
            )
            or (
                not is_top_level_process
                and candidate["code"].startswith(f"{process_code}.")
                and "." not in candidate["code"][len(process_code) + 1:]
            )
        )
    ]

    result = {
        "found": True,
        "process": {
            "code": process["code"],
            "name": process["name"],
            "definition": process["definition"],
        },
        "child_processes": child_processes,
        "has_child_processes": bool(child_processes),
    }
    _log_tool_end("get_process_hierarchy", result)
    return result


def get_practice_details(
    practice_codes: list[str],
    pillars: list[str],
    categories: list[str],
    metric_codes: list[str],
    people_skill_codes: list[str],
) -> list[dict[str, Any]]:
    """Return best-practice details filtered by codes, tags, and direct relations."""
    _log_tool_start(
        "get_practice_details",
        practice_codes=practice_codes,
        pillars=pillars,
        categories=categories,
        metric_codes=metric_codes,
        people_skill_codes=people_skill_codes,
    )
    practices = [_normalize_practice(item) for item in _load_json("practices.json")]
    normalized_practice_codes = _normalize_code_list(practice_codes)
    normalized_metric_codes = _normalize_code_list(metric_codes)
    normalized_people_skill_codes = _normalize_code_list(people_skill_codes)
    result = [
        practice
        for practice in practices
        if _contains_code_prefix(normalized_practice_codes, practice["code"])
        and _match_any(pillars, practice["pillars"])
        and _match_any(categories, practice["categories"])
        and _match_any(normalized_metric_codes, practice["related_metrics"])
        and _match_any(normalized_people_skill_codes, practice["related_people_skills"])
    ]
    _log_tool_end("get_practice_details", result)
    return result


def get_metric_details(
    metric_codes: list[str],
    categories: list[str],
    attributes: list[str],
    level1_metrics: list[str],
    level2_metrics: list[str],
    level3_metrics: list[str],
) -> list[dict[str, Any]]:
    """Return metric details filtered by codes, categories, attributes, and hierarchy."""
    _log_tool_start(
        "get_metric_details",
        metric_codes=metric_codes,
        categories=categories,
        attributes=attributes,
        level1_metrics=level1_metrics,
        level2_metrics=level2_metrics,
        level3_metrics=level3_metrics,
    )
    metrics = _get_metrics()
    normalized_metric_codes = _normalize_code_list(metric_codes)
    normalized_level1_metrics = _normalize_code_list(level1_metrics)
    normalized_level2_metrics = _normalize_code_list(level2_metrics)
    normalized_level3_metrics = _normalize_code_list(level3_metrics)
    result = [
        metric
        for metric in metrics
        if _matches_metric_hierarchy(metric, normalized_metric_codes)
        and _match_any(categories, [metric["category"]])
        and _match_any(attributes, [metric["attribute"]])
        and _match_any(normalized_level1_metrics, [metric["level1_metric"]])
        and _match_any(normalized_level2_metrics, [metric["level2_metric"]])
        and _match_any(normalized_level3_metrics, [metric["level3_metric"]])
    ]
    _log_tool_end("get_metric_details", result)
    return result


def get_skill_details(
    skill_codes: list[str],
    experience_codes: list[str],
    training_codes: list[str],
) -> list[dict[str, Any]]:
    """Return skill details filtered by codes, experiences, and trainings."""
    _log_tool_start(
        "get_skill_details",
        skill_codes=skill_codes,
        experience_codes=experience_codes,
        training_codes=training_codes,
    )
    skills = [_normalize_skill(item) for item in _load_json("skills.json")]
    normalized_skill_codes = _normalize_code_list(skill_codes)
    normalized_experience_codes = _normalize_code_list(experience_codes)
    normalized_training_codes = _normalize_code_list(training_codes)
    result = [
        skill
        for skill in skills
        if _contains_code_prefix(normalized_skill_codes, skill["code"])
        and _match_any(normalized_experience_codes, skill["related_experiences"])
        and _match_any(normalized_training_codes, skill["related_trainings"])
    ]
    _log_tool_end("get_skill_details", result)
    return result


def get_metric_hierarchy(metric_code: str) -> dict[str, Any]:
    """Return one metric together with its parent and child hierarchy nodes."""
    _log_tool_start("get_metric_hierarchy", metric_code=metric_code)
    result = _build_metric_hierarchy(_normalize_code_input(metric_code))
    _log_tool_end("get_metric_hierarchy", result)
    return result


def get_metric_improvement_options(metric_code: str) -> dict[str, Any]:
    """Return the best practices, processes, and skills linked to improving a metric."""
    _log_tool_start("get_metric_improvement_options", metric_code=metric_code)
    normalized_metric_code = _normalize_code_input(metric_code)
    metric_matches = get_metric_details([normalized_metric_code], [], [], [], [], [])
    metric = next(
        (item for item in metric_matches if item["code"].upper() == normalized_metric_code.upper()),
        None,
    )
    if metric is None:
        result = {"metric_code": metric_code, "found": False}
        _log_tool_end("get_metric_improvement_options", result)
        return result

    hierarchy = get_metric_hierarchy(normalized_metric_code)
    metric_scope_codes = [normalized_metric_code]
    metric_scope_codes.extend(child["code"] for child in hierarchy.get("child_metrics", []))

    practice_map: dict[str, dict[str, Any]] = {}
    for scope_code in metric_scope_codes:
        for practice in get_practice_details([], [], [], [scope_code], []):
            if practice["code"] not in practice_map:
                practice_map[practice["code"]] = {
                    **practice,
                    "matched_metric_codes": [scope_code],
                }
            elif scope_code not in practice_map[practice["code"]]["matched_metric_codes"]:
                practice_map[practice["code"]]["matched_metric_codes"].append(scope_code)

    related_practices = list(practice_map.values())
    related_processes = [
        process
        for process in _get_processes()
        if any(scope_code in process["related_metrics"] for scope_code in metric_scope_codes)
    ]

    related_skill_codes = sorted(
        {
            skill_code
            for practice in related_practices
            for skill_code in practice["related_people_skills"]
        }
    )
    related_skills = [
        skill
        for skill in [_normalize_skill(item) for item in _load_json("skills.json")]
        if skill["code"] in related_skill_codes
    ]

    result = {
        "found": True,
        "metric": {
            "code": metric["code"],
            "name": metric["name"],
            "definition": metric["definition"],
            "category": metric["category"],
            "attribute": metric["attribute"],
        },
        "metric_scope_codes": metric_scope_codes,
        "related_best_practices": [
            {
                "code": practice["code"],
                "name": practice["name"],
                "definition": practice["definition"],
                "categories": practice["categories"],
                "pillars": practice["pillars"],
                "related_people_skills": practice["related_people_skills"],
                "matched_metric_codes": practice["matched_metric_codes"],
            }
            for practice in related_practices
        ],
        "related_processes": [
            {
                "code": process["code"],
                "name": process["name"],
                "definition": process["definition"],
            }
            for process in related_processes
        ],
        "related_people_skills": [
            {
                "code": skill["code"],
                "name": skill["name"],
                "definition": skill["definition"],
            }
            for skill in related_skills
        ],
    }
    _log_tool_end("get_metric_improvement_options", result)
    return result


def identify_scor_code(code: str) -> dict[str, Any]:
    """Normalize one SCOR code and identify its entity type and summary."""
    _log_tool_start("identify_scor_code", code=code)
    normalized_code = _normalize_code_input(code)
    canonical = _canonical_code_token(normalized_code)
    entity_type = _get_code_entity_type_map().get(canonical)

    if not normalized_code or not entity_type:
        result = {
            "found": False,
            "input_code": code,
            "normalized_code": _normalize_code_input(code),
            "entity_type": None,
            "entity_type_label": None,
            "summary": {},
        }
        _log_tool_end("identify_scor_code", result)
        return result

    result = {
        "found": True,
        "input_code": code,
        "normalized_code": normalized_code,
        "entity_type": entity_type,
        "entity_type_label": ENTITY_TYPE_LABELS.get(entity_type, entity_type),
        "summary": _get_entity_summary(entity_type, normalized_code),
    }
    _log_tool_end("identify_scor_code", result)
    return result


def identify_scor_codes_in_text(text: str) -> dict[str, Any]:
    """Extract all recognizable SCOR codes from free text."""
    _log_tool_start("identify_scor_codes_in_text", text=text)
    matches = _extract_scor_code_matches(text)
    result = {
        "found": bool(matches),
        "input_text": text,
        "match_count": len(matches),
        "matches": matches,
    }
    _log_tool_end("identify_scor_codes_in_text", result)
    return result


def validate_scor_response(answer_text: str) -> dict[str, Any]:
    """Validate that an answer only references known SCOR codes and process terms."""
    _log_tool_start("validate_scor_response", answer_text=answer_text)
    known_entities = _get_known_scor_entities()
    known_codes = known_entities["codes"]
    allowed_top_level_processes = known_entities["top_level_processes"]

    normalized_found_codes = {
        _normalize_code_input(match.group(0)).upper() for match in SCOR_CODE_PATTERN.finditer(answer_text)
    }
    unknown_codes = sorted(code for code in normalized_found_codes if code not in known_codes)

    unknown_process_terms: list[str] = []
    for match in SLASH_PROCESS_PATTERN.finditer(answer_text):
        terms = [term.lower() for term in match.group(1).split("/")]
        if any(term in allowed_top_level_processes for term in terms):
            unknown_process_terms.extend(
                term for term in terms if term not in allowed_top_level_processes
            )

    unknown_process_terms = sorted(set(unknown_process_terms))
    errors: list[str] = []
    if unknown_codes:
        errors.append(f"Unknown SCOR codes: {', '.join(unknown_codes)}")
    if unknown_process_terms:
        errors.append(
            "Unknown top-level process terms: "
            + ", ".join(term.title() for term in unknown_process_terms)
        )

    result = {
        "valid": not errors,
        "unknown_codes": unknown_codes,
        "unknown_top_level_process_terms": unknown_process_terms,
        "errors": errors,
    }
    _log_tool_end("validate_scor_response", result)
    return result


def get_related_entities(
    entity_type: str,
    entity_code: str,
) -> dict[str, Any]:
    """Return the directly related entities for one process, practice, metric, or skill."""
    _log_tool_start(
        "get_related_entities",
        entity_type=entity_type,
        entity_code=entity_code,
    )
    entity_type_lower = entity_type.lower()
    normalized_entity_code = _normalize_code_input(entity_code)

    if entity_type_lower == "process":
        matches = get_process_details([normalized_entity_code])
        if not matches:
            result = {"entity_type": entity_type, "entity_code": entity_code, "related": {}}
            _log_tool_end("get_related_entities", result)
            return result
        process = matches[0]
        result = {
            "entity_type": "process",
            "entity_code": process["code"],
            "related": {
                "best_practices": process["related_best_practices"],
                "metrics": process["related_metrics"],
                "people_skills": process["related_people_skills"],
                "inputs": process["process_inputs"],
                "outputs": process["process_outputs"],
            },
        }
        _log_tool_end("get_related_entities", result)
        return result

    if entity_type_lower == "practice":
        matches = get_practice_details([normalized_entity_code], [], [], [], [])
        if not matches:
            result = {"entity_type": entity_type, "entity_code": entity_code, "related": {}}
            _log_tool_end("get_related_entities", result)
            return result
        practice = matches[0]
        result = {
            "entity_type": "practice",
            "entity_code": practice["code"],
            "related": {
                "pillars": practice["pillars"],
                "categories": practice["categories"],
                "processes": practice.get("related_processes", []),
                "metrics": practice["related_metrics"],
                "people_skills": practice["related_people_skills"],
            },
        }
        _log_tool_end("get_related_entities", result)
        return result

    if entity_type_lower == "skill":
        matches = get_skill_details([normalized_entity_code], [], [])
        if not matches:
            result = {"entity_type": entity_type, "entity_code": entity_code, "related": {}}
            _log_tool_end("get_related_entities", result)
            return result
        skill = matches[0]
        result = {
            "entity_type": "skill",
            "entity_code": skill["code"],
            "related": {
                "processes": skill.get("related_processes", []),
                "best_practices": skill.get("related_best_practices", []),
                "experiences": skill["related_experiences"],
                "trainings": skill["related_trainings"],
            },
        }
        _log_tool_end("get_related_entities", result)
        return result

    if entity_type_lower == "metric":
        hierarchy = get_metric_hierarchy(normalized_entity_code)
        if not hierarchy.get("found", True):
            result = {"entity_type": entity_type, "entity_code": entity_code, "related": {}}
            _log_tool_end("get_related_entities", result)
            return result
        matches = get_metric_details([normalized_entity_code], [], [], [], [], [])
        metric = next(
            (item for item in matches if item["code"].upper() == normalized_entity_code.upper()),
            None,
        )
        if metric is None:
            result = {"entity_type": entity_type, "entity_code": entity_code, "related": {}}
            _log_tool_end("get_related_entities", result)
            return result
        result = {
            "entity_type": "metric",
            "entity_code": metric["code"],
            "related": {
                "category": metric["category"],
                "attribute": metric["attribute"],
                "metric_level": hierarchy["metric_level"],
                "upper_metrics": hierarchy["upper_metrics"],
                "child_metrics": hierarchy["child_metrics"],
                "level1_metric": metric["level1_metric"],
                "level2_metric": metric["level2_metric"],
                "level3_metric": metric["level3_metric"],
            },
        }
        _log_tool_end("get_related_entities", result)
        return result

    raise ValueError(f"Unsupported entity_type: {entity_type}")


# Entity detail assembly and natural-language search
def _build_entity_details_result(input_code: str, query_text: str = "") -> dict[str, Any]:
    """Assemble the full structured detail payload for one SCOR entity code."""
    identified = identify_scor_code(input_code)
    if not identified["found"]:
        return identified

    normalized_code = identified["normalized_code"]
    entity_type = identified["entity_type"]
    entity_type_label = identified["entity_type_label"]
    query_filters = _build_relation_query_filters(query_text)

    result: dict[str, Any] = {
        "found": True,
        "input_code": input_code,
        "normalized_code": normalized_code,
        "entity_type": entity_type,
        "entity_type_label": entity_type_label,
        "summary": identified["summary"],
    }

    if entity_type == "processes":
        process_matches = get_process_details([normalized_code])
        exact_process = next(
            (
                item
                for item in process_matches
                if item["code"].upper() == normalized_code.upper()
            ),
            None,
        )
        result["details"] = _compact_process(exact_process) if exact_process else {}
        if exact_process:
            result["_match_definition"] = exact_process.get("definition") or ""
        hierarchy = get_process_hierarchy(normalized_code)
        result["hierarchy"] = {
            "found": hierarchy.get("found", False),
            "process": _compact_process(hierarchy["process"]) if hierarchy.get("process") else {},
            "child_processes": [
                _compact_process(item)
                for item in hierarchy.get("child_processes", [])
            ],
            "has_child_processes": hierarchy.get("has_child_processes", False),
        }
        related = get_related_entities("process", normalized_code)["related"]
        related_best_practice_details = [
            _compact_practice(practice)
            for practice in _get_practices_by_exact_codes(related.get("best_practices", []))
        ]
        related_metric_details = [
            _compact_metric(metric)
            for metric in _get_metrics_by_exact_codes(related.get("metrics", []))
        ]
        related_skill_details = [
            _compact_skill(skill)
            for skill in _get_skills_by_exact_codes(related.get("people_skills", []))
        ]
        filtered_best_practice_details = _filter_practice_details(
            related_best_practice_details,
            query_filters,
        )
        filtered_metric_details = _filter_metric_details(
            related_metric_details,
            query_filters,
        )
        result["related"] = {
            "best_practices": filtered_best_practice_details,
            "metrics": filtered_metric_details,
            "people_skills": related_skill_details,
            "inputs": related.get("inputs", []),
            "outputs": related.get("outputs", []),
            "total_best_practice_count": len(related_best_practice_details),
            "returned_best_practice_count": len(filtered_best_practice_details),
            "total_metric_count": len(related_metric_details),
            "returned_metric_count": len(filtered_metric_details),
            "applied_filters": {
                "metric_attributes": query_filters["metric_attributes"],
                "metric_categories": query_filters["metric_categories"],
                "practice_pillars": query_filters["practice_pillars"],
                "practice_categories": query_filters["practice_categories"],
                "focus_relations": sorted(query_filters["focus_relations"]),
            },
        }
    elif entity_type == "practices":
        practice_matches = get_practice_details([normalized_code], [], [], [], [])
        result["details"] = _compact_practice(practice_matches[0]) if practice_matches else {}
        if practice_matches:
            result["_match_definition"] = practice_matches[0].get("definition") or ""
        related = get_related_entities("practice", normalized_code)["related"]
        related_metric_details = [
            _compact_metric(metric)
            for metric in _get_metrics_by_exact_codes(related.get("metrics", []))
        ]
        related_skill_details = [
            _compact_skill(skill)
            for skill in _get_skills_by_exact_codes(related.get("people_skills", []))
        ]
        filtered_metric_details = _filter_metric_details(
            related_metric_details,
            query_filters,
        )
        result["related"] = {
            "pillars": related.get("pillars", []),
            "categories": related.get("categories", []),
            "processes": [
                _compact_process(process)
                for process in _get_processes_by_exact_codes(related.get("processes", []))
            ],
            "metrics": filtered_metric_details,
            "people_skills": related_skill_details,
            "total_metric_count": len(related_metric_details),
            "returned_metric_count": len(filtered_metric_details),
            "applied_filters": {
                "metric_attributes": query_filters["metric_attributes"],
                "metric_categories": query_filters["metric_categories"],
                "practice_pillars": query_filters["practice_pillars"],
                "practice_categories": query_filters["practice_categories"],
                "focus_relations": sorted(query_filters["focus_relations"]),
            },
        }
    elif entity_type == "metrics":
        metric_matches = get_metric_details([normalized_code], [], [], [], [], [])
        exact_metric = next(
            (
                item
                for item in metric_matches
                if item["code"].upper() == normalized_code.upper()
            ),
            None,
        )
        result["details"] = _compact_metric(exact_metric) if exact_metric else {}
        if exact_metric:
            result["_match_definition"] = exact_metric.get("definition") or ""
        hierarchy = get_metric_hierarchy(normalized_code)
        result["hierarchy"] = {
            "metric_code": hierarchy.get("metric_code", normalized_code),
            "metric_name": hierarchy.get("metric_name", ""),
            "metric_level": hierarchy.get("metric_level"),
            "upper_metrics": hierarchy.get("upper_metrics", []),
            "child_metrics": hierarchy.get("child_metrics", []),
        }
        improvement_options = get_metric_improvement_options(normalized_code)
        result["improvement_options"] = {
            "found": improvement_options.get("found", False),
            "metric": improvement_options.get("metric", {}),
            "metric_scope_codes": improvement_options.get("metric_scope_codes", []),
            "related_best_practices": improvement_options.get("related_best_practices", []),
            "related_processes": improvement_options.get("related_processes", []),
            "related_people_skills": improvement_options.get("related_people_skills", []),
        }
        related = get_related_entities("metric", normalized_code)["related"]
        result["related"] = {
            "category": related.get("category", ""),
            "attribute": related.get("attribute", ""),
            "metric_level": related.get("metric_level"),
            "upper_metrics": related.get("upper_metrics", []),
            "child_metrics": related.get("child_metrics", []),
            "level1_metric": related.get("level1_metric", ""),
            "level2_metric": related.get("level2_metric", ""),
            "level3_metric": related.get("level3_metric", ""),
        }
    elif entity_type == "skills":
        skill_matches = get_skill_details([normalized_code], [], [])
        result["details"] = _compact_skill(skill_matches[0]) if skill_matches else {}
        if skill_matches:
            result["_match_definition"] = skill_matches[0].get("definition") or ""
        related = get_related_entities("skill", normalized_code)["related"]
        result["related"] = {
            "best_practices": [
                _compact_practice(practice)
                for practice in _get_practices_by_exact_codes(related.get("best_practices", []))
            ],
            "processes": [
                _compact_process(process)
                for process in _get_processes_by_exact_codes(related.get("processes", []))
            ],
            "experiences": related.get("experiences", []),
            "trainings": related.get("trainings", []),
        }

    return result


def _annotate_match_metadata(
    result: dict[str, Any],
    *,
    match_type: str,
    match_score: int | None = None,
    matched_entity_type: str | None = None,
    matched_name: str | None = None,
) -> dict[str, Any]:
    """Attach search-match metadata to a result payload."""
    result["match_type"] = match_type
    if match_score is not None:
        result["match_score"] = match_score
    if matched_entity_type is not None:
        result["matched_entity_type"] = matched_entity_type
    if matched_name is not None:
        result["matched_name"] = matched_name
    return result


def _bm25_process_scores(
    query_text: str,
    processes: tuple[dict[str, Any], ...],
    process_by_code: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Rank processes generically from query terms and process/parent definitions."""
    query_segments = [
        _tokenize_text(segment)
        for segment in re.split(
            r"[.!?;]+|\b(?:but|while|however|although|yet|whereas)\b",
            query_text,
            flags=re.IGNORECASE,
        )
        if _tokenize_text(segment)
    ]
    full_query_tokens = _tokenize_text(query_text)
    if not full_query_tokens or not processes:
        return {}

    # Expand only terms that occur exactly in the incident text, using the SCOR
    # dictionary itself. This supplies domain meaning (for example, what an
    # operational term entails) without scenario-specific synonym hardcoding.
    dictionary_context_tokens: list[str] = []
    for match in _search_dictionary_terms(query_text, 8):
        if match.get("match_type") != "exact_term":
            continue
        dictionary_context_tokens.extend(
            _tokenize_text(
                f"{match.get('word', '')} {match.get('definition', '')}"
            )
        )
    if dictionary_context_tokens:
        query_segments.append(dictionary_context_tokens)

    documents: dict[str, list[str]] = {}
    document_frequency: Counter[str] = Counter()
    for process in processes:
        parent_code = process["code"].rsplit(".", 1)[0] if "." in process["code"] else ""
        parent = process_by_code.get(parent_code, {})
        tokens = _tokenize_text(
            " ".join(
                filter(
                    None,
                    [
                        process.get("name", ""),
                        process.get("definition", ""),
                        parent.get("name", ""),
                        parent.get("definition", ""),
                    ],
                )
            )
        )
        documents[process["code"]] = tokens
        document_frequency.update(set(tokens))

    average_length = sum(len(tokens) for tokens in documents.values()) / max(
        len(documents), 1
    )
    # Long incident descriptions contain broad symptoms and explicit exclusions.
    # Score their evidence clauses independently so one long bag of words does not
    # overwhelm the clause that describes the directly failing activity.
    query_token_groups = query_segments if len(query_segments) > 1 else [full_query_tokens]
    document_count = len(documents)
    k1 = 1.5
    b = 0.75
    raw_scores_by_group: list[dict[str, float]] = [
        {} for _ in query_token_groups
    ]
    for code, tokens in documents.items():
        if not tokens:
            continue
        term_counts = Counter(tokens)
        for group_index, query_tokens in enumerate(query_token_groups):
            score = 0.0
            for token, query_frequency in Counter(query_tokens).items():
                frequency = term_counts.get(token, 0)
                if not frequency:
                    continue
                frequency_in_documents = document_frequency[token]
                inverse_document_frequency = math.log(
                    1
                    + (document_count - frequency_in_documents + 0.5)
                    / (frequency_in_documents + 0.5)
                )
                length_normalization = frequency + k1 * (
                    1 - b + b * len(tokens) / max(average_length, 1)
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (k1 + 1)
                    / length_normalization
                    * min(query_frequency, 2)
                )
            if score:
                raw_scores_by_group[group_index][code] = score

    # Preserve candidates supported by any individual evidence clause. This is
    # generic reciprocal-rank fusion: it prevents a long incident's vocabulary-
    # rich background clauses from crowding out the process named by a shorter,
    # more diagnostic clause. The final LLM pass still weighs confirmations,
    # exclusions, and ambiguity; retrieval only supplies a diverse candidate set.
    maximum_scores: dict[str, float] = {}
    reciprocal_rank_scores: dict[str, float] = {}
    for group_scores in raw_scores_by_group:
        ranked_group = sorted(
            group_scores.items(), key=lambda item: (-item[1], item[0])
        )
        for rank, (code, score) in enumerate(ranked_group, start=1):
            maximum_scores[code] = max(maximum_scores.get(code, 0.0), score)
            reciprocal_rank_scores[code] = reciprocal_rank_scores.get(code, 0.0) + (
                1.0 / (20 + rank)
            )

    scores: dict[str, int] = {}
    for code, score in maximum_scores.items():
        # The fused component has enough influence to diversify the shortlist,
        # while raw BM25 relevance remains the main ordering signal.
        scores[code] = int(
            round(score * 10 + reciprocal_rank_scores.get(code, 0.0) * 400)
        )
    return scores


def _search_scor_entities(
    query_text: str,
    entity_types: list[str],
    limit: int,
    include_alternatives: bool = False,
) -> list[dict[str, Any]]:
    """Search SCOR entities by exact name, fuzzy name, or text relevance."""
    requested_types = {item.lower() for item in entity_types} if entity_types else {"process", "practice", "metric", "skill"}
    if not _tokenize_text(query_text):
        return []

    exact_matches = _find_exact_name_matches(query_text, requested_types)
    if exact_matches and not include_alternatives:
        top_matches = exact_matches[:limit]
        results = []
        for score, code, entity_type, name in top_matches:
            detailed_result = _build_entity_details_result(code, query_text=query_text)
            results.append(
                _annotate_match_metadata(
                    detailed_result,
                    match_type="exact_name",
                    match_score=score,
                    matched_entity_type=entity_type,
                    matched_name=name,
                )
            )
        return results

    fuzzy_name_matches = _find_fuzzy_name_matches(query_text, requested_types)
    if fuzzy_name_matches and not include_alternatives:
        top_matches = fuzzy_name_matches[:limit]
        results = []
        for score, code, entity_type, name in top_matches:
            detailed_result = _build_entity_details_result(code, query_text=query_text)
            results.append(
                _annotate_match_metadata(
                    detailed_result,
                    match_type="fuzzy_name",
                    match_score=score,
                    matched_entity_type=entity_type,
                    matched_name=name,
                )
            )
        return results

    candidates: list[tuple[int, str, str, str]] = []

    higher_level_process_query = bool(
        "process" in requested_types
        and HIGHER_LEVEL_PROCESS_QUERY_PATTERN.search(query_text)
    )
    default_to_level_three_processes = bool(
        "process" in requested_types and not higher_level_process_query
    )

    if "process" in requested_types:
        processes = _get_processes()
        process_by_code = {process["code"]: process for process in processes}
        bm25_scores = _bm25_process_scores(query_text, processes, process_by_code)
        for process in processes:
            is_level_three = bool(LEVEL_THREE_PROCESS_CODE_PATTERN.fullmatch(process["code"]))
            if default_to_level_three_processes and not is_level_three:
                continue
            if higher_level_process_query and is_level_three:
                continue
            parent_code = process["code"].rsplit(".", 1)[0] if "." in process["code"] else ""
            parent = process_by_code.get(parent_code, {})
            score = _score_text_match(
                query_text,
                " ".join(
                    filter(
                        None,
                        [
                            process["code"],
                            process["name"],
                            parent.get("code", ""),
                            parent.get("name", ""),
                        ],
                    )
                ),
                " ".join(
                    filter(
                        None,
                        [process["definition"] or "", parent.get("definition", "")],
                    )
                ),
            )
            score += bm25_scores.get(process["code"], 0)
            if score:
                candidates.append((score, process["code"], "process", process["name"]))

    if "practice" in requested_types:
        for practice in [_normalize_practice(item) for item in _load_json("practices.json")]:
            score = _score_text_match(
                query_text,
                f"{practice['code']} {practice['name']}",
                practice["definition"],
            )
            if score:
                candidates.append((score, practice["code"], "practice", practice["name"]))

    if "metric" in requested_types:
        for metric in _get_metrics():
            score = _score_text_match(
                query_text,
                f"{metric['code']} {metric['name']}",
                " ".join(filter(None, [metric["definition"], metric["category"], metric["attribute"]])),
            )
            if score:
                candidates.append((score, metric["code"], "metric", metric["name"]))

    if "skill" in requested_types:
        for skill in [_normalize_skill(item) for item in _load_json("skills.json")]:
            score = _score_text_match(
                query_text,
                f"{skill['code']} {skill['name']}",
                skill["definition"],
            )
            if score:
                candidates.append((score, skill["code"], "skill", skill["name"]))

    ranked_matches: list[tuple[int, str, str, str, str]] = []
    ranked_matches.extend(
        (score, code, entity_type, name, "exact_name")
        for score, code, entity_type, name in exact_matches
    )
    ranked_matches.extend(
        (score, code, entity_type, name, "fuzzy_name")
        for score, code, entity_type, name in fuzzy_name_matches
        if not any(
            exact_code.upper() == code.upper() and exact_type == entity_type
            for _, exact_code, exact_type, _ in exact_matches
        )
    )
    ranked_matches.extend(
        (score, code, entity_type, name, "text_search")
        for score, code, entity_type, name in sorted(
            candidates,
            key=lambda item: (-item[0], item[1]),
        )
    )
    top_matches: list[tuple[int, str, str, str, str]] = []
    seen_matches: set[tuple[str, str]] = set()
    for match in ranked_matches:
        _, code, entity_type, _, _ = match
        key = (entity_type, code.upper())
        if key in seen_matches:
            continue
        seen_matches.add(key)
        top_matches.append(match)
        if len(top_matches) >= limit:
            break
    results = []
    for score, code, entity_type, name, match_type in top_matches:
        detailed_result = _build_entity_details_result(code, query_text=query_text)
        results.append(
            _annotate_match_metadata(
                detailed_result,
                match_type=match_type,
                match_score=score,
                matched_entity_type=entity_type,
                matched_name=name,
            )
        )
    return results


def get_scor_entity_details(
    entity_codes: list[str] | None = None,
    query_text: str = "",
    entity_types: list[str] | None = None,
    limit: int = 5,
    include_alternatives: bool = False,
) -> dict[str, Any]:
    """Return detailed SCOR entity data for codes or a natural-language query."""
    _log_tool_start(
        "get_scor_entity_details",
        entity_codes=entity_codes or [],
        query_text=query_text,
        entity_types=entity_types or [],
        limit=limit,
    )
    results: list[dict[str, Any]] = []
    dictionary_matches: list[dict[str, Any]] = []

    for input_code in entity_codes or []:
        results.append(
            _annotate_match_metadata(
                _build_entity_details_result(input_code, query_text=query_text),
                match_type="direct_code",
            )
        )

    if query_text.strip():
        dictionary_matches = _search_dictionary_terms(query_text, limit)
        if not entity_codes:
            results.extend(
                _search_scor_entities(
                    query_text,
                    entity_types or [],
                    limit,
                    include_alternatives=include_alternatives,
                )
            )

    response = {
        "requested_entity_codes": entity_codes or [],
        "query_text": query_text,
        "requested_entity_types": entity_types or [],
        "result_count": len(results),
        "results": results,
        "dictionary_match_count": len(dictionary_matches),
        "dictionary_matches": dictionary_matches,
    }
    _log_tool_end("get_scor_entity_details", response)
    return response


# Internal function registry used by the orchestrator and tests
available_functions = {
    "search_supply_chain_knowledge": search_supply_chain_knowledge,
    "get_scor_entity_details": get_scor_entity_details,
    "identify_scor_code": identify_scor_code,
    "identify_scor_codes_in_text": identify_scor_codes_in_text,
    "get_process_details": get_process_details,
    "get_process_hierarchy": get_process_hierarchy,
    "get_practice_details": get_practice_details,
    "get_metric_details": get_metric_details,
    "get_metric_hierarchy": get_metric_hierarchy,
    "get_metric_improvement_options": get_metric_improvement_options,
    "get_skill_details": get_skill_details,
    "get_related_entities": get_related_entities,
    "validate_scor_response": validate_scor_response,
}


tools = [
    SUPPLY_CHAIN_KNOWLEDGE_TOOL,
    {
        "type": "function",
        "name": "get_scor_entity_details",
        "description": "Returns detailed SCOR entity information for one or more SCOR codes. The tool automatically identifies each code's entity type and returns the relevant details, hierarchy, and related information.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entity_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more SCOR codes to inspect, such as process, metric, practice, or skill codes.",
                },
                "query_text": {
                    "type": "string",
                    "description": "A natural-language description to search against SCOR entities when the user does not know the code.",
                },
                "entity_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["process", "practice", "metric", "skill"],
                    },
                    "description": "Optional entity-type filter for natural-language search.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of natural-language search matches to return.",
                }
            },
            "required": ["entity_codes", "query_text", "entity_types", "limit"],
            "additionalProperties": False,
        },
    },
]
