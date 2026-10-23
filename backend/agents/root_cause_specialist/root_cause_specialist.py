import json
import os
import re
from itertools import count
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from ...config import models
from ...model_provider import get_model_client
from ...tools import available_functions
from ...utils.logging import log_conversation_start, log_with_prefix
from ..scor_agent_tools import (
    BASE_SCOR_AGENT_TOOLS,
    build_detected_code_source_context,
    run_base_tool,
    summarize_base_tool_result,
)
from .relationship_context import (
    _select_context_paths,
    format_relationship_context,
    resolve_relationship_search,
)
from .root_cause_specialist_system_prompt import ROOT_CAUSE_SPECIALIST_SYSTEM_PROMPT
from .scor_path_retrieval import (
    ENTITY_SCOPE_PATTERNS,
    EXPLICIT_SCOR_ENTITY_CODE_PATTERN,
    LEVEL_THREE_PROCESS_PATTERN,
)


load_dotenv()

MODEL_NAME = models["root_cause"]
MAX_TOOL_ROUNDS = 6
MAX_MAPPING_REPAIR_ROUNDS = 2
CONVERSATION_COUNTER = count(1)
ROOT_CAUSE_AGENT_TOOLS = BASE_SCOR_AGENT_TOOLS


def _strip_internal_repair_wording(answer: str) -> str:
    """Remove repair-instruction residue without asking another LLM to rewrite."""
    cleaned = re.sub(
        r"(?i)\s*(?:and|using|based on)\s+(?:only\s+)?the\s+"
        r"(?:allowed|provided|specified|approved)\s+(?:direct\s+)?codes\s+"
        r"(?:you\s+)?(?:specified|provided|allowed|approved)?",
        "",
        answer,
    )
    cleaned = re.sub(
        r"(?im)^.*\b(?:internal (?:contract|validation)|repair instruction|"
        r"retrieval contract|allowed[_ ]direct[_ ]codes)\b.*$\n?",
        "",
        cleaned,
    )
    return cleaned.strip()


def _build_prompt_parts(
    question: str,
    user_context: str,
) -> tuple[str, dict[str, Any] | None]:
    prompt = f"Analysis request:\n{question.strip()}"
    _, exact_code_context = build_detected_code_source_context(question)
    relationship_result = resolve_relationship_search(question)
    relationship_context = (
        format_relationship_context(relationship_result)
        if relationship_result
        else ""
    )
    if relationship_context:
        prompt = f"{relationship_context}\n\n{prompt}"
    if exact_code_context:
        prompt = f"{exact_code_context}\n\n{prompt}"
    if user_context.strip():
        prompt = f"Relevant recent context:\n{user_context.strip()}\n\n{prompt}"
    return prompt, relationship_result


def _build_prompt(question: str, user_context: str) -> str:
    prompt, _ = _build_prompt_parts(question, user_context)
    return prompt


def _entity_request_is_singular(question: str, entity_type: str) -> bool:
    pattern = ENTITY_SCOPE_PATTERNS.get(entity_type)
    if pattern is None:
        return False
    matches = [match.group(0).casefold() for match in pattern.finditer(question)]
    if not matches:
        return False
    plural_terms = {
        "process": ("processes", "process elements"),
        "practice": ("practices",),
        "metric": ("metrics", "measures"),
        "skill": ("skills",),
    }
    return not any(
        any(term in matched_text for term in plural_terms[entity_type])
        for matched_text in matches
    )


def _requested_entity_minimum_count(question: str, entity_type: str) -> int:
    if _entity_request_is_singular(question, entity_type):
        return 1
    pattern = ENTITY_SCOPE_PATTERNS.get(entity_type)
    if pattern is None:
        return 1
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    requested_count = 2
    for match in pattern.finditer(question):
        prefix = question[max(0, match.start() - 80) : match.start()].casefold()
        numeric_matches = list(
            re.finditer(r"\b(?:[1-5]|one|two|three|four|five)\b", prefix)
        )
        if not numeric_matches:
            continue
        value = numeric_matches[-1].group(0)
        requested_count = max(
            requested_count,
            int(value) if value.isdigit() else number_words[value],
        )
    return requested_count


def _dominant_practice_mapping_contract(
    question: str,
    relationship_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not relationship_result or "practice" not in relationship_result.get(
        "requested_types", []
    ):
        return None
    selected_paths = _select_context_paths(relationship_result)
    selected_practice_codes = {
        path.get("entities", {}).get("practice", {}).get("code", "")
        for path in selected_paths
    } - {""}
    if len(selected_practice_codes) != 1:
        return None

    practice_code = next(iter(selected_practice_codes))
    practice_candidate = next(
        (
            candidate
            for candidate in relationship_result.get("candidate_groups", {}).get(
                "practice", []
            )
            if candidate.get("normalized_code") == practice_code
        ),
        None,
    )
    if not practice_candidate:
        return None

    relation_specs = {
        "process": ("processes", "supports_processes"),
        "metric": ("metrics", "impacts_metrics"),
        "skill": ("people_skills", "required_skills"),
    }
    requested_types = set(relationship_result.get("requested_types", []))
    allowed_by_type: dict[str, list[str]] = {}
    preferred_by_type: dict[str, list[str]] = {}
    related = practice_candidate.get("related", {})
    for entity_type, (related_key, relation_name) in relation_specs.items():
        if entity_type not in requested_types:
            continue
        allowed_by_type[entity_type] = [
            str(item.get("code", ""))
            for item in related.get(related_key, [])
            if item.get("code")
        ]
        preferred_codes: list[str] = []
        for path in relationship_result.get("paths", []):
            entities = path.get("entities", {})
            if entities.get("practice", {}).get("code") != practice_code:
                continue
            entity_code = entities.get(entity_type, {}).get("code", "")
            if not entity_code or entity_code in preferred_codes:
                continue
            if any(
                relation.get("from") == practice_code
                and relation.get("relation") == relation_name
                and relation.get("to") == entity_code
                for relation in path.get("verified_relationships", [])
            ):
                preferred_codes.append(entity_code)
        preferred_by_type[entity_type] = preferred_codes

    return {
        "practice": practice_code,
        "practice_entity": {
            "code": practice_code,
            "name": practice_candidate.get("details", {}).get("name", ""),
            "definition": practice_candidate.get("_match_definition")
            or practice_candidate.get("details", {}).get("definition", ""),
        },
        "allowed_by_type": allowed_by_type,
        "preferred_by_type": preferred_by_type,
        "entities_by_type": {
            entity_type: {
                str(item.get("code", "")): {
                    "code": str(item.get("code", "")),
                    "name": str(item.get("name", "")),
                    "definition": str(item.get("definition", "")),
                }
                for item in related.get(relation_specs[entity_type][0], [])
                if item.get("code")
            }
            for entity_type in allowed_by_type
        },
        "singular_by_type": {
            entity_type: _entity_request_is_singular(question, entity_type)
            for entity_type in allowed_by_type
        },
        "minimum_count_by_type": {
            entity_type: _requested_entity_minimum_count(question, entity_type)
            for entity_type in allowed_by_type
        },
    }


def _process_mapping_contract(
    question: str,
    relationship_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a final-answer contract around a dominant process candidate."""
    if not relationship_result or "process" not in relationship_result.get(
        "requested_types", []
    ):
        return None
    process_candidates = relationship_result.get("candidate_groups", {}).get(
        "process", []
    )
    paths = [
        path
        for path in relationship_result.get("paths", [])
        if path.get("entities", {}).get("process", {}).get("code")
    ]
    if not paths:
        # A process-only diagnostic returns entity_candidates rather than relation
        # paths. Build the same minimal ranking shape so ambiguity and primary-
        # process validation still apply.
        paths = [
            {
                "anchor_score": int(candidate.get("match_score") or 0),
                "score": int(candidate.get("match_score") or 0),
                "entities": {
                    "process": {
                        "code": str(candidate.get("normalized_code", "")),
                        "name": str(candidate.get("details", {}).get("name", "")),
                    }
                },
            }
            for candidate in process_candidates
            if candidate.get("normalized_code")
        ]
    if not paths:
        return None

    best_score_by_code: dict[str, int] = {}
    for path in paths:
        code = str(path["entities"]["process"]["code"])
        score = int(path.get("anchor_score", path.get("score", 0)))
        best_score_by_code[code] = max(score, best_score_by_code.get(code, score))
    ranked_codes = sorted(
        best_score_by_code,
        key=lambda code: (-best_score_by_code[code], code),
    )
    top_code = ranked_codes[0]
    top_score = best_score_by_code[top_code]
    runner_up_score = best_score_by_code[ranked_codes[1]] if len(ranked_codes) > 1 else 0
    process_name_by_code = {
        str(candidate.get("normalized_code", "")): str(
            candidate.get("details", {}).get("name", "")
        ).strip().casefold()
        for candidate in process_candidates
        if candidate.get("normalized_code")
    }
    top_name = process_name_by_code.get(top_code, "")
    same_activity_codes = [
        code
        for code in ranked_codes
        if top_name and process_name_by_code.get(code) == top_name
    ]
    explicit_process_code = any(
        available_functions["identify_scor_code"](match.group(0)).get("entity_type")
        == "processes"
        for match in EXPLICIT_SCOR_ENTITY_CODE_PATTERN.finditer(question)
    )
    branch_ambiguous = len(same_activity_codes) > 1 and not explicit_process_code
    dominant = len(ranked_codes) == 1 or (
        top_score - runner_up_score >= 10
        and top_score >= 1.15 * max(runner_up_score, 1)
    )
    if not dominant or branch_ambiguous:
        ambiguity_candidates = (
            same_activity_codes if branch_ambiguous else ranked_codes[:4]
        )
        candidate_branches: list[dict[str, str]] = []
        seen_parent_codes: set[str] = set()
        # Parent branches are meaningful user choices only when the retrieved
        # candidates represent the same named activity under different parents.
        # For general low-confidence retrieval, exposing the top lexical parents
        # would turn noisy candidates into a misleading multiple-choice question.
        if branch_ambiguous:
            for code in ambiguity_candidates[:4]:
                candidate = next(
                    (
                        item
                        for item in process_candidates
                        if str(item.get("normalized_code", "")) == code
                    ),
                    {},
                )
                parent = candidate.get("details", {}).get("parent_process", {})
                parent_code = str(parent.get("code", "")) if isinstance(parent, dict) else ""
                if not parent_code or parent_code in seen_parent_codes:
                    continue
                seen_parent_codes.add(parent_code)
                candidate_branches.append(
                    {
                        "process_code": code,
                        "parent_code": parent_code,
                        "parent_name": str(parent.get("name", "")),
                        "parent_definition": str(parent.get("definition", "")),
                    }
                )
        return {
            "status": "ambiguous",
            "candidate_process_codes": ambiguity_candidates[:4],
            "candidate_branches": candidate_branches,
            "requested_types": list(
                relationship_result.get("requested_types", [])
            ),
            "candidate_related_codes": {
                str(candidate.get("normalized_code", "")): {
                    entity_type: [
                        str(item.get("code", ""))
                        for item in candidate.get("related", {}).get(related_key, [])
                        if item.get("code")
                    ]
                    for entity_type, related_key in {
                        "metric": "metrics",
                        "practice": "best_practices",
                        "skill": "people_skills",
                    }.items()
                }
                for candidate in process_candidates
                if candidate.get("normalized_code")
            },
            "candidate_processes": {
                str(candidate.get("normalized_code", "")): {
                    "code": str(candidate.get("normalized_code", "")),
                    "name": str(candidate.get("details", {}).get("name", "")),
                    "parent_code": str(
                        candidate.get("details", {})
                        .get("parent_process", {})
                        .get("code", "")
                    ),
                    "parent_name": str(
                        candidate.get("details", {})
                        .get("parent_process", {})
                        .get("name", "")
                    ),
                }
                for candidate in process_candidates
                if candidate.get("normalized_code")
            },
        }

    process_candidate = next(
        (
            candidate
            for candidate in process_candidates
            if candidate.get("normalized_code") == top_code
        ),
        None,
    )
    if not process_candidate:
        return None

    related = process_candidate.get("related", {})
    relation_keys = {
        "metric": "metrics",
        "practice": "best_practices",
        "skill": "people_skills",
    }
    requested_types = set(relationship_result.get("requested_types", []))
    allowed_by_type = {
        entity_type: [
            str(item.get("code", ""))
            for item in related.get(related_key, [])
            if item.get("code")
        ]
        for entity_type, related_key in relation_keys.items()
        if entity_type in requested_types
    }
    entities_by_type = {
        entity_type: {
            str(item.get("code", "")).upper(): {
                "code": str(item.get("code", "")),
                "name": str(item.get("name", "")),
            }
            for item in related.get(relation_keys[entity_type], [])
            if item.get("code")
        }
        for entity_type in allowed_by_type
    }
    preferred_by_type: dict[str, list[str]] = {
        entity_type: [] for entity_type in allowed_by_type
    }
    for path in paths:
        entities = path.get("entities", {})
        if entities.get("process", {}).get("code") != top_code:
            continue
        for entity_type in preferred_by_type:
            code = str(entities.get(entity_type, {}).get("code", ""))
            if code and code not in preferred_by_type[entity_type]:
                preferred_by_type[entity_type].append(code)

    return {
        "status": "dominant",
        "process": top_code,
        "process_entity": {
            "code": top_code,
            "name": process_candidate.get("details", {}).get("name", ""),
            "definition": process_candidate.get("_match_definition")
            or process_candidate.get("details", {}).get("definition", ""),
        },
        "allowed_by_type": allowed_by_type,
        "preferred_by_type": preferred_by_type,
        "entities_by_type": entities_by_type,
    }


def _process_consistency_feedback(
    answer: str,
    contract: dict[str, Any] | None,
) -> str:
    if not contract:
        return ""
    if contract.get("status") == "ambiguous":
        primary_line = next(
            (
                line
                for line in answer.splitlines()
                if re.search(
                    r"\b(?:primary|root[- ]cause|ana|kök neden)\b",
                    line,
                    re.IGNORECASE,
                )
                and EXPLICIT_SCOR_ENTITY_CODE_PATTERN.search(line)
            ),
            "",
        )
        primary_match = EXPLICIT_SCOR_ENTITY_CODE_PATTERN.search(primary_line)
        selected_process = primary_match.group(0).upper() if primary_match else ""
        candidate_processes = {
            str(code).upper()
            for code in contract.get("candidate_process_codes", [])
        }
        if not selected_process:
            if "?" in answer:
                return ""
            return (
                "The evidence does not support a broad process-family assertion. Either ask one "
                "useful clarification question or select one exact Level-3 process from the "
                f"retrieved candidates: {', '.join(sorted(candidate_processes))}."
            )
        if not LEVEL_THREE_PROCESS_PATTERN.fullmatch(selected_process):
            return (
                f"{selected_process} is too broad for the requested root-cause activity. Select "
                "one exact Level-3 candidate or ask a useful clarification question: "
                + ", ".join(sorted(candidate_processes))
                + "."
            )
        if selected_process not in candidate_processes:
            return (
                f"The selected primary process {selected_process} is not supported by the "
                "retrieved incident evidence. Select one of these candidates or ask a useful "
                f"clarification question: {', '.join(sorted(candidate_processes))}."
            )

        typed_codes: dict[str, set[str]] = {
            "metric": set(), "practice": set(), "skill": set()
        }
        for match in EXPLICIT_SCOR_ENTITY_CODE_PATTERN.finditer(answer):
            code = match.group(0).upper()
            identified = available_functions["identify_scor_code"](code)
            entity_type = {
                "metrics": "metric",
                "practices": "practice",
                "skills": "skill",
            }.get(str(identified.get("entity_type", "")))
            if entity_type:
                typed_codes[entity_type].add(code)
        allowed_by_type = contract.get("candidate_related_codes", {}).get(
            selected_process, {}
        )
        issues: list[str] = []
        for entity_type in contract.get("requested_types", []):
            if entity_type not in typed_codes:
                continue
            allowed = {
                str(code).upper()
                for code in allowed_by_type.get(entity_type, [])
            }
            unsupported = sorted(typed_codes[entity_type] - allowed)
            if unsupported:
                issues.append(
                    f"unsupported {entity_type} codes for {selected_process}: "
                    + ", ".join(unsupported)
                )
            if allowed and not typed_codes[entity_type].intersection(allowed):
                issues.append(
                    f"no directly related {entity_type} code for {selected_process}"
                )
        if issues:
            return (
                "Keep the Level-3 diagnosis, but replace the recommendations with only direct "
                f"relationships of {selected_process}: "
                + "; ".join(issues)
                + ". Allowed direct codes: "
                + json.dumps(allowed_by_type, ensure_ascii=True)
                + "."
            )
        return ""

    expected_process = str(contract["process"]).upper()
    typed_codes: dict[str, set[str]] = {
        "process": set(), "metric": set(), "practice": set(), "skill": set()
    }
    for match in EXPLICIT_SCOR_ENTITY_CODE_PATTERN.finditer(answer):
        code = match.group(0).upper()
        identified = available_functions["identify_scor_code"](code)
        entity_type = str(identified.get("entity_type", ""))
        normalized_type = {
            "processes": "process",
            "metrics": "metric",
            "practices": "practice",
            "skills": "skill",
        }.get(entity_type)
        if normalized_type:
            typed_codes[normalized_type].add(code)

    primary_line = next(
        (
            line
            for line in answer.splitlines()
            if re.search(r"\b(?:primary|root[- ]cause|ana|kök neden)\b", line, re.IGNORECASE)
            and EXPLICIT_SCOR_ENTITY_CODE_PATTERN.search(line)
        ),
        "",
    )
    primary_match = EXPLICIT_SCOR_ENTITY_CODE_PATTERN.search(primary_line)
    selected_primary = primary_match.group(0).upper() if primary_match else ""
    issues: list[str] = []
    if selected_primary != expected_process:
        issues.append(f"the primary process must be {expected_process}")

    for entity_type, allowed_codes in contract["allowed_by_type"].items():
        allowed = {str(code).upper() for code in allowed_codes}
        unsupported = sorted(typed_codes[entity_type] - allowed)
        if unsupported:
            issues.append(
                f"{entity_type} codes outside {expected_process}'s direct relationships: "
                + ", ".join(unsupported)
            )
        if allowed and not typed_codes[entity_type].intersection(allowed):
            issues.append(f"no direct {entity_type} code for {expected_process} was returned")
    if not issues:
        return ""
    return (
        "Revise the draft to satisfy the dominant-process contract: "
        + "; ".join(issues)
        + ". Use this exact process and only its direct related entities: "
        + json.dumps(
            {
                "process": contract["process_entity"],
                "allowed_direct_codes": contract["allowed_by_type"],
                "preferred_codes_by_relevance": contract["preferred_by_type"],
            },
            ensure_ascii=True,
        )
        + ". Do not mention this contract, retrieval, validation, or internal data."
    )


def _clean_process_answer(
    answer: str,
    contract: dict[str, Any] | None,
    question: str,
) -> str:
    """Produce a safe process-centered fallback after repair attempts fail."""
    feedback = _process_consistency_feedback(answer, contract)
    if not contract or not feedback:
        return answer
    is_turkish = bool(
        re.search(
            r"[çğıöşü]|\b(?:hangi|nedir|için|süreç|kök|neden|metrik|uygulama|beceri)\b",
            question.casefold(),
        )
    )
    if contract.get("status") == "ambiguous":
        primary_line = next(
            (
                line
                for line in answer.splitlines()
                if re.search(
                    r"\b(?:primary|root[- ]cause|ana|kök neden)\b",
                    line,
                    re.IGNORECASE,
                )
                and EXPLICIT_SCOR_ENTITY_CODE_PATTERN.search(line)
            ),
            "",
        )
        primary_match = EXPLICIT_SCOR_ENTITY_CODE_PATTERN.search(primary_line)
        selected_code = primary_match.group(0).upper() if primary_match else ""
        candidate_processes = contract.get("candidate_processes", {})
        resolved_code = selected_code if selected_code in candidate_processes else ""
        if selected_code and not resolved_code:
            children = [
                code
                for code, entity in candidate_processes.items()
                if str(entity.get("parent_code", "")).upper() == selected_code
            ]
            if len(children) == 1:
                resolved_code = children[0]
        if resolved_code:
            resolved = available_functions["get_scor_entity_details"](
                entity_codes=[resolved_code],
                query_text="",
                entity_types=[],
                limit=1,
            ).get("results", [{}])[0]
            details = resolved.get("details", {})
            related = resolved.get("related", {})
            labels = (
                {
                    "process": "Ana kök neden süreci",
                    "metric": "Metrik",
                    "practice": "Uygulama",
                    "skill": "Beceri",
                }
                if is_turkish
                else {
                    "process": "Primary root-cause process",
                    "metric": "Metric",
                    "practice": "Practice",
                    "skill": "Skill",
                }
            )
            lines = [
                f"**{labels['process']}: {resolved_code} — {details.get('name', '')}**"
            ]
            relation_keys = {
                "metric": "metrics",
                "practice": "best_practices",
                "skill": "people_skills",
            }
            for entity_type in contract.get("requested_types", []):
                related_key = relation_keys.get(entity_type)
                if not related_key:
                    continue
                for entity in related.get(related_key, [])[:3]:
                    if entity.get("code") and entity.get("name"):
                        lines.append(
                            f"- {labels[entity_type]}: **{entity['code']} — {entity['name']}**"
                        )
            return "\n".join(lines)

        branches = contract.get("candidate_branches", [])
        branch_labels = [
            f"{branch.get('parent_code')} — {branch.get('parent_name')}"
            for branch in branches
            if branch.get("parent_code") and branch.get("parent_name")
        ]
        if len(branch_labels) >= 2:
            alternatives = " or ".join(f"**{label}**" for label in branch_labels)
            if is_turkish:
                return (
                    "Etkilenen ürün akışı hangi üst SCOR süreç bağlamına ait: "
                    f"{alternatives}?"
                )
            return (
                "Which parent SCOR process context applies to the affected goods flow: "
                f"{alternatives}?"
            )
        if is_turkish:
            return (
                "Kesin SCOR sürecini seçebilmem için, aksamanın ilk görüldüğü fiziksel "
                "ve sistemsel olaylar hangileri ve bu iki olay arasındaki işlemin sahibi hangi "
                "ekip?"
            )
        return (
            "To select the exact SCOR process, what are the first physical and system "
            "events where the disruption appears, and which team owns the activity between them?"
        )

    process = contract["process_entity"]
    labels = (
        {"process": "Ana kök neden süreci", "metric": "Metrik", "practice": "Uygulama", "skill": "Beceri"}
        if is_turkish
        else {"process": "Primary root-cause process", "metric": "Metric", "practice": "Practice", "skill": "Skill"}
    )
    lines = [f"**{labels['process']}: {process['code']} — {process['name']}**"]
    for entity_type in ("metric", "practice", "skill"):
        preferred = contract["preferred_by_type"].get(entity_type, [])
        if not preferred:
            preferred = contract["allowed_by_type"].get(entity_type, [])
        for code in preferred[:3]:
            entity = contract["entities_by_type"].get(entity_type, {}).get(
                str(code).upper(), {}
            )
            if entity:
                lines.append(
                    f"- {labels[entity_type]}: **{entity['code']} — {entity['name']}**"
                )
    return "\n".join(lines)


def _mapping_consistency_feedback(
    answer: str,
    contract: dict[str, Any] | None,
) -> str:
    if not contract:
        return ""
    answer_codes = {
        match.group(0).upper()
        for match in EXPLICIT_SCOR_ENTITY_CODE_PATTERN.finditer(answer)
    }
    practice_code = str(contract["practice"]).upper()
    allowed_by_type = {
        entity_type: [str(code).upper() for code in codes]
        for entity_type, codes in contract["allowed_by_type"].items()
    }
    all_allowed_codes = {practice_code} | {
        code for codes in allowed_by_type.values() for code in codes
    }
    all_allowed_codes.update(
        code.rsplit(".", 1)[0]
        for code in allowed_by_type.get("process", [])
        if "." in code
    )
    issues: list[str] = []
    if practice_code not in answer_codes:
        issues.append(f"the leading practice code {practice_code} is missing")
    unsupported_codes = sorted(answer_codes - all_allowed_codes)
    if unsupported_codes:
        issues.append(
            "codes outside the leading practice's direct relationships: "
            + ", ".join(unsupported_codes)
        )
    for entity_type, allowed_codes in allowed_by_type.items():
        if allowed_codes and not answer_codes.intersection(allowed_codes):
            issues.append(f"no direct {entity_type} code was returned")
        preferred_codes = [
            str(code).upper()
            for code in contract["preferred_by_type"].get(entity_type, [])
        ]
        if (
            contract["singular_by_type"].get(entity_type)
            and preferred_codes
            and preferred_codes[0] not in answer_codes
        ):
            issues.append(
                f"the singular {entity_type} selection does not use the highest "
                f"definition-matched direct candidate {preferred_codes[0]}"
            )
    if not issues:
        return ""
    return (
        "Revise the draft because it failed the source-relationship consistency check: "
        + "; ".join(issues)
        + ". Keep the answer user-facing and concise. Use the leading practice and only "
        "its direct relationships shown in this contract: "
        + json.dumps(
            {
                "practice": contract["practice"],
                "allowed_direct_codes": contract["allowed_by_type"],
                "preferred_codes_by_relevance": contract["preferred_by_type"],
            },
            ensure_ascii=True,
        )
        + ". Do not mention this check, contract, retrieval, or internal data."
    )


def _clean_mapping_answer(
    answer: str,
    contract: dict[str, Any] | None,
    question: str = "",
) -> str:
    """Fail closed to source-backed codes after model repair attempts are exhausted."""
    if not contract or not _mapping_consistency_feedback(answer, contract):
        return answer

    practice_code = str(contract["practice"]).upper()
    allowed_by_type = {
        entity_type: [str(code).upper() for code in codes]
        for entity_type, codes in contract["allowed_by_type"].items()
    }
    allowed_codes = {practice_code} | {
        code for codes in allowed_by_type.values() for code in codes
    }
    allowed_codes.update(
        code.rsplit(".", 1)[0]
        for code in allowed_by_type.get("process", [])
        if "." in code
    )
    answer_code_matches = list(EXPLICIT_SCOR_ENTITY_CODE_PATTERN.finditer(answer))
    unsupported_codes = {
        match.group(0).upper()
        for match in answer_code_matches
        if match.group(0).upper() not in allowed_codes
    }
    singular_nonpreferred_codes: set[str] = set()
    for entity_type, is_singular in contract["singular_by_type"].items():
        preferred_codes = [
            str(code).upper()
            for code in contract["preferred_by_type"].get(entity_type, [])
        ]
        if not is_singular or not preferred_codes:
            continue
        singular_nonpreferred_codes.update(
            code
            for code in allowed_by_type.get(entity_type, [])
            if code != preferred_codes[0]
        )
    codes_to_remove = unsupported_codes | singular_nonpreferred_codes
    cleaned_lines = [
        line
        for line in answer.splitlines()
        if not any(code in line.upper() for code in codes_to_remove)
    ]
    cleaned_answer = "\n".join(cleaned_lines).strip()
    cleaned_codes = {
        match.group(0).upper()
        for match in EXPLICIT_SCOR_ENTITY_CODE_PATTERN.finditer(cleaned_answer)
    }

    is_turkish = bool(
        re.search(
            r"[çğıöşü]|\b(?:hangi|nedir|nelerdir|için|ilişkili|süreç|uygulama|"
            r"metrik|beceri|neden|kök)\b",
            question.casefold(),
        )
    )
    additions: list[str] = []
    if practice_code not in cleaned_codes:
        practice = contract["practice_entity"]
        practice_label = "Uygulama" if is_turkish else "Practice"
        additions.append(
            f"- {practice_label}: **{practice['code']} — {practice['name']}**"
        )
    labels = (
        {"process": "Süreç", "metric": "Metrik", "skill": "Beceri"}
        if is_turkish
        else {"process": "Process", "metric": "Metric", "skill": "Skill"}
    )
    for entity_type, preferred_codes in contract["preferred_by_type"].items():
        minimum_count = int(contract["minimum_count_by_type"].get(entity_type, 1))
        already_present = cleaned_codes.intersection(
            {str(code).upper() for code in allowed_by_type.get(entity_type, [])}
        )
        needed_count = max(0, minimum_count - len(already_present))
        if contract["singular_by_type"].get(entity_type):
            top_code = str(preferred_codes[0]).upper() if preferred_codes else ""
            needed_codes = [] if top_code in cleaned_codes else [top_code]
        else:
            needed_codes = [
                str(code).upper()
                for code in preferred_codes
                if str(code).upper() not in cleaned_codes
            ][:needed_count]
        for code in needed_codes:
            entity = contract["entities_by_type"].get(entity_type, {}).get(code, {})
            if not entity:
                continue
            additions.append(
                f"- {labels[entity_type]}: **{entity['code']} — {entity['name']}**"
            )
            cleaned_codes.add(code)

    definition = str(contract["practice_entity"].get("definition", "")).strip()
    if definition and not is_turkish:
        rationale = definition.split("\n", 1)[0].strip()
        additions.append(f"Why this mapping: {rationale}")
    return "\n".join(part for part in [cleaned_answer, *additions] if part).strip()


def _tool_outputs(function_calls: list[Any]) -> list[dict[str, str]]:
    outputs = []
    for tool_call in function_calls:
        call_id = getattr(tool_call, "call_id", None)
        if not call_id:
            raise RuntimeError(
                "Root-cause tool call_id bulunamadi for tool "
                f"{getattr(tool_call, 'name', '<unknown>')}"
            )
        log_with_prefix(
            "root-cause",
            "Model requested tool: "
            f"{tool_call.name} (call_id={call_id}) | args: "
            f"{str(tool_call.arguments)[:220]}",
        )
        result = run_base_tool(
            tool_call.name,
            tool_call.arguments,
            log_prefix="root-cause",
        )
        summary = summarize_base_tool_result(tool_call.name, result)
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": str(call_id),
                "output": json.dumps(summary, ensure_ascii=True, default=str),
            }
        )
    return outputs


def ask_root_cause_specialist(
    question: str,
    user_context: str = "",
    model: str = MODEL_NAME,
    system_prompt: str = ROOT_CAUSE_SPECIALIST_SYSTEM_PROMPT,
) -> dict[str, object]:
    """Run a dedicated SCOR-DS root-cause analysis with access to shared SCOR tools."""

    conversation_id = next(CONVERSATION_COUNTER)
    log_conversation_start(
        conversation_id,
        question,
        user_context,
        model,
        prefix="root-cause",
    )

    client = get_model_client()
    prompt, _ = _build_prompt_parts(question, user_context)
    request_items: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    previous_response_id: str | None = None
    tool_call_count = 0

    for _ in range(MAX_TOOL_ROUNDS + 1):
        request_args: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt,
            "input": request_items,
            "tools": ROOT_CAUSE_AGENT_TOOLS,
        }
        if previous_response_id:
            request_args["previous_response_id"] = previous_response_id

        response = client.responses.create(**request_args)
        function_calls = [
            item for item in getattr(response, "output", []) if item.type == "function_call"
        ]
        if not function_calls:
            answer = getattr(response, "output_text", "").strip()
            if not answer:
                raise RuntimeError("Root-cause specialist bos bir cevap dondurdu.")
            answer = _strip_internal_repair_wording(answer)
            log_with_prefix(
                "root-cause",
                f"Conversation #{conversation_id} | answer length={len(answer)} | "
                f"tool calls={tool_call_count}",
            )
            return {
                "answer": answer,
                "model": model,
                "tool_call_count": tool_call_count,
            }

        tool_call_count += len(function_calls)
        request_items = _tool_outputs(function_calls)
        previous_response_id = response.id

    raise RuntimeError(
        f"Root-cause specialist {MAX_TOOL_ROUNDS} tool turundan sonra tamamlanmadi."
    )
