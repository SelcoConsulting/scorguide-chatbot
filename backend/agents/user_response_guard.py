from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from ..model_provider import get_model_client


_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "reasoning markup",
        re.compile(
            r"</?(?:analysis|reasoning|chain[-_ ]?of[-_ ]?thought|scratchpad)>"
            r"|^\s*(?:private\s+)?(?:analysis|reasoning|scratchpad)\s*:\s*",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "self-correction wording",
        re.compile(
            r"^\s*(?:wait(?:\s*[—–-]\s*no)?|actually|scratch that|correction|"
            r"no\s*[,—–-]\s*(?:that|this|the previous|i)|i was (?:wrong|mistaken)|"
            r"let me (?:correct|reconsider|recheck|revise)|"
            r"i need to (?:correct|reconsider|recheck|revise))\b\s*[,.:—–-]?"
            r"|^\s*(?:dur(?:\s*[—–-]\s*hayır)?|aslında|düzeltme|"
            r"hayır\s*[,—–-]\s*(?:bu|önceki|ben)|yanılmışım|"
            r"tekrar (?:kontrol edeyim|düşüneyim)|"
            r"bunu düzelt(?:eyim|mem gerek))\b\s*[,.:—–-]?",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "private knowledge-base wording",
        re.compile(r"\bknowledge[\s_-]*base\b|\bbilgi\s+taban[ıi]\b", re.IGNORECASE),
    ),
    (
        "local file path",
        re.compile(
            r"(?:^|[\s`(])(?:\.\.?[\\/])?(?:backend|azure_deployment)[\\/][^\s`)]*"
            r"|\b(?:dictionary|metrics|processes|practices|skills)\.json\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal schema field",
        re.compile(
            r"\b(?:source_manifest|lookup_key|verified_paths?|verified_relationships?|"
            r"direct_relationships|measured_processes|measures_process_performance|"
            r"performance_metrics|associated_skills|input_processes|output_processes|"
            r"supports_processes|impacts_metrics|required_skills|applicable_processes|"
            r"supporting_best_practices|improved_by_best_practices?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal evidence inventory wording",
        re.compile(
            r"\b(?:approved|provided|stored|verified)\s+"
            r"(?:(?:relationship|source)\s+)?"
            r"(?:inventory|records?|paths?|sources?|links?|relationships?|fields?)\b"
            r"|\brelationship\s+inventory\b"
            r"|\b(?:available|shown)\s+(?:records?|relationship fields?|fields?)\b"
            r"|\b(?:the|this) (?:SCOR )?record (?:shows?|lists?|contains?|includes?|"
            r"provides?|does not|doesn't|has no)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal processing wording",
        re.compile(
            r"\b(?:local structured data|internal retrieval|retrieval rank|tool result|"
            r"source[- ]of[- ]truth|approved source locator|verified path|"
            r"tool[_ -]?name|function[_ -]?call[_ -]?output|tool round limit|"
            r"system prompt|developer message|retrieved context)\b"
            r"|\b(?:the|this) tool (?:returned|provided|shows?|found)\b"
            r"|\b(?:i|we) (?:called|used|queried) (?:a|the) tool\b"
            r"|\b(?:araç|tool) çağr(?:ısı|ısını|ısında)\b"
            r"|\b(?:sistem promptu|geliştirici mesajı|dahili bağlam)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "record-oriented answer preface",
        re.compile(
            r"\bwhat i can confirm from (?:your|the) (?:provided |stored )?records?\b"
            r"|\bfrom (?:your|the) (?:provided |stored |approved )?"
            r"(?:records?|relationship inventory|verified paths?)\b"
            r"|\b(?:your|the) provided [\u201c\u201d\"']?verified paths?[\u201c\u201d\"']?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal field-availability wording",
        re.compile(
            r"\b(?:fields?|relationship fields?) (?:shown|provided|available|above)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal evidence heading",
        re.compile(
            r"^\s*(?:#{1,6}\s*)?(?:what is stored|stored evidence|record evidence)\s*:?\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "Turkish internal evidence wording",
        re.compile(
            r"\b(?:sağladığınız|verilen|saklanan|doğrulanmış|onaylı)\s+"
            r"(?:kayıt(?:lar)?(?:da|de|dan|den|a|e)?|"
            r"kaynak(?:lar)?(?:da|de|dan|den|a|e)?|yollar?|ilişki envanteri)\b"
            r"|\biç (?:işleyiş|mekanizma|veri kaynağı|veri kaynakları)\b",
            re.IGNORECASE,
        ),
    ),
)


PRESENTATION_GUARD_INSTRUCTIONS = """
You are a presentation-only guard for an end-user supply-chain assistant.

Rewrite the immediately preceding assistant answer so it reads like a direct consultation answer.
Preserve every material fact, qualification, SCOR code, approved entity name, number, list item,
and requested language. Do not add new facts, relationships, causes, targets, or recommendations.
Silently resolve presentation defects before returning the answer. Never expose draft reasoning,
scratch work, analysis/reasoning tags, or self-correction phrases such as "Wait—no", "Actually",
"Let me correct that", "Dur—hayır", "Aslında", or "Düzeltme".

Never mention or allude to private knowledge bases, supplied/stored/approved records, verified
paths, relationship inventories, retrieval, tools, prompts, schema fields, JSON field names, local
files, repository paths, or system behavior. State domain facts directly. For example, replace
"the knowledge base does not contain a target" with "SCOR DS does not define a target for this
metric." If the user requests a citation, use a human-readable publication name such as "ASCM
SCOR DS Dictionary" or "ASCM SCOR DS framework"; never expose a filename, path, lookup key, or
internal locator.

Return only the revised answer. Do not describe the review or the changes you made.
""".strip()


@dataclass(frozen=True)
class GuardedResponse:
    answer: str
    response_id: str | None
    revised: bool


def find_internal_response_leaks(answer: str) -> list[str]:
    """Return stable policy labels for internal wording found in a user-facing answer."""
    labels: list[str] = []
    for label, pattern in _LEAK_PATTERNS:
        if pattern.search(answer):
            labels.append(label)
    return labels


def guard_user_facing_response(
    answer: str,
    *,
    model: str,
    response_id: str | None,
    api_key: str | None = None,
    client: Any | None = None,
    max_revisions: int = 2,
) -> GuardedResponse:
    """Conditionally revise an answer and fail closed if internal wording remains."""
    text = answer.strip()
    leaks = find_internal_response_leaks(text)
    if not leaks:
        return GuardedResponse(answer=text, response_id=response_id, revised=False)
    if max_revisions < 1:
        raise ValueError("max_revisions must be at least 1")

    if client is None:
        client = get_model_client()

    previous_response_id = response_id
    original_answer = text
    for attempt in range(max_revisions):
        violation_summary = ", ".join(leaks)
        if previous_response_id:
            request_text = (
                "Rewrite your immediately preceding answer. The presentation check detected: "
                f"{violation_summary}. Preserve the answer's facts and language."
            )
        else:
            request_text = (
                "Rewrite the answer below. The presentation check detected: "
                f"{violation_summary}. Preserve the answer's facts and language.\n\n"
                f"Answer to rewrite:\n{original_answer}"
            )

        request_args: dict[str, Any] = {
            "model": model,
            "instructions": PRESENTATION_GUARD_INSTRUCTIONS,
            "input": [{"role": "user", "content": request_text}],
        }
        if previous_response_id:
            request_args["previous_response_id"] = previous_response_id

        response = client.responses.create(**request_args)
        revised_answer = str(getattr(response, "output_text", "")).strip()
        if not revised_answer:
            raise RuntimeError("Presentation guard bos bir cevap dondurdu.")

        leaks = find_internal_response_leaks(revised_answer)
        previous_response_id = str(getattr(response, "id", "") or "") or None
        if not leaks:
            return GuardedResponse(
                answer=revised_answer,
                response_id=previous_response_id,
                revised=True,
            )

    raise RuntimeError(
        "User-facing response still contains internal implementation wording after revision."
    )
