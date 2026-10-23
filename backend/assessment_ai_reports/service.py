import json
import os
from collections.abc import Mapping
from typing import Any

from ..model_provider import get_model_client, model_provider
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL_SETTING = "ASSESSMENT_AI_REPORT_MODEL"
API_KEY_SETTING = "OPENAI_API_KEY"
MODEL_PROVIDER_SETTING = "MODEL_PROVIDER"
AZURE_ENDPOINT_SETTING = "AZURE_OPENAI_ENDPOINT"
AZURE_API_KEY_SETTING = "AZURE_OPENAI_KEY"
DEFAULT_MAX_PROMPT_CHARACTERS = 500_000
SUPPORTED_ASSESSMENT_TYPES = frozenset(
    {
        "sc_maturity_assessment",
        "digital_readiness_assessment",
        "sop_assessment",
    }
)

SYSTEM_INSTRUCTIONS = (
    "You generate supply-chain assessment report narratives. Follow the user prompt "
    "exactly, use only the assessment evidence supplied in that prompt, and return one "
    "valid JSON object. Do not wrap the JSON in Markdown fences or add commentary."
)


class AssessmentReportError(RuntimeError):
    """Base error raised by the assessment report service."""


class AssessmentReportConfigurationError(AssessmentReportError):
    """Raised when a required service setting is missing."""


class AssessmentReportValidationError(AssessmentReportError):
    """Raised when the request or model response is not valid."""


class AssessmentReportGenerationError(AssessmentReportError):
    """Raised when OpenAI cannot generate the assessment report."""


def _required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise AssessmentReportConfigurationError(f"{name} is not configured.")
    return value


def _model_provider() -> str:
    try:
        return model_provider()
    except ValueError as exc:
        raise AssessmentReportConfigurationError(str(exc)) from exc


def _create_model_client() -> OpenAI:
    try:
        return get_model_client()
    except (RuntimeError, ValueError) as exc:
        raise AssessmentReportConfigurationError(str(exc)) from exc



def _max_prompt_characters() -> int:
    raw_value = os.getenv("ASSESSMENT_AI_REPORT_MAX_PROMPT_CHARS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_PROMPT_CHARACTERS
    try:
        value = int(raw_value)
    except ValueError as error:
        raise AssessmentReportConfigurationError(
            "ASSESSMENT_AI_REPORT_MAX_PROMPT_CHARS must be an integer."
        ) from error
    if value < 1:
        raise AssessmentReportConfigurationError(
            "ASSESSMENT_AI_REPORT_MAX_PROMPT_CHARS must be greater than zero."
        )
    return value


def _validate_prompt(prompt: Any) -> str:
    if not isinstance(prompt, str):
        raise AssessmentReportValidationError("prompt must be a string.")
    normalized = prompt.strip()
    if not normalized:
        raise AssessmentReportValidationError("prompt is required.")
    max_characters = _max_prompt_characters()
    if len(normalized) > max_characters:
        raise AssessmentReportValidationError(
            f"prompt exceeds the maximum length of {max_characters} characters."
        )
    return normalized


def _validate_assessment_type(assessment_type: Any) -> str | None:
    if assessment_type is None:
        return None
    if not isinstance(assessment_type, str):
        raise AssessmentReportValidationError("assessment_type must be a string.")
    normalized = assessment_type.strip()
    if normalized not in SUPPORTED_ASSESSMENT_TYPES:
        supported = ", ".join(sorted(SUPPORTED_ASSESSMENT_TYPES))
        raise AssessmentReportValidationError(
            f"assessment_type must be one of: {supported}."
        )
    return normalized


def parse_assessment_report_json(raw_content: Any) -> dict[str, Any]:
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise AssessmentReportValidationError("OpenAI returned an empty assessment report.")

    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        content = "\n".join(lines).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise AssessmentReportValidationError(
            "OpenAI returned invalid JSON for the assessment report."
        ) from error
    if not isinstance(parsed, dict):
        raise AssessmentReportValidationError(
            "OpenAI assessment report response must be a JSON object."
        )
    return parsed


def generate_assessment_ai_report(
    prompt: str,
    *,
    assessment_type: str | None = None,
    client: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate the AI-owned sections of an assessment report from a prepared prompt."""

    validated_prompt = _validate_prompt(prompt)
    validated_type = _validate_assessment_type(assessment_type)
    resolved_model = (model or "").strip() or _required_setting(MODEL_SETTING)

    if client is None:
        client = _create_model_client()

    type_context = (
        f"Assessment type: {validated_type}.\n\n" if validated_type is not None else ""
    )
    try:
        response = client.responses.create(
            model=resolved_model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": f"{type_context}{validated_prompt}",
                }
            ],
        )
    except Exception as error:
        raise AssessmentReportGenerationError(
            "AI assessment report generation failed."
        ) from error

    try:
        return parse_assessment_report_json(getattr(response, "output_text", ""))
    except AssessmentReportValidationError as error:
        raise AssessmentReportGenerationError(
            "OpenAI returned an invalid assessment report."
        ) from error


def generate_assessment_ai_report_response(body: Any) -> dict[str, Any]:
    """Validate an HTTP request body and return the stable service response envelope."""

    if not isinstance(body, Mapping):
        raise AssessmentReportValidationError("Request body must be a JSON object.")

    assessment_type = _validate_assessment_type(body.get("assessment_type"))
    report = generate_assessment_ai_report(
        body.get("prompt"),
        assessment_type=assessment_type,
    )
    return {
        "assessment_type": assessment_type,
        "model": _required_setting(MODEL_SETTING),
        "report": report,
    }
