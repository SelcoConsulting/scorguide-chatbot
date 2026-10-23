import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .logging import log_tool_end, log_tool_start, summarize_log_value, tool_log

_tool_log = tool_log
_summarize_for_log = summarize_log_value
_log_tool_start = log_tool_start
_log_tool_end = log_tool_end


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
KNOWN_TOP_LEVEL_PROCESSES = {
    "orchestrate",
    "plan",
    "order",
    "source",
    "transform",
    "fulfill",
    "return",
}
KNOWN_SECTION_NAMES = KNOWN_TOP_LEVEL_PROCESSES | {
    "processes",
    "performance",
    "people",
    "practices",
}
SEARCH_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "as",
    "for",
    "with",
    "from",
    "into",
    "onto",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "did",
    "has",
    "have",
    "had",
    "during",
    "while",
    "than",
    "because",
    "that",
    "this",
    "what",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "you",
    "your",
    "can",
    "could",
    "would",
    "should",
    "does",
    "mean",
    "means",
    "meaning",
    "explain",
    "define",
    "term",
    "call",
    "called",
    "document",
    "method",
    "approach",
    "concept",
    "strategy",
    "check",
    "scor",
    "process",
    "people",
    "skill",
    "knowledge",
    "base",
    "support",
    "supported",
    "directly",
    "most",
}
SCOR_CODE_PATTERN = re.compile(
    r"\b[A-Z]{1,3}(?:[.,-]?\d+){1,3}\b",
    re.IGNORECASE,
)
CODE_CHUNK_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9.,-]*", re.IGNORECASE)
SLASH_PROCESS_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:/[A-Z][a-z]+)+)\b")
QUOTED_TEXT_PATTERN = re.compile(r'"([^"]+)"')
ENTITY_TYPE_LABELS = {
    "processes": "process",
    "practices": "best practice",
    "metrics": "metric",
    "skills": "people skill",
    "experiences": "people experience",
    "trainings": "people training",
}


def _get_first_present(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-null value found among multiple candidate keys."""
    for key in keys:
        if key in record and record.get(key) is not None:
            return record.get(key)
    return default


def _get_scalar_text(value: Any, default: str = "") -> str:
    """Extract the first non-empty scalar-looking text from a value."""
    if value is None:
        return default
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return default
    text = str(value).strip()
    return text or default


@lru_cache(maxsize=4)
def _load_json(filename: str) -> list[dict[str, Any]]:
    """Load one knowledge-base JSON file and cache the parsed content."""
    path = KNOWLEDGE_BASE_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_scor_entity_code_lists() -> dict[str, list[str]]:
    """Load the precomputed entity-code index used for code normalization."""
    path = KNOWLEDGE_BASE_DIR / "scor_entities_code_lists.json"
    if not path.exists():
        return {
            "processes": [],
            "practices": [],
            "metrics": [],
            "skills": [],
            "experiences": [],
            "trainings": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_dictionary_entries() -> tuple[dict[str, str], ...]:
    """Load and cache glossary-style dictionary entries used for term clarification."""
    path = KNOWLEDGE_BASE_DIR / "dictionary.json"
    if not path.exists():
        return tuple()

    raw_entries = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, str]] = []
    for item in raw_entries:
        word = str(item.get("word", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if word and definition:
            entries.append({"word": word, "definition": definition})
    return tuple(entries)


def _normalize_name_for_match(value: str) -> str:
    """Normalize free text into a comparison-friendly lowercase form."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _canonical_code_token(value: str) -> str:
    """Remove separators so variant spellings of a SCOR code compare equally."""
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _tokenize_text(value: str) -> list[str]:
    """Split text into normalized search tokens and drop very short terms."""
    normalized = re.sub(r"\bpo\b", "purchase order", value.lower())
    return [
        token
        for token in re.split(r"[^a-z0-9]+", normalized)
        if len(token) >= 3 and token not in SEARCH_STOPWORDS
    ]


def _truncate_text(value: str, max_length: int = 280) -> str:
    """Shorten long text fields so aggregated responses stay compact."""
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def _extract_quoted_phrases(text: str) -> list[str]:
    """Return quoted phrases that may represent explicit entity names."""
    return [match.group(1).strip() for match in QUOTED_TEXT_PATTERN.finditer(text) if match.group(1).strip()]
