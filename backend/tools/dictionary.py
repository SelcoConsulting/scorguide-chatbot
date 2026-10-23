import difflib
import re
from typing import Any

from ..utils.shared import (
    _extract_quoted_phrases,
    _load_dictionary_entries,
    _normalize_name_for_match,
    _summarize_for_log,
    _tokenize_text,
    _tool_log,
    _truncate_text,
)


def _normalize_search_token(token: str) -> str:
    """Loosely normalize common inflection variants for relevance comparison."""
    if len(token) <= 4:
        return token
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 5:
        return token[:-2]
    if token.endswith("ss"):
        return token
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _build_ngrams(tokens: list[str], size: int) -> set[str]:
    """Build fixed-size token n-grams for simple relevance scoring."""
    return {
        " ".join(tokens[index:index + size])
        for index in range(len(tokens) - size + 1)
    }


def _build_search_features(value: str) -> dict[str, Any]:
    """Tokenize once and expose the search features used by text scoring."""
    tokens = _tokenize_text(value)
    normalized_tokens = [_normalize_search_token(token) for token in tokens]
    return {
        "tokens": tokens,
        "token_set": set(tokens),
        "normalized_token_set": set(normalized_tokens),
        "bigrams": _build_ngrams(tokens, 2),
        "trigrams": _build_ngrams(tokens, 3),
        "normalized_bigrams": _build_ngrams(normalized_tokens, 2),
        "normalized_trigrams": _build_ngrams(normalized_tokens, 3),
    }


def _score_text_match(query_text: str, name: str, definition: str) -> int:
    """Score how well a query matches an entity name and definition."""
    query_features = _build_search_features(query_text)
    if not query_features["tokens"]:
        return 0

    name_features = _build_search_features(name)
    definition_features = _build_search_features(definition)

    score = 0
    score += 4 * sum(1 for token in query_features["token_set"] if token in name_features["token_set"])
    score += sum(1 for token in query_features["token_set"] if token in definition_features["token_set"])
    score += 3 * sum(
        1 for token in query_features["normalized_token_set"] if token in name_features["normalized_token_set"]
    )
    score += 2 * sum(
        1 for token in query_features["normalized_token_set"] if token in definition_features["normalized_token_set"]
    )
    score += 6 * sum(1 for gram in query_features["bigrams"] if gram in name_features["bigrams"])
    score += 3 * sum(1 for gram in query_features["bigrams"] if gram in definition_features["bigrams"])
    score += 10 * sum(1 for gram in query_features["trigrams"] if gram in name_features["trigrams"])
    score += 5 * sum(1 for gram in query_features["trigrams"] if gram in definition_features["trigrams"])
    score += 8 * sum(
        1 for gram in query_features["normalized_bigrams"] if gram in name_features["normalized_bigrams"]
    )
    score += 5 * sum(
        1 for gram in query_features["normalized_bigrams"] if gram in definition_features["normalized_bigrams"]
    )
    score += 12 * sum(
        1 for gram in query_features["normalized_trigrams"] if gram in name_features["normalized_trigrams"]
    )
    score += 8 * sum(
        1 for gram in query_features["normalized_trigrams"] if gram in definition_features["normalized_trigrams"]
    )

    normalized_query = _normalize_name_for_match(query_text)
    normalized_name = _normalize_name_for_match(name)
    normalized_definition = _normalize_name_for_match(definition)
    if normalized_name and normalized_name in normalized_query:
        score += 12
    if normalized_query and normalized_query in normalized_definition:
        score += 8

    return score


def _extract_lookup_targets(normalized_query: str, quoted_phrases: list[str]) -> set[str]:
    """Return explicit glossary targets found in a normalized query."""
    targets = {phrase for phrase in quoted_phrases if phrase}
    stripped = re.sub(
        r"^(?:what is|what does|define|explain|meaning of)\s+",
        "",
        normalized_query,
    ).strip()
    if stripped:
        targets.add(stripped)
    return targets


def _deduplicate_candidates(
    candidates: list[tuple[int, str, str, str]],
) -> list[tuple[int, str, str, str]]:
    """Keep the strongest candidate for each normalized dictionary term."""
    unique: dict[str, tuple[int, str, str, str]] = {}
    for candidate in candidates:
        normalized_word = _normalize_name_for_match(candidate[1])
        existing = unique.get(normalized_word)
        if existing is None or candidate[0] > existing[0]:
            unique[normalized_word] = candidate
    return list(unique.values())


def _search_dictionary_terms(query_text: str, limit: int) -> list[dict[str, Any]]:
    """Search glossary entries by exact phrase, fuzzy term, or text relevance."""
    query_tokens = _tokenize_text(query_text)
    if not query_text.strip() or not query_tokens:
        _tool_log(
            "dictionary_search",
            f"skipped | query={_summarize_for_log(query_text)} | reason='empty_or_no_tokens'",
        )
        return []

    _tool_log(
        "dictionary_search",
        f"called | query={_summarize_for_log(query_text)} | limit={limit}",
    )
    normalized_query = _normalize_name_for_match(query_text)
    quoted_phrases = [_normalize_name_for_match(value) for value in _extract_quoted_phrases(query_text)]
    lookup_targets = _extract_lookup_targets(normalized_query, quoted_phrases)
    is_reverse_lookup_query = bool(
        re.search(
            r"\b(?:which|what)\b.*(?:means|refers to|describes|stands for|do you call|strategy|concept|method|approach|document|metric|acronym)",
            normalized_query,
        )
    )
    candidates: list[tuple[int, str, str, str]] = []

    for entry in _load_dictionary_entries():
        normalized_word = _normalize_name_for_match(entry["word"])
        compact_word = re.sub(r"[^a-z0-9]+", "", normalized_word)
        match_type = ""
        score = 0

        if normalized_word and normalized_word in lookup_targets:
            match_type = "exact_term"
            score = 20_000
        elif normalized_word and f" {normalized_word} " in f" {normalized_query} ":
            match_type = "exact_term"
            # Prefer the most specific phrase when several nested dictionary
            # terms occur in the same query (for example supply, chain, and
            # supply chain visibility).
            score = 9_500 + len(normalized_word.split()) * 100 + len(normalized_word)
        else:
            ratio = difflib.SequenceMatcher(None, normalized_query, normalized_word).ratio()
            if ratio >= 0.9:
                match_type = "fuzzy_term"
                score = 8_500 + int(ratio * 100)
            else:
                score = _score_text_match(
                    query_text,
                    "" if is_reverse_lookup_query else entry["word"],
                    entry["definition"],
                )
                if score:
                    match_type = "text_search"

        if match_type == "text_search":
            if score < 8 or len(compact_word) < 3:
                continue

        if match_type:
            candidates.append((score, entry["word"], entry["definition"], match_type))

    unique_candidates = _deduplicate_candidates(candidates)
    target_matches = [candidate for candidate in unique_candidates if candidate[0] >= 20_000]
    if target_matches:
        # A direct target is authoritative; lower-quality fuzzy/text results
        # only add ambiguity and can displace the requested definition.
        top_matches = sorted(target_matches, key=lambda item: (-item[0], item[1].lower()))[:1]
    else:
        top_matches = sorted(
            unique_candidates,
            key=lambda item: (-item[0], -len(_normalize_name_for_match(item[1])), item[1].lower()),
        )[:limit]
    results = [
        {
            "word": word,
            "definition": _truncate_text(definition, 500),
            "match_type": match_type,
            "match_score": score,
        }
        for score, word, definition, match_type in top_matches
    ]
    _tool_log(
        "dictionary_search",
        f"returning | match_count={len(results)} | top_words={[match['word'] for match in results[:3]]}",
    )
    return results
