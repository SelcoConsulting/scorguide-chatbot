"""Semantic retrieval tool backed by a Milvus collection."""

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from ..model_provider import get_model_client

from ..utils.logging import log_tool_end, log_tool_start

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _required_setting(name: str) -> str:
    value = _env(name)
    if not value:
        raise RuntimeError(f"{name} tanimli degil.")
    return value


@lru_cache(maxsize=1)
def _milvus_client():
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError(
            "Milvus tool icin pymilvus kurulu degil. `pip install pymilvus` calistirin."
        ) from exc

    kwargs: dict[str, Any] = {"uri": _required_setting("MILVUS_URI")}
    token = _env("MILVUS_TOKEN")
    database = _env("MILVUS_DATABASE")
    if token:
        kwargs["token"] = token
    if database:
        kwargs["db_name"] = database
    return MilvusClient(**kwargs)


def _embed_query(query: str) -> list[float]:
    model = _env("MILVUS_EMBEDDING_MODEL", "text-embedding-3-small")
    response = get_model_client().embeddings.create(model=model, input=query)
    return response.data[0].embedding


def _output_fields(available_fields: set[str]) -> list[str]:
    fields = [
        _env("MILVUS_TEXT_FIELD", "text"),
        _env("MILVUS_TITLE_FIELD", "title"),
        _env("MILVUS_SOURCE_FIELD", "source"),
    ]
    extra_fields = [item.strip() for item in _env("MILVUS_OUTPUT_FIELDS").split(",")]
    requested = list(dict.fromkeys(field for field in [*fields, *extra_fields] if field))
    return [field for field in requested if field in available_fields]


def search_supply_chain_knowledge(query: str, limit: int = 5) -> dict[str, Any]:
    """Search general supply-chain documents stored in Milvus."""
    query = query.strip()
    if not query:
        raise ValueError("Arama sorgusu bos olamaz.")
    limit = max(1, min(int(limit), 10))
    log_tool_start("search_supply_chain_knowledge", query=query, limit=limit)

    collection_name = _required_setting("MILVUS_COLLECTION")
    client = _milvus_client()
    available_fields = [
        "topic", "sub_topic", "title", "paragraph_id", "text"
    ]
    search_kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "data": [_embed_query(query)],
        "limit": limit,
        "output_fields": _output_fields(available_fields),
        "search_params": {
            "metric_type": _env("MILVUS_METRIC_TYPE", "COSINE"),
            "params": {},
        },
    }
    vector_field = _env("MILVUS_VECTOR_FIELD")
    if vector_field:
        search_kwargs["anns_field"] = vector_field

    raw_hits = client.search(**search_kwargs)
    hits = raw_hits[0] if raw_hits else []
    text_field = _env("MILVUS_TEXT_FIELD", "text")
    title_field = _env("MILVUS_TITLE_FIELD", "title")
    source_field = _env("MILVUS_SOURCE_FIELD", "source")
    relevance_threshold = float(_env("MILVUS_RELEVANCE_THRESHOLD", "0.50"))
    ranked_hits = sorted(
        hits,
        key=lambda item: float(item.get("distance") or 0.0),
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for index, hit in enumerate(ranked_hits):
        score = float(hit.get("distance") or 0.0)
        # The best available match is always useful context for the LLM. Apply
        # the threshold only to additional passages and keep it strictly above.
        if index > 0 and score <= relevance_threshold:
            continue
        entity = hit.get("entity") or {}
        source = entity.get(source_field, "")
        if not source:
            source = " / ".join(
                str(entity.get(field, "")).strip()
                for field in ("topic", "sub_topic")
                if str(entity.get(field, "")).strip()
            )
        results.append(
            {
                "id": hit.get("id"),
                "score": score,
                "relevance_percent": round(score * 100, 2),
                "title": entity.get(title_field, ""),
                "text": entity.get(text_field, ""),
                "source": source,
                "metadata": {
                    key: value
                    for key, value in entity.items()
                    if key not in {text_field, title_field, source_field}
                },
            }
        )

    response = {
        "query": query,
        "relevance_threshold": relevance_threshold,
        "selection_policy": "best_match_always; additional_matches_above_threshold",
        "result_count": len(results),
        "results": results,
    }
    log_tool_end("search_supply_chain_knowledge", response)
    print("---------------------------------MILVUS---------------------------------")
    print(f"search_supply_chain_knowledge response:")
    for index, item in enumerate(results, start=1):
        print(item["text"][:200].replace("\n", " ") + "...")
    print(f"Total results: {len(results)}")
    print("---------------------------------MILVUS---------------------------------")
    return response


SUPPLY_CHAIN_KNOWLEDGE_TOOL = {
    "type": "function",
    "name": "search_supply_chain_knowledge",
    "description": (
        "Semantically searches the private general supply-chain document library in Milvus. "
        "Use it for supply-chain, logistics, procurement, planning, inventory, warehousing, "
        "transportation, manufacturing, fulfillment, or return questions that need information "
        "beyond the structured SCOR entity database."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A self-contained semantic search query based on the user's question.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of document passages to retrieve (1-10).",
            },
        },
        "required": ["query", "limit"],
        "additionalProperties": False,
    },
}
