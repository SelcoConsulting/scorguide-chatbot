import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ...config import models
from ...model_provider import model_client_options
from ...tools.supply_chain_knowledge import search_supply_chain_knowledge
from .supply_chain_specialist_system_prompt import SUPPLY_CHAIN_SPECIALIST_SYSTEM_PROMPT

load_dotenv()

def _format_knowledge_context(search_result: dict) -> str:
    passages = []
    for index, item in enumerate(search_result.get("results", []), start=1):
        text = str(item.get("text", "")).strip()
        if text:
            relevance = item.get("relevance_percent")
            relevance_line = (
                f"Internal retrieval relevance: {relevance}%\n"
                if isinstance(relevance, (int, float))
                else ""
            )
            passages.append(
                f"[{index}] Internal retrieval rank: {index}\n"
                f"{relevance_line}Title: {item.get('title') or '-'}\n"
                f"Source: {item.get('source') or '-'}\nPassage: {text[:3000]}"
            )
    return "\n\n".join(passages)


def ask_supply_chain_specialist(question: str, user_context: str = "") -> dict[str, object]:
    """Answer broad questions, retrieving private supply-chain knowledge when relevant."""
    model_name = (
        os.getenv("GENERAL_AGENT_MODEL")
        or os.getenv("OPENAI_FALLBACK_MODEL")
        or models["supply_chain_specialist"]
    )
    prompt = question.strip()
    knowledge_result = {"result_count": 0, "results": []}

    knowledge_result = search_supply_chain_knowledge(query=question, limit=5)
    knowledge_context = _format_knowledge_context(knowledge_result)
    if knowledge_context:
        prompt = (
            "Candidate private knowledge passages are listed below in descending retrieval "
            "order. The first passage is the best available match, but its presence does not "
            "guarantee that it answers the request. Retrieval metadata is internal and must "
            "not appear in the answer.\n\n"
            f"{knowledge_context}\n\n"
            "Evaluate the passages, then answer the current request using only supported "
            "facts. If they are insufficient, explain the limitation clearly and specifically. "
            "Do not print all passages verbatim.\n\n"
            f"Current request:\n{question.strip()}"
        )
    else:
        prompt = (
            "No usable private knowledge passage was available for this request. Explain in "
            "the user's language that the available knowledge is insufficient for a reliable "
            "answer, and briefly say what specific information would be needed. Do not answer "
            "the factual part from outside knowledge and do not mention retrieval internals.\n\n"
            f"Current request:\n{question.strip()}"
        )

    if user_context.strip():
        prompt = f"User context:\n{user_context.strip()}\n\n{prompt}"

    llm = ChatOpenAI(model=model_name, **model_client_options())
    message = llm.invoke(
        [
            ("system", SUPPLY_CHAIN_SPECIALIST_SYSTEM_PROMPT.strip()),
            ("human", prompt),
        ]
    )
    text = getattr(message, "content", "")
    if isinstance(text, list):
        text = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in text
        )
    text = str(text).strip()
    if not text:
        raise RuntimeError("General model returned an empty response.")
    return {
        "answer": text,
        "model": model_name,
        "used_supply_chain_knowledge": bool(knowledge_result.get("result_count", 0)),
        "knowledge_result_count": int(knowledge_result.get("result_count", 0)),
    }
