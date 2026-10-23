"""Shared provider selection for every model and embedding client."""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

def model_provider() -> str:
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "azure"}:
        raise ValueError("MODEL_PROVIDER must be either 'openai' or 'azure'.")
    return provider

def required_model_settings() -> tuple[str, ...]:
    if model_provider() == "azure":
        return ("AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT")
    return ("OPENAI_API_KEY",)

def model_client_options() -> dict[str, str]:
    settings = required_model_settings()
    for name in settings:
        if not os.getenv(name, "").strip():
            raise RuntimeError(f"{name} is not configured.")
    options = {"api_key": os.environ[settings[0]].strip()}
    if model_provider() == "azure":
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].strip().rstrip("/")
        if not endpoint.endswith("/openai/v1"):
            endpoint += "/openai/v1"
        options["base_url"] = endpoint + "/"
    return options

def get_model_client() -> OpenAI:
    return OpenAI(**model_client_options())
