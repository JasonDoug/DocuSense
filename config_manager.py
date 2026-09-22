import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
    "llm_provider": "gemini",
    "llm_api_key": os.environ.get("GEMINI_API_KEY", os.environ.get("LLM_API_KEY", "")),
    "llm_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "llm_model": "gemini-2.5-flash",
    "vlm_provider": "gemini",
    "vlm_api_key": os.environ.get("GEMINI_API_KEY", os.environ.get("VLM_API_KEY", "")),
    "vlm_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "vlm_model": "gemini-2.5-flash",
    "process_sequentially": True,
    "ocr_fallback": True,
    "max_workers": 4,
    "min_image_size": 100,
}

PROVIDER_DEFAULTS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "llm_model": "gemini-2.5-flash",
        "vlm_model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY"
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "llm_model": "google/gemini-2.5-flash",
        "vlm_model": "google/gemini-2.5-flash",
        "env_key": "OPENROUTER_API_KEY"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "llm_model": "llama3.2",
        "vlm_model": "llava",
        "env_key": ""
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "llm_model": "gpt-4o-mini",
        "vlm_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY"
    },
    "custom": {
        "base_url": "",
        "llm_model": "",
        "vlm_model": "",
        "env_key": ""
    }
}


def load_settings() -> Dict[str, Any]:
    """Load settings from settings.json, falling back to environment variables / defaults."""
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                settings.update(saved)
        except Exception as e:
            logger.error(f"Failed to read {SETTINGS_FILE}: {e}")

    # Fallback to environment variables if keys are empty
    if not settings["llm_api_key"]:
        provider = settings.get("llm_provider", "gemini")
        env_key_name = PROVIDER_DEFAULTS.get(provider, {}).get("env_key")
        if env_key_name and os.environ.get(env_key_name):
            settings["llm_api_key"] = os.environ[env_key_name]
        elif os.environ.get("LLM_API_KEY"):
            settings["llm_api_key"] = os.environ["LLM_API_KEY"]

    if not settings["vlm_api_key"]:
        provider = settings.get("vlm_provider", "gemini")
        env_key_name = PROVIDER_DEFAULTS.get(provider, {}).get("env_key")
        if env_key_name and os.environ.get(env_key_name):
            settings["vlm_api_key"] = os.environ[env_key_name]
        elif os.environ.get("VLM_API_KEY"):
            settings["vlm_api_key"] = os.environ["VLM_API_KEY"]

    return settings


def save_settings(new_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Save updated settings to settings.json and update os.environ."""
    current = load_settings()
    current.update(new_settings)

    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        logger.info(f"Saved settings to {SETTINGS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save settings to {SETTINGS_FILE}: {e}")
        raise e

    # Update environment variables
    if current.get("llm_api_key"):
        os.environ["LLM_API_KEY"] = current["llm_api_key"]
    if current.get("llm_base_url"):
        os.environ["LLM_API_BASE_URL"] = current["llm_base_url"]
    if current.get("llm_model"):
        os.environ["LLM_MODEL_NAME"] = current["llm_model"]

    if current.get("vlm_api_key"):
        os.environ["VLM_API_KEY"] = current["vlm_api_key"]
    if current.get("vlm_base_url"):
        os.environ["VLM_API_BASE_URL"] = current["vlm_base_url"]
    if current.get("vlm_model"):
        os.environ["VLM_MODEL_NAME"] = current["vlm_model"]

    return current
