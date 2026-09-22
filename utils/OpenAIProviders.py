from .BaseProviders import BaseLLMProvider, BaseVLMProvider
from typing import Tuple, Optional
import openai
import logging
import io
import base64
import os
from PIL import Image
from .helpers import get_text_and_last_paragraph

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')
logger = logging.getLogger(__name__)


PROVIDER_DEFAULTS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "llm_model": "gemini-2.5-flash",
        "vlm_model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "llm_model": "google/gemini-2.5-flash",
        "vlm_model": "google/gemini-2.5-flash",
        "env_key": "OPENROUTER_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "llm_model": "llama3.2",
        "vlm_model": "llava",
        "env_key": "",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "llm_model": "gpt-4o-mini",
        "vlm_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "custom": {
        "base_url": "",
        "llm_model": "",
        "vlm_model": "",
        "env_key": "",
    }
}


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM API provider for text processing.
    Supports OpenAI, Gemini, OpenRouter, Ollama, and custom endpoints.
    """

    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None, provider_type: str = "openai"):
        self.provider_type = provider_type.lower()
        defaults = PROVIDER_DEFAULTS.get(self.provider_type, PROVIDER_DEFAULTS["custom"])

        if not base_url and defaults["base_url"]:
            base_url = defaults["base_url"]
        if not model_name and defaults["llm_model"]:
            model_name = defaults["llm_model"]

        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name)

    def _get_api_key_env_var(self):
        return "LLM_API_KEY"

    def _get_base_url_env_var(self) -> str:
        return "LLM_API_BASE_URL"

    def _get_model_name_env_var(self) -> str:
        return "LLM_MODEL_NAME"

    def _initialize(self):
        try:
            key = self.api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
            url = self.base_url or None

            headers = {}
            if self.provider_type == "openrouter":
                headers = {"HTTP-Referer": "https://docusense.app", "X-Title": "DocuSense"}

            self.client = openai.OpenAI(
                api_key=key,
                base_url=url,
                default_headers=headers if headers else None
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise

    def process_text(self, text: str, context: str = None, prompt_template: str = None) -> Tuple[str, str]:
        context_str = f"Previous context: {context}\n\n" if context else ""

        if not prompt_template:
            prompt_template = """
            {context}The following is a chunk of text from a PDF document:

            {text}

            Extract and clean the text while preserving meaning but fixing OCR or formatting issues. Identify document structure elements (headings, lists, tables, etc.) by listing their positions. Provide a brief summary in 200 words or less. Start directly with the cleaned text followed by structure elements and summary. And do not include any other text.Use proper Markdown syntax without unnecessary escape characters or formatting issues. Avoid wrapping the response in a fenced code block unless required. Ensure clean, well-structured output with appropriate headings, lists, bold, and italics for direct rendering in a Markdown viewer.
            """

        prompt = prompt_template.format(context=context_str, text=text)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts and processes text from PDF documents."},
                    {"role": "user", "content": prompt}
                ]
            )

            result = response.choices[0].message.content if hasattr(
                response, 'choices') and response.choices else None

            if not result:
                logger.warning("Received None response from LLM API")
                return text, "No content extracted"

            processed_text, summary = get_text_and_last_paragraph(result)

            if not processed_text:
                logger.error("Invalid response format from LLM API")
                return text, "No content extracted"
            if not summary:
                summary = "No summary provided"

            return processed_text, summary

        except openai.RateLimitError as e:
            msg = f"API Quota/Rate Limit Exceeded (HTTP 429) for '{self.provider_type}' model '{self.model_name}'. Switch provider or check quota in Settings."
            logger.error(msg)
            return text, msg
        except openai.AuthenticationError as e:
            msg = f"API Authentication Failed (HTTP 401) for '{self.provider_type}'. Please check your API Key in Settings."
            logger.error(msg)
            return text, msg
        except Exception as e:
            msg = f"API Error ({self.provider_type}): {str(e)}"
            logger.error(msg, exc_info=True)
            return text, f"Error generating summary: {str(e)}"


class OpenAIVisionProvider(BaseVLMProvider):
    """OpenAI-compatible Vision API provider for image processing.
    Supports OpenAI, Gemini, OpenRouter, Ollama, and custom endpoints.
    """

    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None, provider_type: str = "openai"):
        self.provider_type = provider_type.lower()
        defaults = PROVIDER_DEFAULTS.get(self.provider_type, PROVIDER_DEFAULTS["custom"])

        if not base_url and defaults["base_url"]:
            base_url = defaults["base_url"]
        if not model_name and defaults["vlm_model"]:
            model_name = defaults["vlm_model"]

        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name)

    def _get_api_key_env_var(self):
        return "VLM_API_KEY"

    def _get_base_url_env_var(self) -> str:
        return "VLM_API_BASE_URL"

    def _get_model_name_env_var(self) -> str:
        return "VLM_MODEL_NAME"

    def _initialize(self):
        try:
            key = self.api_key or os.environ.get("VLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
            url = self.base_url or None

            headers = {}
            if self.provider_type == "openrouter":
                headers = {"HTTP-Referer": "https://docusense.app", "X-Title": "DocuSense"}

            self.client = openai.OpenAI(
                api_key=key,
                base_url=url,
                default_headers=headers if headers else None
            )
            self.base64 = base64
        except Exception as e:
            logger.error(f"Failed to initialize Vision API client: {e}", exc_info=True)
            raise

    def process_image(self, image: Image.Image, context: str = None, prompt_template: str = None) -> Tuple[str, str]:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = self.base64.b64encode(buffered.getvalue()).decode()

        context_str = f"Previous context: {context}\n\n" if context else ""

        if not prompt_template:
            prompt = f"""
            {context_str} This image is a page from a PDF document.

            Extract all text visible in the image while preserving structure and layout. Describe any images, tables, charts, graphs, diagrams, or non-text elements in detail, including their content, purpose, and visual characteristics. Then provide a brief summary of the page content. Start directly with the extracted text followed by your descriptions of visual elements and your summary in next paragraph.Use proper Markdown syntax without unnecessary escape characters or formatting issues. Avoid wrapping the response in a fenced code block unless required. Ensure clean, well-structured output with appropriate headings, lists, bold, and italics for direct rendering in a Markdown viewer.
            """
        else:
            prompt = prompt_template.format(context=context_str)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )

            if response is None or not hasattr(response, 'choices') or not response.choices:
                logger.warning("Received invalid response from Vision API")
                return "", "No content extracted"

            result = response.choices[0].message.content

            if not result:
                logger.error("Invalid response content from Vision API")
                return "", "No content extracted"

            processed_text, summary = get_text_and_last_paragraph(result)

            if not processed_text:
                logger.error("Invalid response format from Vision API")
                return "", "No content extracted"
            if not summary:
                summary = "No summary provided"

            return processed_text, summary

        except openai.RateLimitError as e:
            msg = f"API Quota/Rate Limit Exceeded (HTTP 429) for '{self.provider_type}' model '{self.model_name}'. Switch provider or check quota in Settings."
            logger.error(msg)
            return "", msg
        except openai.AuthenticationError as e:
            msg = f"API Authentication Failed (HTTP 401) for '{self.provider_type}'. Please check your API Key in Settings."
            logger.error(msg)
            return "", msg
        except Exception as e:
            msg = f"Vision API Error ({self.provider_type}): {str(e)}"
            logger.error(msg, exc_info=True)
            return "", f"Error: {str(e)}"


class GeminiLLMProvider(OpenAIProvider):
    """Google Gemini API LLM Provider"""
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name, provider_type="gemini")


class GeminiVLMProvider(OpenAIVisionProvider):
    """Google Gemini API Vision Provider"""
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name, provider_type="gemini")


class OpenRouterLLMProvider(OpenAIProvider):
    """OpenRouter LLM Provider"""
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name, provider_type="openrouter")


class OpenRouterVLMProvider(OpenAIVisionProvider):
    """OpenRouter Vision Provider"""
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name, provider_type="openrouter")


class OllamaLLMProvider(OpenAIProvider):
    """Ollama Local LLM Provider"""
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name, provider_type="ollama")


class OllamaVLMProvider(OpenAIVisionProvider):
    """Ollama Local Vision Provider"""
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name, provider_type="ollama")


def create_llm_provider(
    provider_type: str = "gemini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None
) -> BaseLLMProvider:
    """Factory to build an LLM provider based on provider name and configuration."""
    provider_type = provider_type.lower()
    return OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        provider_type=provider_type
    )


def create_vlm_provider(
    provider_type: str = "gemini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None
) -> BaseVLMProvider:
    """Factory to build a VLM provider based on provider name and configuration."""
    provider_type = provider_type.lower()
    return OpenAIVisionProvider(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        provider_type=provider_type
    )
