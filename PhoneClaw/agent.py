"""OpenRouter VLM agent for PhoneClaw.

OpenRouter (https://openrouter.ai) provides unified access to many VLMs
(GPT-4o, Claude, Gemini, Qwen-VL, etc.) through an OpenAI-compatible API.

Key differences from the base OpenAIAgent in Android-Lab:
  - Base URL: https://openrouter.ai/api/v1
  - Auth header: Authorization: Bearer <OPENROUTER_API_KEY>
  - Extra recommended headers: HTTP-Referer, X-Title
  - Image format: standard OpenAI image_url (data URI), NOT the non-standard
    "type": "image" format used by QwenVLAgent/OpenAIAgent in Android-Lab
"""

import base64
import io
from typing import List, Dict, Any, Optional

import backoff
from openai import OpenAI
from PIL import Image


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _handle_backoff(details):
    args_str = str(details['args'])[:500]
    print(f"[OpenRouterAgent] Backing off {details['wait']:.1f}s after {details['tries']} tries. "
          f"Args: {args_str}")


def _handle_giveup(details):
    print(f"[OpenRouterAgent] Giving up after {details['tries']} tries.")


class OpenRouterAgent:
    """
    VLM agent that calls models via OpenRouter's API.

    Supports any multimodal model available on OpenRouter, e.g.:
      - openai/gpt-4o
      - anthropic/claude-3.5-sonnet
      - google/gemini-2.0-flash-exp
      - z-ai/glm-4.6v
      - meta-llama/llama-3.2-90b-vision-instruct

    Image format uses the standard OpenAI image_url (data URI) which all
    OpenRouter vision models understand.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_base: str = OPENROUTER_BASE_URL,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.9,
        site_url: str = "None",
        app_title: str = "PhoneClaw",
        max_image_width: int = 1280,
        max_image_height: int = 2800,
        **kwargs,
    ):
        """
        Args:
            api_key: OpenRouter API key (from https://openrouter.ai/keys).
            model_name: OpenRouter model identifier, e.g. "openai/gpt-4o".
            api_base: API base URL (default: https://openrouter.ai/api/v1).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = deterministic).
            top_p: Nucleus sampling parameter.
            site_url: HTTP-Referer header value (recommended by OpenRouter).
            app_title: X-Title header value (shown in OpenRouter dashboard).
            max_image_width: Images wider than this will be resized before upload.
            max_image_height: Images taller than this will be resized before upload.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height
        self.name = "OpenRouterAgent"

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_title,
            },
        )

    # ------------------------------------------------------------------
    # Core API call (with exponential backoff)
    # ------------------------------------------------------------------

    @backoff.on_exception(
        backoff.expo,
        Exception,
        on_backoff=_handle_backoff,
        on_giveup=_handle_giveup,
        max_tries=5,
    )
    def act(self, messages: List[Dict[str, Any]]) -> str:
        """
        Send messages to the model and return the response text.

        Args:
            messages: List of OpenAI-format chat messages.

        Returns:
            Model response as a string.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        content = response.choices[0].message.content
        print("-------------------------------")
        print(content)
        print("-------------------------------")
        return content

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def prompt_to_message_visual(
        self,
        prompt: str,
        img: str,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build an OpenAI-compatible user message with text + image.

        Uses the standard image_url format (data URI) accepted by all
        OpenRouter vision models.

        Args:
            prompt: Text prompt.
            img: Path to the screenshot image file.
            max_width: Override instance max_image_width.
            max_height: Override instance max_image_height.

        Returns:
            List containing one user message dict.
        """
        max_w = max_width or self.max_image_width
        max_h = max_height or self.max_image_height

        img_obj = Image.open(img).convert("RGB")
        orig_w, orig_h = img_obj.size

        if orig_w > max_w or orig_h > max_h:
            ratio = min(max_w / orig_w, max_h / orig_h)
            img_obj = img_obj.resize(
                (int(orig_w * ratio), int(orig_h * ratio)),
                Image.Resampling.LANCZOS,
            )

        buf = io.BytesIO()
        img_obj.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Standard OpenAI / OpenRouter image_url format
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]

        return [{"role": "user", "content": content}]

    def prompt_to_message_text(self, prompt: str) -> Dict[str, Any]:
        """Build a plain text user message (no image)."""
        return {"role": "user", "content": prompt}
