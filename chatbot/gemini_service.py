"""
Google Gemini AI Service Module.

Provides integration with Google's Generative AI REST API using standard
HTTP requests, eliminating heavy compilation/grpc dependencies while delivering
reliable multi-domain question answering.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple
import requests

logger = logging.getLogger("GeminiService")

GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-pro",
    "gemini-3.8-flash",
]
DEFAULT_MODEL = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = (
    "You are an intelligent, friendly, and helpful universal AI assistant. "
    "You can answer questions across every domain: science, programming, history, "
    "mathematics, arts, campus life, career guidance, language, and everyday inquiries. "
    "Keep answers informative, well-structured, clear, and polite. "
    "Use markdown formatting with bullet points or code blocks where appropriate."
)


class GeminiService:
    """
    Handles communication with Google's Gemini API.
    Supports environment keys and per-request/client keys.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = 15,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout

    def set_api_key(self, api_key: str) -> None:
        """Update the active API key."""
        self.api_key = api_key.strip() if api_key else None

    def is_configured(self) -> bool:
        """Check if an API key is available."""
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def verify_key(self, test_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Tests if an API key is valid by sending a minimal generation request.
        Returns (is_valid, message).
        """
        key_to_test = (test_key or self.api_key or "").strip()
        if not key_to_test:
            return False, "No Google API key provided."

        last_error = ""
        for m in DEFAULT_MODELS:
            url = GEMINI_API_ENDPOINT.format(model=m)
            params = {"key": key_to_test}
            payload = {
                "contents": [{"parts": [{"text": "Hello"}]}],
                "generationConfig": {"maxOutputTokens": 5, "temperature": 0.0}
            }

            try:
                resp = requests.post(
                    url,
                    params=params,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=8
                )

                if resp.status_code == 200:
                    self.model = m
                    return True, f"Google API Key is valid and active (using {m})."

                err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                err_msg = err_data.get("error", {}).get("message", f"HTTP {resp.status_code}")

                # If key is invalid, fail immediately
                if "API key not valid" in err_msg or "expired" in err_msg or resp.status_code in (401, 403):
                    return False, f"Google API Error: {err_msg}"

                last_error = err_msg

            except requests.exceptions.RequestException as e:
                last_error = str(e)

        return False, f"Google API Error: {last_error}"

    def generate_answer(
        self,
        prompt: str,
        api_key_override: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer for a user prompt across any domain.

        :param prompt: User's question or message.
        :param api_key_override: Optional per-request API key.
        :param context: Optional background context (e.g. from local FAQ).
        :return: Dict containing success, answer, model, or error.
        """
        active_key = (api_key_override or self.api_key or "").strip()
        if not active_key:
            return {
                "success": False,
                "error": "Google API key is not configured.",
                "needs_key": True
            }

        # Build prompt parts
        user_text = prompt.strip()
        if context:
            user_text = f"Context from local knowledge base:\n{context}\n\nUser Question:\n{user_text}"

        payload = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_text}]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }

        # Try active model first, then fallback to other models if model is not found
        candidate_models = [self.model] + [m for m in DEFAULT_MODELS if m != self.model]
        last_error = ""
        is_auth_error = False

        for model_name in candidate_models:
            url = GEMINI_API_ENDPOINT.format(model=model_name)
            params = {"key": active_key}

            try:
                resp = requests.post(
                    url,
                    params=params,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )

                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            self.model = model_name  # Remember working model
                            return {
                                "success": True,
                                "answer": parts[0]["text"].strip(),
                                "model": model_name,
                                "source": "Google Gemini AI"
                            }

                # Parse error
                try:
                    err_data = resp.json()
                    err_message = err_data.get("error", {}).get("message", f"HTTP {resp.status_code}")
                except Exception:
                    err_message = resp.text[:200]

                if "API key" in err_message or resp.status_code in (401, 403):
                    is_auth_error = True
                    last_error = err_message
                    break

                # If 404 (model deprecated/not supported on this key), try next model
                if resp.status_code == 404 or "not found" in err_message.lower():
                    last_error = err_message
                    continue

                last_error = err_message

            except requests.exceptions.Timeout:
                last_error = "Google Gemini API request timed out."
            except requests.exceptions.RequestException as e:
                last_error = f"Network connection error: {str(e)}"

        return {
            "success": False,
            "error": f"Google Gemini Error: {last_error}",
            "needs_key": is_auth_error
        }
