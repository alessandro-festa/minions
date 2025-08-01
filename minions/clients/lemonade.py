import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

from minions.clients.openai import OpenAIClient
from minions.usage import Usage
from pydantic import BaseModel


try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

import asyncio

class LemonadeClient(OpenAIClient):
    """
    Uses Lemonade API Server to run local clients in Minion, Minions, Minions-MCP, and DeepResearch Protocols.
    Lemonade is still experimental, more protocols will be integrated soon.
    """

    def __init__(
        self,
        model_name: str = "Llama-3.2-3B-Instruct-Hybrid",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        structured_output_schema: Optional[BaseModel] = None,
        use_async: bool = False,
        local: bool = True,
        **kwargs: Any,
    ) -> None:
        base_url = base_url or os.getenv("LEMONADE_BASE_URL", "http://localhost:8000/api/v1")
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            local=local,
            **kwargs,
        )
        self.session = requests.Session()
        self.base_url = base_url
        self.logger.setLevel(logging.INFO)
        self.structured_output_schema = structured_output_schema
        self.use_async = use_async
        self.is_gguf = "GGUF" in self.model_name.upper()
        # Lemonade only supports GGUF models for structured output schemas for now
        if self.structured_output_schema and not self.is_gguf:
            raise TypeError(f"The model used for Minions and Minions-MCP must be GGUF. A GGUF model was not used.")
        # Validate Lemonade server connection and model
        self._ensure_model_available()

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Main chat method: dispatches to schat or achat depending on use_async.
        """
        if self.use_async:
            return self.achat(messages, **kwargs)
        else:
            return self.schat(messages, **kwargs)

    def schat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        Synchronous chat: used for Minion and DeepResearch.
        """
        assert len(messages) > 0, "Messages cannot be empty."
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }
        # Check if there is a structured output - for DeepResearch only
        if self.structured_output_schema:
            try:
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": self.structured_output_schema.model_json_schema()
                }
            except Exception as e:
                raise RuntimeError(f"Failed to generate schema for structured_output_schema: {e}")
            
        final_url = f"{self.base_url.rstrip('/api/v1')}/api/v1/chat/completions" if "api" in self.base_url else f"{self.base_url.rstrip('/api/v1')}/v1/chat/completions"
        
        response = self.session.post(
            final_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        response_data = response.json()
        choices = response_data.get("choices", [])
        responses = [choice["message"]["content"] for choice in choices if "message" in choice]
        usage = Usage()
        usage += Usage(
            prompt_tokens=response_data.get('usage', {}).get('prompt_tokens', 0),
            completion_tokens=response_data.get('usage', {}).get('completion_tokens', 0),
        )
        done_reason = [choice.get("finish_reason", "stop") for choice in choices]
        if self.local:
            return responses, usage, done_reason
        else:
            return responses, usage

    def achat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Parallel asynchronous chat for Lemonade.
        Accepts a list of message dicts or a single dict.
        This is only used for the Minions and Minions-MCP protocols.
        Returns (responses, usage_total, done_reasons).
        """
        if not self.use_async:
            raise RuntimeError(
                "This client is not in async mode. Set `use_async=True`."
            )
        
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for async Lemonade client. Please install with: pip install aiohttp")
        import asyncio

        # Accept both a single dict and a list of dicts
        if isinstance(messages, dict):
            messages = [messages]

        async def process_one(msg):
            payload = {
                "model": self.model_name,
                "messages": [msg],  # Each request gets its own message
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }
            if self.structured_output_schema:
                try:
                    payload["response_format"] = {
                        "type": "json_object",
                        "schema": self.structured_output_schema.model_json_schema()
                    }
                except Exception as e:
                    raise RuntimeError(f"Failed to generate schema for structured_output_schema: {e}")
            async with aiohttp.ClientSession() as session:

                final_url = f"{self.base_url.rstrip('/api/v1')}/api/v1/chat/completions" if "api" in self.base_url else f"{self.base_url.rstrip('/api/v1')}/v1/chat/completions"

                async with session.post(
                    final_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()
            choices = response_data.get("choices", [])
            content = choices[0]["message"]["content"] if choices and "message" in choices[0] else ""
            usage = Usage(
                prompt_tokens=response_data.get('usage', {}).get('prompt_tokens', 0),
                completion_tokens=response_data.get('usage', {}).get('completion_tokens', 0),
            )
            done_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
            return content, usage, done_reason

        async def run_all():
            results = await asyncio.gather(*(process_one(m) for m in messages))
            texts, usages, done_reasons = zip(*results)
            usage_total = Usage()
            for u in usages:
                usage_total += u
            if self.local:
                return list(texts), usage_total, list(done_reasons)
            else:
                return list(texts), usage_total

        # Handle event loop: support Streamlit/Jupyter
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import threading, concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.new_event_loop().run_until_complete(run_all()))
                    return future.result()
            else:
                return loop.run_until_complete(run_all())
        except RuntimeError:
            return asyncio.run(run_all())

        
    # ------------------------------------------------------------------
    # Lemonade specific helper APIs
    # ------------------------------------------------------------------
    def get_models(self) -> Dict[str, Any]:
        """Return models available on the server."""
        resp = self.session.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()

    def get_available_models(self) -> List[str]:
        """Return a list of model names available on the server."""
        models = self.get_models().get("data", [])
        return [model["id"] for model in models]
    
    def _ensure_model_available(self):
        """Ensure the specified model is available on the Lemonade server."""

        # Catch any connection issues when fetching available models
        # as that typically means the Lemonade server is not running.
        try:
            available_models = self.get_available_models()
        except requests.RequestException as e:
            msg = (f"Failed to fetch available models from Lemonade server."
                   f"Check if the Lemonade server is running")
            self.logger.error(msg)
            raise RuntimeError(msg)

        if self.model_name not in available_models:
            self.logger.info("Pulling model: %s", self.model_name)
            try:
                self.pull_model(self.model_name)
                self.logger.info(f"Successfully pulled model {self.model_name}")
            except:
                msg = (f"Model '{self.model_name}' not found on Lemonade server and unable to pull.\n"
                    f"Available models: {available_models}")
                self.logger.error(msg)
                raise RuntimeError(msg)

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Download and register a model on the server."""
        resp = self.session.post(f"{self.base_url}/pull", json={"model_name": model_name})
        resp.raise_for_status()
        return resp.json()

    def load_model(
        self,
        *,
        model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        recipe: Optional[str] = None,
        reasoning: Optional[bool] = None,
        mmproj: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explicitly load a model into memory."""
        payload: Dict[str, Any] = {}
        if model_name:
            payload["model_name"] = model_name
        if checkpoint:
            payload["checkpoint"] = checkpoint
        if recipe:
            payload["recipe"] = recipe
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if mmproj:
            payload["mmproj"] = mmproj
        resp = self.session.post(f"{self.base_url}/load", json=payload)
        resp.raise_for_status()
        return resp.json()

    def unload_model(self) -> Dict[str, Any]:
        """Unload the currently loaded model."""
        resp = self.session.post(f"{self.base_url}/unload")
        resp.raise_for_status()
        return resp.json()

    def set_params(self, **params: Any) -> Dict[str, Any]:
        """Set generation parameters that persist across requests."""
        resp = self.session.post(f"{self.base_url}/params", json=params)
        resp.raise_for_status()
        return resp.json()

    def get_health(self) -> Dict[str, Any]:
        """Check the health of the server."""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def get_stats(self) -> Dict[str, Any]:
        """Return performance statistics from the last request."""
        resp = self.session.get(f"{self.base_url}/stats")
        resp.raise_for_status()
        return resp.json()