"""
LLM Client - Centralized interface for all LLM interactions
Handles OpenAI API calls with error handling and response parsing.
"""

import pdb
import os
import sys
import openai
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# import anthropic

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class LLMConfig:
    api_key: str
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 1024
    temperature: float = 0.3
    timeout: int = 30


class LLMClient:

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM client with configuration."""
        if config is None:
            config = LLMConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),  # GPT‑4
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            )

        self.config = config
        self.client = openai.OpenAI(api_key=config.api_key)

    def analyze(self, prompt: str, system_message: str = None) -> str:
        return self._make_request(
            prompt=prompt,
            system_message=system_message
            or "You are an expert data scientist analyzing datasets for machine learning tasks.",
            function_name="analyze",
        )

    def retrain(self, prompt: str, system_message: str = None) -> str:
        return self._make_request(
            prompt=prompt,
            system_message=system_message or "You're an ML training assistant.",
            function_name="retrain",
        )

    def plan(self, prompt: str, system_message: str = None) -> str:
        return self._make_request(
            prompt=prompt,
            system_message=system_message
            or "You are an expert ML engineer creating optimal training plans for fine-tuning language models.",
            function_name="plan",
        )

    def metrics_analyze(self, prompt: str, system_message: str = None) -> str:
        return self._make_request(
            prompt=prompt,
            system_message=system_message or "You are an expert ML model evaluator. ",
            function_name="metrics_analyze",
        )

    def _make_request(
        self, prompt: str, system_message: str, function_name: str
    ) -> str:
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_msg = f"OpenAI API error in {function_name}: {e}"
            print(f"Warning: {error_msg}")
            return error_msg
