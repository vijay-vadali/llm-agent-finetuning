# Request from dataagent
# DataProfile(task_type='classification', num_samples=52, num_classes=2, data_quality='good', column_info={'text': {'type': 'object', 'unique_values': 52, 'sample_values': ["This product is absolutely amazing! Best purchase I've ever made.", 'Terrible quality. Broke after one day. Would not recommend.', 'The delivery was fast and the item works as expected.']}, 'label': {'type': 'object', 'unique_values': 2, 'sample_values': ['positive', 'negative', 'positive']}}, preprocessing_needed=['text_normalization', 'tokenization', 'remove_stop_words'], confidence=0.9)

# response from planning agent
# {'base_model': 'distilbert-base-uncased', 'task_approach': 'sequence_classification', 'training_config': {'learning_rate': 2e-05, 'num_epochs': 3, 'batch_size': 16, 'warmup_steps': 500, 'weight_decay': 0.01, 'save_steps': 1000, 'eval_steps': 500, 'logging_steps': 100}, 'preprocessing_steps': ['text_cleaning', 'tokenization'], 'evaluation_metrics': ['accuracy', 'f1'], 'resource_requirements': {'gpu': 'optional', 'memory': '4GB'}, 'confidence': 0.7}

from typing import Dict, Any, List
from dataclasses import dataclass
import os, sys
import logging
import traceback

import json
import pdb

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.client import LLMClient
from llm.prompts import PLANNING_PROMPT
from .data_agent import DataProfile

@dataclass
class TrainingPlan:
    base_model: str
    task_approach: str
    training_config: Dict
    preprocessing_steps: List
    evaluation_metrics: List
    resource_requirements: Dict
    confidence: float


class PlanningAgent:

    def __init__(self):
        self.llm = LLMClient()
        self.name = "PlanningAgent"

        self.model_catalog = {
            "text_classification": [
                "distilbert-base-uncased",
                "bert-base-uncased",
                "roberta-base",
                "albert-base-v2",
            ],
            "text_generation": ["gpt2", "distilgpt2", "microsoft/DialoGPT-medium"],
            "sentiment_analysis": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "distilbert-base-uncased-finetuned-sst-2-english",
            ],
        }

    def create_training_plan(
        self,
        data_profile: DataProfile,
        user_prompt: str,
        constraints: Dict[str, Any] = None,
    ) -> TrainingPlan:
        constraints = constraints or {}

        llm_plan = self._llm_create_plan(data_profile, user_prompt, constraints)
        validated_plan = self._validate_and_enhance_plan(llm_plan, data_profile)

        return TrainingPlan(
            base_model=validated_plan["base_model"],
            task_approach=validated_plan["task_approach"],
            training_config=validated_plan["training_config"],
            preprocessing_steps=validated_plan["preprocessing_steps"],
            evaluation_metrics=validated_plan["evaluation_metrics"],
            resource_requirements=validated_plan["resource_requirements"],
            confidence=validated_plan["confidence"],
        )

    def _llm_create_plan(
        self, data_profile: DataProfile, user_prompt: str, constraints: Dict
    ) -> Dict[str, Any]:

        planning_context = {
            "user_request": user_prompt,
            "task_type": data_profile.task_type,
            "num_samples": data_profile.num_samples,
            "num_classes": data_profile.num_classes,
            "data_quality": data_profile.data_quality,
            "preprocessing_needed": data_profile.preprocessing_needed,
            "available_models": self.model_catalog["text_classification"],
            "constraints": constraints,
        }

        planning_prompt = PLANNING_PROMPT.format(**planning_context)
        llm_response = self.llm.plan(planning_prompt)
        return self._parse_plan_response(llm_response)

    def _parse_plan_response(self, response: str) -> Dict[str, Any]:

        try:
            parsed = json.loads(response)

            return {
                "base_model": parsed.get("base_model", "distilbert-base-uncased"),
                "task_approach": parsed.get("task_approach", "sequence_classification"),
                "training_config": parsed.get(
                    "training_config", self._default_training_config()
                ),
                "preprocessing_steps": parsed.get("preprocessing_steps", []),
                "evaluation_metrics": parsed.get(
                    "evaluation_metrics", ["accuracy", "f1"]
                ),
                "resource_requirements": parsed.get(
                    "resource_requirements", {"gpu": "optional", "memory": "4GB"}
                ),
                "confidence": parsed.get("confidence", 0.8),
                "reasoning": parsed.get("reasoning", ""),
            }

        except Exception as e:
            logging.error("An error occurred: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())
            return "Parse plan response failed)"

    def _default_training_config(self) -> Dict[str, Any]:
        return {
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "batch_size": 16,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "save_steps": 1000,
            "eval_steps": 500,
            "logging_steps": 100,
        }

    def _validate_and_enhance_plan(
        self, plan: Dict, data_profile: DataProfile
    ) -> Dict[str, Any]:
        """Validate and enhance the LLM-generated plan."""

        if data_profile.num_samples < 1000:
            plan["training_config"]["batch_size"] = 8
            plan["training_config"]["num_epochs"] = 5
        elif data_profile.num_samples > 10000:
            plan["training_config"]["batch_size"] = 32
            plan["training_config"]["num_epochs"] = 2

        if data_profile.num_samples < 500:
            if plan["base_model"] == "bert-base-uncased":
                plan["base_model"] = "distilbert-base-uncased"

        if data_profile.data_quality == "needs_cleaning":
            if "text_cleaning" not in plan["preprocessing_steps"]:
                plan["preprocessing_steps"].insert(0, "text_cleaning")

        return plan