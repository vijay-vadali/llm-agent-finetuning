import json

from typing import Dict


import traceback
import logging
from dataclasses import dataclass
from llm.prompts import ANALYZE_METRICS_PROMPT
from llm.client import LLMClient
import pdb

logging.basicConfig(level=logging.INFO)


@dataclass
class MetricsResponse:
    performance_level: str
    f1_score: float
    accuracy: float
    interpretation: str
    recommendations: list
    overall_assessment: str


class MetricsAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.current_model_path = None

    def handle_metrics_query(self, user_input, current_model_path):
        query_type = self._parse_metrics_query(user_input)
        #pdb.set_trace()
        if query_type == "f1_score":
            return self._get_f1_score(current_model_path)
        elif query_type == "accuracy":
            return self._get_accuracy(current_model_path)
        elif query_type == "both":
            return self._get_all_metrics(current_model_path)
        else:
            return "I can show F1 score, accuracy, or both metrics."

    def _get_f1_score(self, current_model_path):
        metrics = self._load_metrics(current_model_path)
        if metrics:
            return f"F1 Score: {metrics['f1_score']:.3f}"
        return "No metrics available. Train a model first."

    def _get_accuracy(self, current_model_path):
        
        metrics = self._load_metrics(current_model_path)
        pdb.set_trace()
        if metrics:
            return f"Accuracy: {metrics['accuracy']:.3f}"
        return "No metrics available. Train a model first."

    def _get_all_metrics(self,current_model_path):
        metrics = self._load_metrics(current_model_path)
        #pdb.set_trace()
        if metrics:
            #logging.info(f"Loaded metrics type: {type(metrics)}")
            #logging.info(f"Loaded metrics content: {metrics}")
            if isinstance(metrics, str):
                logging.error("Metrics loaded as string instead of dict. Check JSON format.")
                return "Error: Invalid metrics format."
            if not isinstance(metrics, dict):
                logging.error(f"Metrics is not a dictionary. Type: {type(metrics)}")
                return "Error: Invalid metrics format."
            try:
                data_summary = {
                    "f1_score": metrics["f1_score"],
                    "accuracy": metrics["accuracy"],
                }
                analyze_metrics_prompt = ANALYZE_METRICS_PROMPT.format(**data_summary)
                llm_response = self.llm.metrics_analyze(analyze_metrics_prompt)
                return llm_response
            except KeyError as e:
                logging.error(f"Key error accessing metrics: {e}")
                return f"Error: Missing key {e} in metrics."
            except Exception as e:
                logging.error("An error occurred: %s", str(e))
                logging.debug("Traceback:\n%s", traceback.format_exc())
                return "Error analyzing metrics."
        return "No metrics available. Train a model first."
        
    def _load_metrics(self, current_model_path):
        if not current_model_path:
            return None
        try:
            with open(f"{current_model_path}/metrics.json", "r") as f:
                return json.load(f)
        except:
            return None

    def _parse_metrics_query(self, user_input):
        user_input = user_input.lower()

        has_f1 = "f1" in user_input or "f-1" in user_input
        has_accuracy = "accuracy" in user_input

        if has_f1 and has_accuracy:
            return "both"
        elif has_f1:
            return "f1_score"
        elif has_accuracy:
            return "accuracy"
        elif "performance" in user_input or "metrics" in user_input:
            return "both"
        else:
            return "unknown"
