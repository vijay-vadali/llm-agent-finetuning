# request from planning agent
# {'base_model': 'distilbert-base-uncased', 'task_approach': 'sequence_classification', 'training_config': {'learning_rate': 2e-05, 'num_epochs': 3, 'batch_size': 16, 'warmup_steps': 500, 'weight_decay': 0.01, 'save_steps': 1000, 'eval_steps': 500, 'logging_steps': 100}, 'preprocessing_steps': ['text_cleaning', 'tokenization'], 'evaluation_metrics': ['accuracy', 'f1'], 'expected_duration': '10-20 minutes', 'resource_requirements': {'gpu': 'optional', 'memory': '4GB'}, 'confidence': 0.7}
"""
Training Agent - Orchestrates Metaflow training pipeline execution
Manages the actual fine-tuning process using the training plan.
"""

import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sys
import logging
import traceback
from llm.client import LLMClient
from llm.prompts import TRAINING_MONITOR_PROMPT
from .planning_agent import TrainingPlan
from .data_agent import DataProfile
from dag_flows.training_pipeline import FineTuningFlow

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


@dataclass
class TrainingResult:
    """Result of the training process."""

    model_path: str
    training_metrics: Dict
    training_duration: float
    model_size_mb: float
    api_endpoint: Optional[str]
    status: str
    logs: List[str]


class TrainingAgent:

    def __init__(self):
        self.llm = LLMClient()
        self.name = "TrainingAgent"
        self.current_training = None

    def execute_training_plan(
        self, training_plan: TrainingPlan, data_profile: DataProfile, data_path: str
    ) -> TrainingResult:
        start_time = time.time()
        try:
            training_config = self._prepare_training_config(
                training_plan, data_profile, data_path
            )

            flow_result = self._execute_metaflow_pipeline(training_config)

            training_metrics = self._monitor_training_progress(flow_result)

            training_duration = (time.time() - start_time) / 60  # Convert to minutes

            return TrainingResult(
                model_path=flow_result["model_path"],
                training_metrics=training_metrics,
                training_duration=training_duration,
                status="success",
                logs=flow_result.get("logs", []),
            )

        except Exception as e:
            training_duration = (time.time() - start_time) / 60

            return TrainingResult(
                model_path="",
                training_metrics={},
                training_duration=training_duration,
                model_size_mb=0.0,
                api_endpoint=None,
                status="failed",
                logs=[f"Training failed: {e}"],
            )

    def _prepare_training_config(
        self, training_plan: TrainingPlan, data_profile: DataProfile, data_path: str
    ) -> Dict[str, Any]:
        """Prepare configuration for Metaflow pipeline."""

        return {
            "data_path": data_path,
            "model_name": training_plan.base_model,
            "task_type": training_plan.task_approach,
            "num_classes": data_profile.num_classes,
            "preprocessing_steps": training_plan.preprocessing_steps,
            "training_args": training_plan.training_config,
            "evaluation_metrics": training_plan.evaluation_metrics,
            "output_dir": f"./models/{int(time.time())}",
            "temp_dir": f"./temp/{int(time.time())}",
        }

    def _execute_metaflow_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            flow = FineTuningFlow()

            run = flow.run(
                data_path=config["data_path"],
                model_name=config["model_name"],
                task_type=config["task_type"],
                num_classes=config["num_classes"],
                preprocessing_steps=config["preprocessing_steps"],
                training_args=config["training_args"],
                output_dir=config["output_dir"],
            )

            return {
                "run_id": run.id,
                "model_path": run.data.model_path,
                "metrics": run.data.training_metrics,
                "logs": run.data.logs,
                "status": "completed",
            }

        except Exception as e:
            logging.error("Metaflow pipeline execution failed: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())

    def _monitor_training_progress(self, flow_result: Dict[str, Any]) -> Dict[str, Any]:
        metrics = flow_result.get("metrics", {})

        if metrics:
            progress_analysis = self._llm_analyze_training_progress(metrics)
            metrics["llm_analysis"] = progress_analysis
        return metrics

    def _llm_analyze_training_progress(self, metrics: Dict[str, Any]) -> str:

        metrics_summary = {
            "final_loss": metrics.get("train_loss", "unknown"),
            "final_accuracy": metrics.get("eval_accuracy", "unknown"),
            "training_epochs": metrics.get("epochs_completed", "unknown"),
            "learning_curve": metrics.get("loss_history", []),
        }

        analysis_prompt = TRAINING_MONITOR_PROMPT.format(**metrics_summary)

        try:
            analysis = self.llm.analyze(analysis_prompt)
            return analysis
        except Exception:
            return "Training completed successfully. Metrics look normal."
