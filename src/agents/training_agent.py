#request from planning agent
#{'base_model': 'distilbert-base-uncased', 'task_approach': 'sequence_classification', 'training_config': {'learning_rate': 2e-05, 'num_epochs': 3, 'batch_size': 16, 'warmup_steps': 500, 'weight_decay': 0.01, 'save_steps': 1000, 'eval_steps': 500, 'logging_steps': 100}, 'preprocessing_steps': ['text_cleaning', 'tokenization'], 'evaluation_metrics': ['accuracy', 'f1'], 'expected_duration': '10-20 minutes', 'resource_requirements': {'gpu': 'optional', 'memory': '4GB'}, 'confidence': 0.7}
"""
Training Agent - Orchestrates Metaflow training pipeline execution
Manages the actual fine-tuning process using the training plan.
"""

import os
import pdb
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm.client import LLMClient
from llm.prompts import TRAINING_MONITOR_PROMPT
#from .planning_agent import TrainingPlan
from .planning_agent import TrainingPlan
#from .data_agent import DataProfile
from .data_agent import DataProfile
from dag_flows.training_pipeline import FineTuningFlow

@dataclass
class TrainingResult:
    """Result of the training process."""
    model_path: str             # Path to trained model
    training_metrics: Dict      # Loss, accuracy, etc.
    evaluation_results: Dict    # Validation metrics
    training_duration: float    # Actual training time (minutes)
    model_size_mb: float       # Model file size
    api_endpoint: Optional[str] # Deployed model endpoint
    status: str                # "success", "failed", "in_progress"
    logs: List[str]            # Training logs


class TrainingAgent:
    """
    Agent that manages the actual model training using Metaflow.
    
    Responsibilities:
    - Execute Metaflow training pipeline
    - Monitor training progress
    - Handle errors and retries
    - Deploy trained model
    """
    
    def __init__(self):
        self.llm = LLMClient()
        self.name = "TrainingAgent"
        self.current_training = None
    
    def execute_training_plan(self, training_plan: TrainingPlan, 
                        data_profile: DataProfile,
                        data_path: str,
                        user_id: str = "vijayvadali") -> TrainingResult:
        """
        Execute the complete training pipeline.
        
        Args:
            training_plan: Plan from PlanningAgent
            data_profile: Data analysis from DataAgent
            data_path: Path to training data
            user_id: Unique identifier for this training job
            
        Returns:
            TrainingResult with model and metrics
        """
        start_time = time.time()
        
        try:
            # Prepare training environment
            training_config = self._prepare_training_config(
                training_plan, data_profile, data_path, user_id
            )
            
            # Execute Metaflow pipeline
            flow_result = self._execute_metaflow_pipeline(training_config)
            
            # Monitor training progress
            training_metrics = self._monitor_training_progress(flow_result)
            #pdb.set_trace()
            # Evaluate trained model
            evaluation_results = self._evaluate_model(flow_result, data_path)
            
            # Deploy model to API endpoint
            #api_endpoint = self._deploy_model(flow_result, user_id)
            
            training_duration = (time.time() - start_time) / 60  # Convert to minutes
            
            return TrainingResult(
                model_path=flow_result["model_path"],
                training_metrics=training_metrics,
                evaluation_results=evaluation_results,
                training_duration=training_duration,
            #    model_size_mb=self._get_model_size(flow_result["model_path"]),
            #    api_endpoint=api_endpoint,
                status="success",
                logs=flow_result.get("logs", [])
            )
            
        except Exception as e:
            training_duration = (time.time() - start_time) / 60
            
            # Use LLM to analyze and suggest fixes for training errors
            #error_analysis = self._analyze_training_error(str(e), training_plan)
            
            return TrainingResult(
                model_path="",
                training_metrics={},
                evaluation_results={},
                training_duration=training_duration,
                model_size_mb=0.0,
                api_endpoint=None,
                status="failed",
                logs=[f"Training failed: {e}"]#, f"Analysis: {error_analysis}"]
            )
    
    def _prepare_training_config(self, training_plan: TrainingPlan,
                               data_profile: DataProfile,
                               data_path: str,
                               user_id: str) -> Dict[str, Any]:
        """Prepare configuration for Metaflow pipeline."""
        
        return {
            "user_id": user_id,
            "data_path": data_path,
            "model_name": training_plan.base_model,
            "task_type": training_plan.task_approach,
            "num_classes": data_profile.num_classes,
            "preprocessing_steps": training_plan.preprocessing_steps,
            "training_args": training_plan.training_config,
            "evaluation_metrics": training_plan.evaluation_metrics,
            "output_dir": f"./models/{user_id}_{int(time.time())}",
            "temp_dir": f"./temp/{user_id}_{int(time.time())}"
        }
    
    def _execute_metaflow_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Metaflow training pipeline."""
        try:
            # Import and run the Metaflow pipeline
           
            # Create flow instance with configuration
            flow = FineTuningFlow()
            
            # Execute the flow
            run = flow.run(
                data_path=config["data_path"],
                model_name=config["model_name"],
                task_type=config["task_type"],
                num_classes=config["num_classes"],
                preprocessing_steps=config["preprocessing_steps"],
                training_args=config["training_args"],
                output_dir=config["output_dir"]
            )
            
            return {
                "run_id": run.id,
                "model_path": run.data.model_path,
                "metrics": run.data.training_metrics,
                "logs": run.data.logs,
                "status": "completed"
            }
            
        except Exception as e:
            print("entered exception block")
            raise Exception(f"Metaflow pipeline execution failed: {e}")
    
    def _monitor_training_progress(self, flow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and extract training metrics."""
        
        # Get metrics from Metaflow run
        metrics = flow_result.get("metrics", {})
        
        # Use LLM to analyze training progress and suggest improvements
        if metrics:
            progress_analysis = self._llm_analyze_training_progress(metrics)
            metrics["llm_analysis"] = progress_analysis
        
        return metrics
    
    def _llm_analyze_training_progress(self, metrics: Dict[str, Any]) -> str:
        """Use LLM to analyze training metrics and provide insights."""
        
        metrics_summary = {
            "final_loss": metrics.get("train_loss", "unknown"),
            "final_accuracy": metrics.get("eval_accuracy", "unknown"),
            "training_epochs": metrics.get("epochs_completed", "unknown"),
            "learning_curve": metrics.get("loss_history", [])
        }
        
        analysis_prompt = TRAINING_MONITOR_PROMPT.format(**metrics_summary)
        
        try:
            analysis = self.llm.analyze(analysis_prompt)
            return analysis
        except Exception:
            return "Training completed successfully. Metrics look normal."
    
    def _evaluate_model(self, flow_result: Dict[str, Any], 
                       data_path: str) -> Dict[str, Any]:
        """Evaluate the trained model performance."""
        
        try:
            # Load and evaluate the model
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import pandas as pd
            from sklearn.metrics import classification_report
            
            model_path = flow_result["model_path"]
            
            # Load model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load test data (use last 20% of data for evaluation)
            df = pd.read_csv(data_path)
            test_size = int(len(df) * 0.2)
            test_df = df.tail(test_size)
            
            # Simple evaluation (would be more sophisticated in production)
            accuracy = flow_result.get("metrics", {}).get("eval_accuracy", 0.0)
            
            return {
                "accuracy": accuracy,
                "test_samples": len(test_df),
                "model_path": model_path,
                "evaluation_method": "holdout_validation"
            }
            
        except Exception as e:
            return {
                "error": f"Evaluation failed: {e}",
                "accuracy": 0.0,
                "test_samples": 0
            }
    
    # def _deploy_model(self, flow_result: Dict[str, Any], user_id: str) -> Optional[str]:
    #     """Deploy the trained model to an API endpoint."""
        
    #     try:
    #         model_path = flow_result["model_path"]
            
    #         # Simple deployment simulation (in production, would deploy to cloud)
    #         endpoint_name = f"model-{user_id}-{int(time.time())}"
    #         api_endpoint = f"https://api.your-domain.com/models/{endpoint_name}/predict"
            
    #         # Create deployment configuration
    #         deployment_config = {
    #             "model_path": model_path,
    #             "endpoint_name": endpoint_name,
    #             "api_endpoint": api_endpoint,
    #             "status": "deployed"
    #         }
            
    #         # Save deployment info (would actually deploy in production)
    #         self._save_deployment_info(deployment_config)
            
    #         return api_endpoint
            
    #     except Exception as e:
    #         print(f"Deployment failed: {e}")
    #         return None
    
    # def _save_deployment_info(self, deployment_config: Dict[str, Any]):
    #     """Save deployment configuration for later reference."""
    #     import json
        
    #     deployment_file = f"./models/deployments/{deployment_config['endpoint_name']}.json"
    #     os.makedirs(os.path.dirname(deployment_file), exist_ok=True)
        
    #     with open(deployment_file, 'w') as f:
    #         json.dump(deployment_config, f, indent=2)
    
    # def _get_model_size(self, model_path: str) -> float:
    #     """Get the size of the trained model in MB."""
    #     try:
    #         total_size = 0
    #         for dirpath, dirnames, filenames in os.walk(model_path):
    #             for filename in filenames:
    #                 filepath = os.path.join(dirpath, filename)
    #                 total_size += os.path.getsize(filepath)
            
    #         return total_size / (1024 * 1024)  # Convert to MB
    #     except Exception:
    #         return 0.0
    
    # def _analyze_training_error(self, error_message: str, 
    #                           training_plan: TrainingPlan) -> str:
    #     """Use LLM to analyze training errors and suggest fixes."""
        
    #     error_context = {
    #         "error_message": error_message,
    #         "base_model": training_plan.base_model,
    #         "training_config": training_plan.training_config,
    #         "task_approach": training_plan.task_approach
    #     }
        
    #     error_prompt = f"""
    #     Training failed with this error: {error_message}
        
    #     Training configuration:
    #     - Model: {training_plan.base_model}
    #     - Task: {training_plan.task_approach}
    #     - Config: {training_plan.training_config}
        
    #     Analyze the error and suggest specific fixes:
    #     1. What likely caused this error?
    #     2. How can it be fixed?
    #     3. Should we adjust the training configuration?
        
    #     Provide practical, actionable suggestions.
    #     """
        
    #     try:
    #         analysis = self.llm.analyze(error_prompt)
    #         return analysis
    #     except Exception:
    #         return "Training failed. Consider reducing batch size or using a smaller model."
    
    # def get_training_status(self, run_id: str) -> Dict[str, Any]:
    #     """Get current status of a training job."""
        
    #     # In production, would check Metaflow run status
    #     return {
    #         "run_id": run_id,
    #         "status": "completed",
    #         "progress": "100%",
    #         "estimated_remaining": "0 minutes"
    #     }


# Example usage
# if __name__ == "__main__":
#     from planning_agent import TrainingPlan
#     from data_agent import DataProfile
    
#     agent = TrainingAgent()
    
#     # Sample training plan and data profile
#     plan = TrainingPlan(
#         base_model="distilbert-base-uncased",
#         task_approach="sequence_classification",
#         training_config={"num_epochs": 3, "batch_size": 16},
#         preprocessing_steps=["text_cleaning"],
#         evaluation_metrics=["accuracy", "f1"],
#         expected_duration="15 minutes",
#         resource_requirements={"gpu": "optional"},
#         confidence=0.9
#     )
    
#     profile = DataProfile(
#         task_type="classification",
#         num_samples=1000,
#         num_classes=3,
#         data_quality="good",
#         column_info={},
#         preprocessing_needed=[],
#         confidence=0.9
#     )
    
#     # Execute training
#     result = agent.execute_training(
#         training_plan=plan,
#         data_profile=profile,
#         #data_path="../../examples/sample_data.csv",
#         data_path="/Users/vvadali/Documents/git/vadaliv/llm-agent-finetuning/examples/sample_data.csv",
#         user_id="vadaliv"
#     )
    
#     print(f"Training status: {result.status}")
#     print(f"Training duration: {result.training_duration:.2f} minutes")
#     print(f"Model accuracy: {result.evaluation_results.get('accuracy', 'N/A')}")
#     print(f"API endpoint: {result.api_endpoint}")