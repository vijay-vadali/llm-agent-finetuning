"""
LLM-Driven Orchestrator - Uses LLM to intelligently coordinate agents
The LLM acts as the "brain" that decides which agents to call and how.
"""

import json
import pdb
import os
import sys
import re
import traceback
from typing import Dict, Any
import logging
import traceback

logging.basicConfig(level=print)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .llm.client import LLMClient
from .agents.data_agent import DataAgent
from .agents.planning_agent import PlanningAgent
from .agents.training_agent import TrainingAgent
from .agents.metrics_agent import MetricsAgent
from .agents.retrain_agent import RetrainAgent

# from llm.client import LLMClient
# from agents.data_agent import DataAgent
# from agents.planning_agent import PlanningAgent
# from agents.training_agent import TrainingAgent

from llm.prompts import (
    STRATEGY_PROMPT,
)


class LLMOrchestrator:
    def __init__(self):
        self.llm = LLMClient()
        self.metrics_agent = MetricsAgent()
        self.retrain_agent = RetrainAgent()

        self.agents = {
            "data_agent": DataAgent(),
            "planning_agent": PlanningAgent(),
            "training_agent": TrainingAgent(),
        }

    def process_user_request(self, user_prompt: str, data_path: str, model_path: str) -> Dict[str, Any]:

        try:

            if self._is_metrics_query(user_prompt):
                print(f"\n📊 METRICS QUERY DETECTED")
                return self._handle_metrics_query(user_prompt, model_path)

            elif self._is_retrain_request(user_prompt):
                print(f"\n🔄 RETRAIN REQUEST DETECTED")
                return self._handle_retrain_request(user_prompt)

            else:
                execution_strategy = self._llm_create_strategy(user_prompt, data_path)
                #execution_strategy response
                # {'user_intent': 'classification task for customer emails', 'approach': 'standard_pipeline', 'reasoning': 'This strategy follows a standard pipeline of data analysis, planning, and training for classification tasks.', 'execution_steps': [{'step': 1, 'agent': 'data_agent', 'action': 'analyze_data', 'params': {'data_path': '/Users/vvadali/Documents/git/vadaliv/llm-agent-finetuning/examples/sample_data.csv', 'user_prompt': 'Classify customer reviews as positive or negative'}, 'expected_output': 'data profile with task type'}, {'step': 2, 'agent': 'planning_agent', 'action': 'create_training_plan', 'params': {'data_profile': 'from_step_1', 'user_prompt': 'Classify customer reviews as positive or negative'}, 'expected_output': 'training strategy'}, {'step': 3, 'agent': 'training_agent', 'action': 'execute_training_plan', 'params': {'data_profile': 'from_step_1', 'training_plan': 'from_step_2', 'data_path': '/Users/vvadali/Documents/git/vadaliv/llm-agent-finetuning/examples/sample_data.csv'}, 'expected_output': 'trained model for classifying customer reviews'}], 'success_criteria': 'Achieving a high accuracy in classifying customer reviews as positive or negative.', 'fallback_strategy': 'If the training process fails, reevaluate the data quality and adjust the training plan accordingly.'}
                results = self._execute_strategy(
                    execution_strategy, user_prompt, data_path
                )
                return {
                    "status": "success",
                    "user_prompt": user_prompt,
                    "llm_strategy": execution_strategy,
                    "agent_results": results,
                }

        except Exception as e:
            logging.error("❌ Orchestration failed: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())

            return {"status": "failed", "error": str(e)}

    def _is_metrics_query(self, user_prompt: str) -> bool:
        user_prompt = user_prompt.lower()

        metrics_keywords = [
            "f1 score",
            "f1",
            "accuracy",
            "performance",
            "metrics",
            "how good",
            "what is the",
            "show me the",
        ]

        return any(keyword in user_prompt for keyword in metrics_keywords)

    def _handle_metrics_query(self, user_prompt: str, model_path: str) -> Dict[str, Any]:

        metrics_response = self.metrics_agent.handle_metrics_query(
            user_prompt, model_path
        )
        return metrics_response

        # return {
        #     "status": "success",
        #     "user_prompt": user_prompt,
        #     "query_type": "metrics",
        #     "metrics_response": metrics_response,
        # }

    def _is_retrain_request(self, user_prompt: str) -> bool:
        user_prompt = user_prompt.lower()

        retrain_keywords = [
            "retrain",
            "re-train",
            "train again",
            "retrain with",
            "change epochs",
            "use different epochs",
            "try with",
            "re train",
        ]

        return any(keyword in user_prompt for keyword in retrain_keywords)

    def _handle_retrain_request(self, user_prompt: str) -> Dict[str, Any]:

        retrain_result = self.retrain_agent.handle_retrain(
            user_prompt, self.metrics_agent.current_model_path
        )

        if retrain_result.get("status") == "success":
            new_model_path = retrain_result.get("new_model_path")
            if new_model_path:
                self.metrics_agent.set_current_model(new_model_path)

        return {
            "status": retrain_result.get("status", "failed"),
            "user_prompt": user_prompt,
            "query_type": "retrain",
            "retrain_response": retrain_result.get("message", "Retraining failed"),
        }

    def _update_current_model(self, training_results: Dict[str, Any]):
        """Update metrics agent with latest trained model info"""

        if not hasattr(self, "metrics_agent"):
            self.metrics_agent = MetricsAgent()

        model_path = training_results.get("model_path")
        if model_path:
            self.metrics_agent.set_current_model(model_path)
            print(f"✅ Updated metrics agent with model: {model_path}")

    def _llm_create_strategy(self, user_prompt: str, data_path: str) -> Dict[str, Any]:
        data_summary = {"user_prompt": user_prompt, "data_path": data_path}
        DATA_STRATEGY_PROMPT = STRATEGY_PROMPT.format(**data_summary)
        strategy_response = self.llm.plan(DATA_STRATEGY_PROMPT)
        try:
            return self.parse_orchestrator_response_regex(strategy_response)
        except json.JSONDecodeError as e:
            logging.error("Strategy parsing failed: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())
            return {
                "status": "failed",
                "error": f"Invalid JSON: {str(e)}",
                "raw_response": strategy_response[:200],
            }

    def _execute_strategy(
        self, strategy: Dict, user_prompt: str, data_path: str
    ) -> Dict[str, Any]:
        results = {}
        agent_outputs = {}
        #pdb.set_trace()
        for step in strategy["execution_steps"]:
            step_num = step["step"] #1,2
            agent_name = step["agent"] #data_agent,planning_agent
            action = step["action"] #analyze_data,create_training_plan
            params = step["params"] #{'data_path': 'sample_data.csv', 'user_prompt': 'Classify customer reviews as positive or negative'}
            print(f"\n🎯 Step {step_num}: Calling {agent_name}.{action}")
            try:
                actual_params = self._prepare_params(
                    params, agent_outputs, user_prompt, data_path
                )

                agent = self.agents[agent_name]
                method = getattr(agent, action)

                print(
                    f"   📞 Calling: {agent_name}.{action}({list(actual_params.keys())})"
                )
                result = method(**actual_params)
                results[f"step_{step_num}_{agent_name}"] = result
                agent_outputs[f"step_{step_num}"] = result
                print(f"   ✅ {agent_name} completed successfully")
            except Exception as e:
                logging.error("   ❌ {agent_name} failed: %s", str(e))
                logging.debug("Traceback:\n%s", traceback.format_exc())
        return results

    def _prepare_params(
        self,
        params: Dict,
        agent_outputs: Dict,
        user_prompt: str,
        data_path: str,
    ) -> Dict:
        actual_params = {}
        for key, value in params.items():
            if value == "user_prompt":
                actual_params[key] = user_prompt
            elif value == "data_path":
                actual_params[key] = data_path
            elif isinstance(value, str) and value.startswith("from_step_"):
                step_key = value.replace("from_", "")
                if step_key in agent_outputs:
                    actual_params[key] = agent_outputs[step_key]
                else:
                    print(f"⚠️ Warning: {value} not found in previous outputs")
            else:
                actual_params[key] = value
        return actual_params

    def parse_orchestrator_response_regex(self, response):
        if not response:
            return {"status": "failed", "error": "Empty response"}

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()
        try:
            strategy = json.loads(json_str)
            return strategy
        except json.JSONDecodeError as e:
            logging.error("Parse orchestrator response regex failed: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())
            return {
                "status": "failed",
                "error": f"JSON parse error: {str(e)}",
                "raw_response": response[:200],
            }
