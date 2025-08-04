from typing import Dict, Any, List
from dag_flows.training_pipeline import FineTuningFlow
import json
from llm.client import LLMClient
import pdb
import logging
from llm.prompts import MODEL_RETRAIN_PROMPT
import traceback
import re
import datetime


logging.basicConfig(level=logging.info)


class RetrainAgent:
    def __init__(self):
        pass
        self.llm = LLMClient()

    def handle_retrain(
        self, user_prompt: str, current_model_path: str
    ) -> Dict[str, Any]:

        try:
            new_params = self._extract_retrain_params(user_prompt)

            previous_config = self._load_previous_config(current_model_path)

            retrain_config = {**previous_config, **new_params}

            logging.info(f"🔄 RETRAINING with updated config:")
            for key, value in new_params.items():
                logging.info(f"   {key}: {previous_config.get(key)} → {value}")

            new_model_path = self._execute_retrain(retrain_config, current_model_path)

            return {
                "status": "success",
                "message": f"Retraining completed! Updated: {', '.join(new_params.keys())}",
                "new_model_path": new_model_path,
                "updated_params": new_params,
            }

        except Exception as e:
            logging.error("An error occurred: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())
            return {"status": "failed", "message": f"Retraining failed: {str(e)}"}

    def _extract_retrain_params(self, user_prompt: str) -> Dict[str, Any]:
        data_summary = {"user_prompt": user_prompt}
        try:
            retrain_prompt = MODEL_RETRAIN_PROMPT.format(**data_summary)
            llm_response = self.llm.retrain(retrain_prompt)
            return self._parse_retrain_llm_response(llm_response)
        except Exception as e:
            logging.error("An error occurred: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())

    def _parse_retrain_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data profile."""

        # pdb.set_trace()
        try:
            match = re.search(r"{[\s\S]*}", response)
            if match:
                json_str = match.group(0)
                parsed = json.loads(json_str)
                return {
                    "num_epochs": parsed.get("num_epochs", "3"),
                    "learning_rate": parsed.get("learning_rate", 2e-5),
                    "batch_size": parsed.get("batch_size", "16"),
                }
        except Exception as e:
            logging.error("An error occurred: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())

    def _load_previous_config(self, model_path: str) -> Dict[str, Any]:
        """Load configuration from previous training"""
        try:
            config_path = f"{model_path}/config.json"
            with open(config_path, "r") as f:
                return json.load(f)
        except:
            return {"num_epochs": 3, "learning_rate": 2e-5, "batch_size": 16}

    def _execute_retrain(self, config: Dict[str, Any], base_model_path: str) -> str:

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_output_dir = f"{base_model_path}_retrain_{timestamp}"

        training_flow = FineTuningFlow()
        training_flow.num_epochs = config["num_epochs"]
        training_flow.learning_rate = config["learning_rate"]
        training_flow.batch_size = config["batch_size"]
        # training_flow.dataset = config['dataset']
        training_flow.output_dir = new_output_dir

        # Execute training
        training_flow.run()

        # Save new config
        with open(f"{new_output_dir}/config.json", "w") as f:
            json.dump(config, f)

        return new_output_dir
