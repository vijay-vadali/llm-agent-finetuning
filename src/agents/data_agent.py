# Response from dataagent
# DataProfile(task_type='classification', num_samples=52, num_classes=2, data_quality='good', column_info={'text': {'type': 'object', 'unique_values': 52, 'sample_values': ["This product is absolutely amazing! Best purchase I've ever made.", 'Terrible quality. Broke after one day. Would not recommend.', 'The delivery was fast and the item works as expected.']}, 'label': {'type': 'object', 'unique_values': 2, 'sample_values': ['positive', 'negative', 'positive']}}, preprocessing_needed=['text_normalization', 'tokenization', 'remove_stop_words'], confidence=0.9)
import sys
import os
import pandas as pd
import re
from typing import Dict, Any, List
from dataclasses import dataclass
import logging
import pdb
import traceback

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.client import LLMClient
from llm.prompts import DATA_ANALYSIS_PROMPT


@dataclass
class DataProfile:
    """Profile of user's training data."""

    task_type: str
    num_samples: int
    num_classes: int
    data_quality: str
    column_info: Dict
    preprocessing_needed: List[str]
    confidence: float


class DataAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.name = "DataAgent"

    def analyze_data(self, data_path: str, user_prompt: str) -> DataProfile:
        df = pd.read_csv(data_path)
        basic_stats = self._get_basic_stats(df)

        llm_analysis = self._llm_analyze_data(df, user_prompt, basic_stats)
        return DataProfile(
            task_type=llm_analysis["task_type"],
            num_samples=len(df),
            num_classes=llm_analysis.get("num_classes", 0),
            data_quality=llm_analysis["data_quality"],
            column_info=self._get_column_info(df),
            preprocessing_needed=llm_analysis["preprocessing_steps"],
            confidence=llm_analysis["confidence"],
        )

    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict(),
            "unique_values": {col: df[col].nunique() for col in df.columns},
        }

    def _llm_analyze_data(
        self, df: pd.DataFrame, user_prompt: str, basic_stats: Dict
    ) -> Dict[str, Any]:
        data_summary = {
            "user_request": user_prompt,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": list(df.columns),
            "sample_rows": df.head(2).to_string(),
            "data_types": basic_stats["dtypes"],
            "missing_data": basic_stats["missing_values"],
        }
        analysis_prompt = DATA_ANALYSIS_PROMPT.format(**data_summary)
        llm_response = self.llm.analyze(analysis_prompt)
        #llm_response = "task_type": "classification",\n    "num_classes": 2,\n    "data_quality": "good",\n    "preprocessing_steps": ["text cleaning", "tokenization", "lemmatization", "vectorization"],\n    "confidence": 0.85,\n    "reasoning": "Classifying customer reviews as positive or negative is a classic binary classification task. The dataset contains text data and corresponding labels, making it suitable for sentiment analysis. Preprocessing steps such as text cleaning, tokenization, lemmatization, and vectorization are needed to convert text data into a format suitable for machine learning algorithms."\n}'
        #pdb.set_trace()
        return self._parse_llm_response(llm_response)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data profile."""
        import json

        try:
            match = re.search(r"{[\s\S]*}", response)
            if match:
                json_str = match.group(0)
                parsed = json.loads(json_str)
                return {
                    "task_type": parsed.get("task_type", "classification"),
                    "num_classes": parsed.get("num_classes", 2),
                    "data_quality": parsed.get("data_quality", "good"),
                    "preprocessing_steps": parsed.get("preprocessing_steps", []),
                    "confidence": parsed.get("confidence", 0.8),
                    "reasoning": parsed.get("reasoning", ""),
                }
        except Exception as e:
            logging.error("An error occurred: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())

    def _get_column_info(self, df: pd.DataFrame) -> Dict[str, str]:
        return {
            col: {
                "type": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "sample_values": df[col].dropna().head(3).tolist(),
            }
            for col in df.columns
        }
