#Response from dataagent
#DataProfile(task_type='classification', num_samples=52, num_classes=2, data_quality='good', column_info={'text': {'type': 'object', 'unique_values': 52, 'sample_values': ["This product is absolutely amazing! Best purchase I've ever made.", 'Terrible quality. Broke after one day. Would not recommend.', 'The delivery was fast and the item works as expected.']}, 'label': {'type': 'object', 'unique_values': 2, 'sample_values': ['positive', 'negative', 'positive']}}, preprocessing_needed=['text_normalization', 'tokenization', 'remove_stop_words'], confidence=0.9)
"""
Data Agent - Analyzes user data and determines training requirements
Uses LLM to intelligently understand data structure and quality.
"""
import sys
import pdb
import os 
import pandas as pd
import re
from typing import Dict, Any, List
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.client import LLMClient
from llm.prompts import DATA_ANALYSIS_PROMPT

@dataclass
class DataProfile:
    """Profile of user's training data."""
    task_type: str          # "classification", "regression", "generation"
    num_samples: int        # Number of training examples
    num_classes: int        # For classification tasks
    data_quality: str       # "excellent", "good", "needs_cleaning"
    column_info: Dict       # Column names and types
    preprocessing_needed: List[str]  # Required preprocessing steps
    confidence: float       # Agent's confidence in analysis


class DataAgent:
    """
    Agent that analyzes user data using LLM intelligence.
    
    Determines:
    - What type of ML task this is
    - Data quality and preprocessing needs
    - Optimal training configuration
    """
    
    def __init__(self):
        self.llm = LLMClient()
        self.name = "DataAgent"
    
    def analyze_data(self, data_path: str, user_prompt: str) -> DataProfile:
        """
        Analyze user's training data and determine requirements.
        
        Args:
            data_path: Path to user's CSV file
            user_prompt: User's description of what they want
            
        Returns:
            DataProfile with analysis results
        """
        # Load and examine the data
        df = pd.read_csv(data_path)
        basic_stats = self._get_basic_stats(df)
        
        # Use LLM to intelligently analyze the data
        llm_analysis = self._llm_analyze_data(df, user_prompt, basic_stats)
        #pdb.set_trace()
        # Combine statistical analysis with LLM insights
        return DataProfile(
            task_type=llm_analysis["task_type"],
            num_samples=len(df),
            num_classes=llm_analysis.get("num_classes", 0),
            data_quality=llm_analysis["data_quality"],
            column_info=self._get_column_info(df),
            preprocessing_needed=llm_analysis["preprocessing_steps"],
            confidence=llm_analysis["confidence"]
        )
    
    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistical information about the dataset."""
        #pdb.set_trace()
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict(),
            "unique_values": {col: df[col].nunique() for col in df.columns}
        }
    
    def _llm_analyze_data(self, df: pd.DataFrame, user_prompt: str, 
                         basic_stats: Dict) -> Dict[str, Any]:
        """Use LLM to intelligently analyze the data structure and requirements."""
        
        # Prepare data summary for LLM
        data_summary = {
            "user_request": user_prompt,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": list(df.columns),
            "sample_rows": df.head(2).to_string(),
            "data_types": basic_stats["dtypes"],
            "missing_data": basic_stats["missing_values"]
        }
        # Get LLM analysis
        analysis_prompt = DATA_ANALYSIS_PROMPT.format(**data_summary)
        llm_response = self.llm.analyze(analysis_prompt)
        #print(type(llm_response)) #str
        #pdb.set_trace()
        return self._parse_llm_response(llm_response)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data profile."""
        import json
        #pdb.set_trace()
        try:
            # LLM should return JSON format
            match = re.search(r'{[\s\S]*}', response)
            if match:
                json_str = match.group(0)
                parsed = json.loads(json_str)
                #pdb.set_trace()
                return {
                    "task_type": parsed.get("task_type", "classification"),
                    "num_classes": parsed.get("num_classes", 2),
                    "data_quality": parsed.get("data_quality", "good"),
                    "preprocessing_steps": parsed.get("preprocessing_steps", []),
                    "confidence": parsed.get("confidence", 0.8),
                    "reasoning": parsed.get("reasoning", "")
                }
            
        except json.JSONDecodeError:
            # Fallback parsing if LLM doesn't return valid JSON
            print("entered exception")
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when LLM response isn't valid JSON."""
        # Simple keyword-based parsing
        task_type = "classification"
        if "generation" in response.lower():
            task_type = "generation"
        elif "regression" in response.lower():
            task_type = "regression"
        
        return {
            "task_type": task_type,
            "num_classes": 2,
            "data_quality": "good",
            "preprocessing_steps": ["text_cleaning"],
            "confidence": 0.9,
            "reasoning": response
        }
    
    def _get_column_info(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get information about each column in the dataset."""
        return {
            col: {
                "type": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "sample_values": df[col].dropna().head(3).tolist()
            }
            for col in df.columns
        }
    
#Example usage
if __name__ == "__main__":
    agent = DataAgent()
    
    # Analyze sample data
    profile = agent.analyze_data(
        data_path="../../examples/sample_data.csv",
        user_prompt="Classify customer reviews as positive or negative"
    )
    pdb.set_trace()
    
    print(f"Task type: {profile.task_type}")
    print(f"Number of samples: {profile.num_samples}")
    print(f"Data quality: {profile.data_quality}")
    print(f"Preprocessing needed: {profile.preprocessing_needed}")
    print(f"Confidence: {profile.confidence}")