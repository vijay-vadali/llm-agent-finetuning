"""
LLM Client - Centralized interface for all LLM interactions
Handles OpenAI API calls with error handling and response parsing.
"""
import pdb
import os
import sys
import time
from typing import Dict, Any, List, Optional
import openai
from dataclasses import dataclass
import anthropic

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    api_key: str
    #model: str = "gpt-4"
    model:str ="claude-3-sonnet-20240229"
    max_tokens: int = 1024
    temperature: float = 0.3
    timeout: int = 30


class LLMClient:
    """
    Centralized LLM client for all agent interactions.
    
    Provides methods for different types of LLM tasks:
    - Data analysis
    - Training planning
    - Progress monitoring
    - Error analysis
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM client with configuration."""
        if config is None:
            config = LLMConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),#GPT‑4
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
            )
            # config = LLMConfig(
            #     api_key=os.getenv("CLAUDE_API_KEY"),
            #     model=os.getenv("CLAUDE_MODEL", "gpt-4"),
            #     max_tokens=int(os.getenv("CLAUDE_MAX_TOKENS", "1000")),
            #     temperature=float(os.getenv("CLAUDE_TEMPERATURE", "0.3"))
            # )
        
        self.config = config
        self.client = openai.OpenAI(api_key=config.api_key)
        #self.client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY")  # Assuming config.api_key holds your Claude key
        # Track usage for monitoring
        #self.total_tokens_used = 0
        #self.requests_made = 0
    
    def analyze(self, prompt: str, system_message: str = None) -> str:
        """
        General analysis method for data understanding.
        
        Args:
            prompt: The analysis prompt
            system_message: Optional system context
            
        Returns:
            LLM analysis response
        """
        return self._make_request(
            prompt=prompt,
            system_message=system_message or "You are an expert data scientist analyzing datasets for machine learning tasks.",
            function_name="analyze"
        )
    
    def plan(self, prompt: str, system_message: str = None) -> str:
        """
        Planning method for creating training strategies.
        
        Args:
            prompt: The planning prompt
            system_message: Optional system context
            
        Returns:
            LLM planning response
        """
        #print("entered plan")
        return self._make_request(
            prompt=prompt,
            system_message=system_message or "You are an expert ML engineer creating optimal training plans for fine-tuning language models.",
            function_name="plan"
        )
    
    # def monitor(self, prompt: str, system_message: str = None) -> str:
    #     """
    #     Monitoring method for training progress analysis.
        
    #     Args:
    #         prompt: The monitoring prompt
    #         system_message: Optional system context
            
    #     Returns:
    #         LLM monitoring response
    #     """
    #     return self._make_request(
    #         prompt=prompt,
    #         system_message=system_message or "You are an expert ML engineer monitoring training progress and providing insights.",
    #         function_name="monitor"
    #     )
    
    def debug(self, prompt: str, system_message: str = None) -> str:
        """
        Debugging method for error analysis and troubleshooting.
        
        Args:
            prompt: The debugging prompt
            system_message: Optional system context
            
        Returns:
            LLM debugging response
        """
        return self._make_request(
            prompt=prompt,
            system_message=system_message or "You are an expert ML engineer helping debug training issues and suggesting practical solutions.",
            function_name="debug"
        )
    
    def _make_request(self, prompt: str, system_message: str, function_name: str) -> str:
        """
        Make a request to the OpenAI API with error handling.
        
        Args:
            prompt: User prompt
            system_message: System context
            function_name: Name of calling function for logging
            
        Returns:
            LLM response text
        """
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Make OPEN API request
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            # Make Claude API request
            # response = self.client.messages.create(
            #     model       = self.config.model,        # "claude-3-sonnet-20240229"
            #     system      ="You are a helpful AI assistant.",
            #     messages    = messages,                 # list[dict] just like OpenAI
            #     max_tokens  = self.config.max_tokens,   # 1024
            #     temperature = self.config.temperature,  # 0.3
            #     timeout     = self.config.timeout       # 30  (handled by httpx client)
            # )
            # Update usage tracking
            # self.requests_made += 1
            # if hasattr(response, 'usage'):
            #     self.total_tokens_used += response.usage.total_tokens
            
            # Extract and return response
            response_text = response.choices[0].message.content.strip()
            
            # Log the interaction (in production, use proper logging)
            # self._log_interaction(function_name, prompt[:100], response_text[:100])
            #pdb.set_trace()
            print(response_text)
            #pdb.set_trace()
            return response_text
            
        # except openai.RateLimitError:
        #     # Handle rate limiting with exponential backoff
        #     return self._handle_rate_limit(prompt, system_message, function_name)
            
        except openai.APIError as e:
            # Handle API errors gracefully
            error_msg = f"OpenAI API error in {function_name}: {e}"
            print(f"Warning: {error_msg}")
            return error_msg
            return self._get_fallback_response(function_name, prompt)
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error in {function_name}: {e}"
            print(f"Warning: {error_msg}")
            return self._get_fallback_response(function_name, prompt)
    
    # def _handle_rate_limit(self, prompt: str, system_message: str, function_name: str, retry_count: int = 0) -> str:
    #     """Handle rate limiting with exponential backoff."""
        
    #     if retry_count >= 3:
    #         print("Max retries reached for rate limiting")
    #         return self._get_fallback_response(function_name, prompt)
        
    #     # Exponential backoff: 1, 2, 4 seconds
    #     wait_time = 2 ** retry_count
    #     print(f"Rate limited. Waiting {wait_time} seconds before retry...")
    #     time.sleep(wait_time)
        
    #     # Retry the request
    #     return self._make_request(prompt, system_message, function_name)
    
    def _get_fallback_response(self, function_name: str, prompt: str) -> str:
        """Provide fallback responses when LLM is unavailable."""
        
        fallback_responses = {
            "analyze": """{
                "task_type": "classification",
                "data_quality": "good",
                "preprocessing_steps": ["text_cleaning"],
                "confidence": 0.7
            }""",
            
            "plan": """{
                "base_model": "distilbert-base-uncased",
                "task_approach": "sequence_classification",
                "training_config": {
                    "learning_rate": 2e-5,
                    "num_epochs": 3,
                    "batch_size": 16
                },
                "preprocessing_steps": ["text_cleaning"],
                "evaluation_metrics": ["accuracy", "f1"],
                "expected_duration": "15 minutes",
                "confidence": 0.7
            }""",
            
            "monitor": "Training appears to be progressing normally. Monitor loss reduction and accuracy improvements.",
            
            "debug": "Common solutions: 1) Reduce batch size, 2) Use smaller model, 3) Check data format, 4) Ensure sufficient memory."
        }
        
        return fallback_responses.get(function_name, "Unable to process request at this time.")
    
    # def _log_interaction(self, function_name: str, prompt_preview: str, response_preview: str):
    #     """Log LLM interactions for monitoring and debugging."""
        
    #     log_entry = {
    #         "timestamp": time.time(),
    #         "function": function_name,
    #         "prompt_preview": prompt_preview + "..." if len(prompt_preview) >= 100 else prompt_preview,
    #         "response_preview": response_preview + "..." if len(response_preview) >= 100 else response_preview,
    #         "tokens_used": self.total_tokens_used,
    #         "requests_made": self.requests_made
    #     }
        
    #     # In production, would use proper logging framework
    #     print(f"LLM {function_name}: {prompt_preview[:50]}... -> {response_preview[:50]}...")
    
    # def get_usage_stats(self) -> Dict[str, Any]:
    #     """Get usage statistics for monitoring costs and performance."""
        
    #     return {
    #         "total_tokens_used": self.total_tokens_used,
    #         "requests_made": self.requests_made,
    #         "model": self.config.model,
    #         "average_tokens_per_request": self.total_tokens_used / max(self.requests_made, 1)
    #     }
    
    def validate_connection(self) -> bool:
        """Test if LLM connection is working."""
        
        try:
            test_response = self._make_request(
                prompt="Say 'OK' if you can hear me.",
                system_message="You are a helpful assistant.",
                function_name="test"
            )
            #pdb.set_trace()
            return "OK" in test_response or "ok" in test_response.lower()
            
        except Exception:
            return False
    
    def set_model(self, model_name: str):
        """Change the model being used."""
        self.config.model = model_name
        print(f"Switched to model: {model_name}")
    
    def set_temperature(self, temperature: float):
        """Change the temperature for responses."""
        self.config.temperature = max(0.0, min(1.0, temperature))
        print(f"Set temperature to: {self.config.temperature}")


# Utility functions for easy access
def create_llm_client(api_key: str = None, model: str = "gpt-4") -> LLMClient:
    """Create and return a configured LLM client."""
    
    config = LLMConfig(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        model=model
    )
    
    return LLMClient(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the LLM client
    client = LLMClient()
    
    # Test connection
    if client.validate_connection():
        print("✅ LLM connection successful")
        
        # Test different methods
        analysis = client.analyze("Analyze this dataset with 1000 customer reviews for sentiment classification.")
        print(f"Analysis result: {analysis[:100]}...")
        
        # Check usage
        # stats = client.get_usage_stats()
        # print(f"Usage stats: {stats}")
        
    else:
        print("❌ LLM connection failed - check API key")