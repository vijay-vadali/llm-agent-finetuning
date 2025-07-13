"""
LLM-Driven Orchestrator - Uses LLM to intelligently coordinate agents
The LLM acts as the "brain" that decides which agents to call and how.
"""

import json
import pdb
import time
import os 
import sys
import re
import traceback
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from .llm.client import LLMClient
# from .agents.data_agent import DataAgent
# from .agents.planning_agent import PlanningAgent
# from .agents.training_agent import TrainingAgent

from llm.client import LLMClient
from agents.data_agent import DataAgent
from agents.planning_agent import PlanningAgent
from agents.training_agent import TrainingAgent
from llm.prompts import DATA_ANALYSIS_PROMPT,STRATEGY_PROMPT,ADAPTATION_PROMPT,FAILURE_PROMPT,SYNTHESIS_PROMPT



class LLMOrchestrator:
    """
    REAL LLM-driven orchestrator that intelligently coordinates agents.
    
    The LLM analyzes user requests and decides:
    - Which agents to call
    - In what order
    - What parameters to pass
    - How to adapt based on results
    """
    
    def __init__(self):
        self.llm = LLMClient()
        
        # Initialize all available agents
        self.agents = {
            "data_agent": DataAgent(),
            "planning_agent": PlanningAgent(),
            "training_agent": TrainingAgent()
        }
        
        # Track execution history
        self.execution_history = []
        
    def process_user_request(self, user_prompt: str, data_path: str, user_id: str = "user") -> Dict[str, Any]:
        """
        MAIN ENTRY POINT: LLM analyzes user request and orchestrates agents.
        
        Args:
            user_prompt: What the user wants to accomplish
            data_path: Path to their training data
            user_id: Unique identifier
            
        Returns:
            Complete results from agent orchestration
        """
        start_time = time.time()
        
        print(f"\n🚀 LLM ORCHESTRATOR: Processing request")
        print(f"   User: '{user_prompt}'")
        print(f"   Data: {data_path}")
        
        try:
            #print(f"Entered process_user_request")
            # Step 1: LLM analyzes the request and creates execution strategy
            execution_strategy = self._llm_create_strategy(user_prompt, data_path)
            #print(f"\n🧠 LLM STRATEGY: {execution_strategy['approach']}")
            
            # Step 2: Execute the strategy using agents
            results = self._execute_strategy(execution_strategy, user_prompt, data_path, user_id)
            
            # Step 3: LLM synthesizes final response
            final_response = self._llm_synthesize_results(user_prompt, results, execution_strategy)
            
            total_time = (time.time() - start_time) / 60
            #print(f"Closed process_user_request")
            return {
                "status": "success",
                "user_prompt": user_prompt,
                "llm_strategy": execution_strategy,
                "agent_results": results,
                "final_response": final_response,
                "execution_time_minutes": total_time
            }
            
        except Exception as e:
            total_time = (time.time() - start_time) / 60
            
            print(f"❌ Orchestration failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time_minutes": total_time
                #"raw_response": final_response[:200]
            }
    
    def _llm_create_strategy(self, user_prompt: str, data_path: str) -> Dict[str, Any]:
        """
        LLM analyzes user request and creates intelligent execution strategy.
        """
        
        print(f"🤖 Asking LLM to create execution strategy...")
        data_summary = {
                "user_prompt": user_prompt,
                "data_path": data_path
        }
        #print("data_summary",data_summary)
        DATA_STRATEGY_PROMPT = STRATEGY_PROMPT.format(**data_summary)
        #print("DATA_STRATEGY_PROMPT",DATA_STRATEGY_PROMPT)
        strategy_response = self.llm.plan(DATA_STRATEGY_PROMPT)
        #pdb.set_trace()
        #print(strategy_response)
        try:
            # match = re.search(r'{[\s\S]*}', strategy_response)
            # pdb.set_trace()
            # if match:
            #     json_str = match.group(0)
            #     strategy = json.loads(json_str)
            # #strategy = json.loads(strategy_response)
            # print("strategy",strategy)
            return self.parse_orchestrator_response_regex(strategy_response)
            #print(f"✅ LLM Strategy: {strategy.get('approach', 'unknown')}")
            #return strategy
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Strategy parsing failed: {e}")
            #return self._default_strategy()
            return {
            "status": "failed",
            "error": f"Invalid JSON: {str(e)}",
            "raw_response": strategy_response[:200]  # First 200 chars for debugging
        }
    
    def _execute_strategy(self, strategy: Dict, user_prompt: str, data_path: str, user_id: str) -> Dict[str, Any]:
        """
        Execute the LLM-created strategy by calling agents in sequence.
        """
        #pdb.set_trace()
        results = {}
        agent_outputs = {}  # Store outputs to pass between agents
        
        print(f"\n⚙️ EXECUTING STRATEGY: {len(strategy['execution_steps'])} steps")
        #print(strategy['execution_steps'])
        for step in strategy["execution_steps"]:
            step_num = step["step"]
            agent_name = step["agent"]
            action = step["action"]
            params = step["params"]
            #print("params",params)
            print(f"\n🎯 Step {step_num}: Calling {agent_name}.{action}")
            
            try:
                # Prepare parameters (substitute outputs from previous steps)
                #pdb.set_trace()
                actual_params = self._prepare_params(params, agent_outputs, user_prompt, data_path, user_id)
                
                #print("actual_params from orchestrator",actual_params)
                # Call the agent
                agent = self.agents[agent_name]
                method = getattr(agent, action)
                #print("agent,method",agent,method)
                
                print(f"   📞 Calling: {agent_name}.{action}({list(actual_params.keys())})")
                result = method(**actual_params)
                #pdb.set_trace()
                # Store result for next steps
                results[f"step_{step_num}_{agent_name}"] = result
                agent_outputs[f"step_{step_num}"] = result
                
                print(f"   ✅ {agent_name} completed successfully")
                
                # LLM can adapt strategy based on intermediate results
                # if self._should_adapt_strategy(result, strategy):
                #     print(f"   🔄 LLM adapting strategy based on results...")
                #     strategy = self._adapt_strategy(strategy, result, step_num)
                #pdb.set_trace()                
            except Exception as e:
                #pdb.set_trace()
                print(f"   ❌ {agent_name} failed: {e}")
                traceback.print_exc()     
                
                # LLM decides how to handle failure
                # fallback = self._llm_handle_failure(e, step, strategy)
                # if fallback["action"] == "retry":
                #     print(f"   🔄 LLM suggests retry with: {fallback['new_params']}")
                #     # Could implement retry logic here
                # elif fallback["action"] == "skip":
                #     print(f"   ⏭️ LLM suggests skipping this step")
                #     continue
                # else:
                #     raise e
        
        return results
    
    def _prepare_params(self, params: Dict, agent_outputs: Dict, user_prompt: str, data_path: str, user_id: str) -> Dict:
        """
        Prepare actual parameters for agent calls, substituting outputs from previous steps.
        """
        
        actual_params = {}
        #print("params before _prepare_params",params)
        #print("agent_outputs before _prepare_params",agent_outputs)
        for key, value in params.items():
            if value == "user_prompt":
                actual_params[key] = user_prompt
            elif value == "data_path":
                actual_params[key] = data_path
            elif value == "user_id":
                actual_params[key] = user_id
            elif isinstance(value, str) and value.startswith("from_step_"):
                # Get output from previous step
                step_key = value.replace("from_", "")
                #pdb.set_trace()
                if step_key in agent_outputs:
                    #pdb.set_trace()
                    actual_params[key] = agent_outputs[step_key]
                else:
                    print(f"⚠️ Warning: {value} not found in previous outputs")
            else:
                actual_params[key] = value
        #print("params after _prepare_params",actual_params)
        return actual_params
    
    def _should_adapt_strategy(self, result: Any, strategy: Dict) -> bool:
        """
        LLM decides if strategy should be adapted based on intermediate results.
        """
        
        # Simple heuristic: adapt if confidence is low
        if hasattr(result, 'confidence') and result.confidence < 0.6:
            return True
        
        return False
    
    def _adapt_strategy(self, strategy: Dict, result: Any, current_step: int) -> Dict:
        """
        LLM adapts the strategy based on intermediate results.
        """
        
        try:
            adaptation_response = self.llm.plan(ADAPTATION_PROMPT)
            adaptation = json.loads(adaptation_response)
            
            # Apply the adaptation
            if adaptation["action"] == "modify_parameters":
                for step_key, changes in adaptation.get("changes", {}).items():
                    # Find and update the step
                    for step in strategy["execution_steps"]:
                        if f"step_{step['step']}" == step_key:
                            step["params"].update(changes.get("new_params", {}))
            
            return strategy
            
        except Exception as e:
            print(f"⚠️ Strategy adaptation failed: {e}")
            return strategy
    
    # def _llm_handle_failure(self, error: Exception, failed_step: Dict, strategy: Dict) -> Dict:
    #     """
    #     LLM decides how to handle agent failures.
    #     """
    #     try:
    #         failure_response = self.llm.plan(FAILURE_PROMPT)
    #         return json.loads(failure_response)
    #     except:
    #         return {"action": "abort", "reasoning": "Unable to determine recovery strategy"}
    
    def _llm_synthesize_results(self, user_prompt: str, results: Dict, strategy: Dict) -> Dict[str, Any]:
        """
        LLM synthesizes all agent results into final user-friendly response.
        """
        try:
            synthesis_response = self.llm.plan(SYNTHESIS_PROMPT)
            return json.loads(synthesis_response)
        except Exception as e:
            print(f"⚠️ Result synthesis failed: {e}")
            #return self._basic_synthesis(results)
    def parse_orchestrator_response_regex(self,response):
        """
        Parse orchestrator response using regex to extract JSON
        """
        if not response:
            return {"status": "failed", "error": "Empty response"}
        
        # Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        #pdb.set_trace()
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code blocks, assume the entire response is JSON
            json_str = response.strip()
        
        try:
            strategy = json.loads(json_str)
            print("exiting parse_orchestrator_response_regex")
            #print(strategy)
            return strategy
        except json.JSONDecodeError as e:
            return {
                "status": "failed",
                "error": f"JSON parse error: {str(e)}",
                "raw_response": response[:200]
            }
    
    # def _default_strategy(self) -> Dict[str, Any]:
    #     """Default strategy when LLM strategy creation fails."""
    #     return {
    #         "user_intent": "unknown",
    #         "approach": "standard_pipeline",
    #         "reasoning": "fallback to standard approach",
    #         "execution_steps": [
    #             {
    #                 "step": 1,
    #                 "agent": "data_agent", 
    #                 "action": "analyze_data",
    #                 "params": {"data_path": "data_path", "user_prompt": "prompt"},
    #                 "expected_output": "data profile"
    #             },
    #             {
    #                 "step": 2,
    #                 "agent": "planning_agent",
    #                 "action": "create_training_plan", 
    #                 "params": {"data_profile": "from_step_1", "user_prompt": "prompt"},
    #                 "expected_output": "training plan"
    #             },
    #             {
    #                 "step": 3,
    #                 "agent": "training_agent",
    #                 "action": "execute_training_plan",
    #                 "params": {"training_plan": "from_step_2", "data_profile": "from_step_1", "data_path": "data_path", "user_id": "user_id"},
    #                 "expected_output": "trained model"
    #             }
    #         ],
    #         "success_criteria": "trained model with >70% accuracy",
    #         "fallback_strategy": "use default parameters"
    #     }
    
    # def _basic_synthesis(self, results: Dict) -> Dict[str, Any]:
    #     """Basic result synthesis when LLM fails."""
    #     return {
    #         "summary": "Model training completed",
    #         "key_results": {"status": "completed"},
    #         "next_steps": ["Use the trained model for predictions"],
    #         "technical_details": results
    #     }


# Convenience function for simple usage
def llm_orchestrate_fine_tuning(user_prompt: str, data_path: str, user_id: str = "user") -> Dict[str, Any]:
    """
    Simple function that uses LLM to orchestrate the entire fine-tuning process.
    
    Args:
        user_prompt: What the user wants to accomplish
        data_path: Path to training data
        user_id: User identifier
        
    Returns:
        Complete orchestration results
    """
    #print("entered llm_orchestrate_fine_tuning")
    orchestrator = LLMOrchestrator()
    return orchestrator.process_user_request(user_prompt, data_path, user_id)


# Test the REAL LLM orchestrator
if __name__ == "__main__":
    import os
    
    # Test LLM orchestration
    result = llm_orchestrate_fine_tuning(
        user_prompt="Classify customer reviews as positive or negative",
        data_path = "/Users/vvadali/Documents/git/vadaliv/llm-agent-finetuning/examples/sample_data.csv",
        #data_path="examples/sample_data.csv",
        user_id="vijayvadali"
    )
    #print("result",result)
    print(f"\n🎉 LLM ORCHESTRATION COMPLETE!")
    print(f"Status: {result['status']}")
    print(f"Strategy: {result.get('llm_strategy', {}).get('approach', 'unknown')}")
    #print(f"Results: {result.get('final_response', {}).get('summary', 'No summary')}")
