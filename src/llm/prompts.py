"""
LLM Prompts - Template prompts for different agent tasks
"""

# Data Analysis Agent Prompts
DATA_ANALYSIS_PROMPT = """
Analyze this dataset for machine learning training:

USER REQUEST: {user_request}
DATA INFO:
- Rows: {num_rows}
- Columns: {num_columns}
- Column names: {column_names}
- Sample data: {sample_rows}
- Data types: {data_types}

Determine:
1. Task type (classification, generation, regression)
2. Data quality (good, needs_cleaning)
3. Preprocessing steps needed
4. Number of classes (for classification)

Return JSON format:
{{
    "task_type": "classification or generation",
    "data_quality": "good or needs_cleaning",
    "preprocessing_steps": ["step1", "step2"],
    "num_classes": 3,
    "confidence": 0.9
}}
"""

# Planning Agent Prompts
PLANNING_PROMPT = """
Create an optimal training plan for this ML task:

USER REQUEST: {user_request}
TASK TYPE: {task_type}
DATA SIZE: {num_samples} samples, {num_classes} classes
DATA QUALITY: {data_quality}
PREPROCESSING NEEDED: {preprocessing_needed}
AVAILABLE MODELS: {available_models}
CONSTRAINTS: {constraints}

Create a training plan with:
1. Best model selection
2. Training configuration (epochs, batch_size, learning_rate)
3. Preprocessing steps
4. Expected training time

Return JSON format:
{{
    "base_model": "distilbert-base-uncased",
    "task_approach": "sequence_classification",
    "training_config": {{
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5
    }},
    "preprocessing_steps": ["text_cleaning"],
    "evaluation_metrics": ["accuracy", "f1"],
    "expected_duration": "15 minutes",
    "confidence": 0.9
}}
"""

# Training Monitoring Prompts
TRAINING_MONITOR_PROMPT = """
Analyze these training metrics and provide insights:

TRAINING RESULTS:
- Final loss: {final_loss}
- Final accuracy: {final_accuracy}
- Epochs completed: {training_epochs}
- Loss history: {learning_curve}

Provide analysis:
1. Is training successful?
2. Are there any issues (overfitting, underfitting)?
3. Suggestions for improvement

Keep response concise and actionable.
"""

# Error Analysis Prompts
ERROR_ANALYSIS_PROMPT = """
Training failed with this error:
ERROR: {error_message}

CONFIGURATION:
- Model: {model_name}
- Task: {task_type}
- Training config: {training_config}

Provide:
1. Root cause analysis
2. Specific fix suggestions
3. Alternative approaches

Be practical and specific in your recommendations.
"""

# Model Selection Prompts
MODEL_SELECTION_PROMPT = """
Choose the best model for this task:

TASK: {task_description}
DATA SIZE: {data_size} samples
TASK TYPE: {task_type}
CONSTRAINTS: {constraints}

Available models:
- distilbert-base-uncased (fast, good for small data)
- bert-base-uncased (better accuracy, slower)
- roberta-base (best accuracy, slowest)

Recommend the optimal model and explain why.
"""

# Deployment Prompts
DEPLOYMENT_PROMPT = """
Create deployment configuration for this trained model:

MODEL INFO:
- Task: {task_type}
- Accuracy: {accuracy}
- Model size: {model_size_mb}MB
- Expected usage: {expected_usage}

Suggest:
1. Deployment strategy (API, batch processing)
2. Resource requirements
3. Scaling considerations

Return practical deployment recommendations.
"""



#Strategy Prompt
STRATEGY_PROMPT = """
You are an AI orchestrator that coordinates specialized agents for machine learning tasks.

USER REQUEST: {user_prompt}
DATA FILE: {data_path}

Available agents:
1. DATA_AGENT: Analyzes datasets, determines task type, assesses data quality
2. PLANNING_AGENT: Creates optimal training strategies, selects models and hyperparameters  
3. TRAINING_AGENT: Executes model training and handles evaluation and deployment
Analyze the user request and create an execution strategy:

1. UNDERSTAND: What does the user want to accomplish?
2. PLAN: Which agents should be called and in what order?
3. REASONING: Why this approach?
4. PARAMETERS: What specific parameters should be passed to each agent?
5. ADAPTATION: How should we adapt if agents return unexpected results?

Think step by step, then return strategy as JSON:

{{
    "user_intent": "classification task for customer emails",
    "approach": "standard_pipeline or custom_approach",
    "reasoning": "why this strategy makes sense",
    "execution_steps": [
        {{
            "step": 1,
            "agent": "data_agent",
            "action": "analyze_data",
            "params": {{"data_path": {data_path}, "user_prompt": {user_prompt}}},
            "expected_output": "data profile with task type"
        }},
        {{
            "step": 2, 
            "agent": "planning_agent",
            "action": "create_training_plan",
            "params": {{"data_profile": "from_step_1", "user_prompt": {user_prompt}}},
            "expected_output": "training strategy"
        }},
        {{
            "step": 3, 
            "agent": "training_agent",
            "action": "execute_training_plan",
            "params": {{"data_profile": "from_step_1", "training_plan": "from_step_2", data_path": {data_path}, "user_id": "vadaliv"}},
            "expected_output": "training strategy"
        }}
    ],
    "success_criteria": "what constitutes success",
    "fallback_strategy": "what to do if something fails"
}}
"""
#         training_plan=plan,
#         data_profile=profile,
#         #data_path="../../examples/sample_data.csv",
#         data_path="/Users/vvadali/Documents/git/vadaliv/llm-agent-finetuning/examples/sample_data.csv",
#         user_id="vadaliv"
#     )


ADAPTATION_PROMPT = """
The execution strategy needs adaptation based on intermediate results.

CURRENT STRATEGY: {strategy['approach']}
STEP {current_step} RESULT: {str(result)[:200]}...

The result shows some issues. How should I adapt the remaining strategy?

Should I:
1. Continue with current plan
2. Modify parameters for next steps  
3. Skip certain steps
4. Add additional steps

Return adaptation as JSON:
{{
    "action": "modify_parameters",
    "reasoning": "why adapt",
    "changes": {{"step_3": {{"new_params": {{...}}}}}}
}}
"""



FAILURE_PROMPT = """
Agent execution failed:

FAILED STEP: {failed_step}
ERROR: {str(error)}
STRATEGY: {strategy['approach']}

How should I handle this failure?

Options:
1. "retry" with modified parameters
2. "skip" this step and continue
3. "abort" the entire process

Consider the error type and overall strategy. Return decision as JSON:
{{
    "action": "retry",
    "reasoning": "why this choice",
    "new_params": {{"batch_size": 8}} // if retry
}}
"""


SYNTHESIS_PROMPT = """
Synthesize the results from multiple AI agents into a final response for the user.

USER ORIGINAL REQUEST: "{user_prompt}"
EXECUTION STRATEGY: {strategy['approach']}

AGENT RESULTS:
{json.dumps(results, default=str, indent=2)}

Create a final response that:
1. Directly answers what the user wanted
2. Summarizes what was accomplished
3. Provides key metrics and outcomes
4. Gives next steps or usage instructions
5. Is clear and non-technical

Return as JSON:
{{
    "summary": "what was accomplished",
    "key_results": {{
        "model_endpoint": "url",
        "accuracy": 0.95,
        "training_time": "15 minutes"
    }},
    "next_steps": ["how to use the model", "suggestions"],
    "technical_details": {{"for_advanced_users": "..."}}
}}
"""