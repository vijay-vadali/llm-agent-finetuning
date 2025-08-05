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
    "num_classes": 3,
    "data_quality": "good or needs_cleaning",
    "preprocessing_steps": ["step1", "step2"],
    "confidence": 0.9
    "reasoning": "why this strategy makes sense"
}}
"""


MODEL_RETRAIN_PROMPT = """
Extract config parameters from this user prompt:

USER REQUEST: {user_prompt}
Return JSON format: num_epochs, learning_rate, batch_size.
{{
    "num_epochs": "3",
    "learning_rate": "2e-5",
    "batch_size": 16
}}
"""

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
    "confidence": 0.9,
     "reasoning": "why this strategy makes sense"
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

STRATEGY_PROMPT = """
You are an AI orchestrator that coordinates specialized agents for machine learning tasks.

USER REQUEST: {user_prompt}
DATA FILE: {data_path}

Available agents:
1. DATA_AGENT: Analyzes datasets, determines task type, assesses data quality
2. PLANNING_AGENT: Creates optimal training strategies, selects models and hyperparameters  
3. TRAINING_AGENT: Executes model training and analyzes the user request and create an execution strategy:

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
            "params": {{"data_profile": "from_step_1", "training_plan": "from_step_2", data_path": {data_path}}},
            "expected_output": "training strategy"
        }}
    ],
    "success_criteria": "what constitutes success",
    "fallback_strategy": "what to do if something fails"
}}
"""
ANALYZE_METRICS_PROMPT = """
Given the model's evaluation metrics, provide a brief summary covering:

Performance level (Excellent, Good, Fair, or Poor)
F1 score and accuracy (with values)
A short interpretation of what those scores mean in context
Actionable recommendations for improvement (if any)
A high-level overall assessment of the model's performance

Format with simple dash bullets (e.g., -) and no markdown, colors, or special characters.

USER REQUEST: F1 Score: {f1_score}, Accuracy: {accuracy}

Performance Level Guidelines:
- Good: F1 > 0.8 and Accuracy > 0.8
- Fair: F1 > 0.7 and Accuracy > 0.7
- Poor: F1 <= 0.6 or Accuracy <= 0.6
"""
