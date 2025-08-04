#Train Your Models with a Prompt: The Multi-Agent Revolution
"""
Main Entry Point - Multiple ways to invoke the LLM orchestrator
Choose your preferred interface: CLI, API, or programmatic
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import pdb
import traceback

# Configure logging if not already configured
logging.basicConfig(level=logging.INFO)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import LLMOrchestrator

#prompt = "re train model with 4 epochs 1e-5 learning rate and 8 as batch size"
prompt = "what are the model metrics" 
#prompt = "what is the f1 score" 
#prompt = "Classify customer reviews as positive or negative" 
data_path = "/Users/vvadali/Documents/git/vadaliv/llm-agent-finetuning/examples/sample_data.csv"
model_path = "/Users/vvadali/Documents/git/vadaliv/llm-agent-finetuning/models/output/final_model"

def main_cli():
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable required")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return 1

    # Check data file exists
    if not Path(data_path).exists():
        print(f"❌ Error: Data file '{data_path}' not found")
        return 1

    try:
        orchestrator = LLMOrchestrator()
        result = orchestrator.process_user_request(
            user_prompt=prompt, data_path=data_path,model_path = model_path
        )
        print(result)
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        logging.debug("Traceback:\n%s", traceback.format_exc())

if __name__ == "__main__":
    main_cli()
