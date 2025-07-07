#!/usr/bin/env python3
"""
Main XGBoost Agent Implementation

This is the primary agent that connects to the XGBoost MCP server and provides
a conversational interface for machine learning operations.

Usage:
    python agent.py                    # Interactive mode
    python agent.py --message "..."    # Single message mode
    python agent.py --agent data_analyst --message "..."  # Use specific agent
"""

import asyncio
import argparse
from mcp_agent.core.fastagent import FastAgent

# Create the FastAgent application using the configuration file
fast = FastAgent("XGBoost ML Agent", config_path="fastagent.config.yaml")

@fast.agent(
    name="ml_assistant",
    instruction="""
    You are an expert machine learning assistant with access to XGBoost tools and filesystem access.
    
    Your capabilities include:
    - Training XGBoost models (both regression and classification)
    - Making predictions with trained models
    - Analyzing model performance and feature importance
    - Reading and writing data files (CSV, JSON, etc.)
    - Providing insights about model behavior
    - Helping with data preparation and ML workflow guidance
    
    When working with users:
    1. Ask clarifying questions about their data and ML goals
    2. Help them load and examine their data files
    3. Suggest appropriate model types and parameters
    4. Explain your reasoning for model choices
    5. Provide clear interpretations of results
    6. Save results and models to files when appropriate
    7. Offer suggestions for model improvement
    
    Always be helpful, clear, and educational in your responses.
    """,
    servers=["xgboost", "filesystem"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
    human_input=True,
)
async def main():
    """Main agent for general ML assistance"""
    async with fast.run() as agent:
        print("🤖 XGBoost ML Assistant ready!")
        print("I can help you with:")
        print("  • Training XGBoost models")
        print("  • Making predictions")
        print("  • Analyzing model performance")
        print("  • Reading and writing data files")
        print("  • ML workflow guidance")
        print("\nType your question or 'help' for more information.")
        await agent.interactive()

@fast.agent(
    name="data_analyst",
    instruction="""
    You are a data analyst specializing in XGBoost model analysis and interpretation.
    
    Your role:
    - Analyze existing trained models
    - Read and examine data files (CSV, JSON, etc.)
    - Explain model performance metrics
    - Identify important features and their relationships
    - Provide actionable insights from model results
    - Compare different models and their characteristics
    - Generate reports and save analysis results to files
    
    Always provide clear, data-driven insights with specific recommendations.
    """,
    servers=["xgboost", "filesystem"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)
async def data_analyst():
    """Specialized agent for data analysis and model interpretation"""
    async with fast.run() as agent:
        print("📊 Data Analyst Agent ready!")
        print("I specialize in model analysis and interpretation.")
        print("I have access to both XGBoost tools and filesystem operations.")
        await agent.interactive()

@fast.agent(
    name="model_trainer",
    instruction="""
    You are a model training specialist focused on XGBoost optimization.
    
    Your expertise:
    - Selecting optimal XGBoost parameters
    - Training models with different configurations
    - Handling both regression and classification tasks
    - Providing guidance on hyperparameter tuning
    - Explaining the impact of different parameters
    - Reading data files and saving training results
    
    Guide users through the model training process step-by-step.
    """,
    servers=["xgboost", "filesystem"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)
async def model_trainer():
    """Specialized agent for model training and optimization"""
    async with fast.run() as agent:
        print("🎯 Model Trainer Agent ready!")
        print("I specialize in XGBoost training and optimization.")
        print("I can read data files and save training results.")
        await agent.interactive()

@fast.agent(
    name="predictor",
    instruction="""
    You are a prediction specialist for trained XGBoost models.
    
    Your focus:
    - Making predictions with existing models
    - Interpreting prediction results
    - Explaining prediction confidence and reliability
    - Handling prediction data formatting
    - Reading input data files and saving prediction results
    - Providing prediction summaries and insights
    
    Help users understand not just what the model predicts, but why.
    """,
    servers=["xgboost", "filesystem"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)
async def predictor():
    """Specialized agent for making and interpreting predictions"""
    async with fast.run() as agent:
        print("🔮 Predictor Agent ready!")
        print("I specialize in making and interpreting predictions.")
        print("I can read data files and save prediction results.")
        await agent.interactive()

async def run_specific_agent(agent_name):
    """Run a specific agent by name"""
    agent_functions = {
        "ml_assistant": main,
        "data_analyst": data_analyst,
        "model_trainer": model_trainer,
        "predictor": predictor,
    }
    
    if agent_name in agent_functions:
        await agent_functions[agent_name]()
    else:
        print(f"❌ Unknown agent: {agent_name}")
        print("Available agents: ml_assistant, data_analyst, model_trainer, predictor")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost ML Agent")
    parser.add_argument("--agent", type=str, help="Specific agent to run", default="ml_assistant")
    parser.add_argument("--message", type=str, help="Single message to send")
    
    args = parser.parse_args()
    
    if args.message:
        # For single message, use the specified agent or default to ml_assistant
        print(f"Running single message with agent: {args.agent}")
        asyncio.run(run_specific_agent(args.agent))
    elif args.agent and args.agent != "ml_assistant":
        asyncio.run(run_specific_agent(args.agent))
    else:
        asyncio.run(main()) 