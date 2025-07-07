#!/usr/bin/env python3
"""
XGBoost ML Workflow Examples

This module demonstrates various agent workflows for machine learning tasks,
including training pipelines, model evaluation, and prediction workflows.

Usage:
    python workflows.py                           # Interactive workflow selector
    python workflows.py --workflow ml_pipeline    # Run specific workflow
    python workflows.py --workflow model_comparison --message "Compare models"
"""

import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the FastAgent application
fast = FastAgent("XGBoost ML Workflows")

# Individual Agent Definitions
@fast.agent(
    name="data_preprocessor",
    instruction="""
    You are a data preprocessing specialist for machine learning.
    
    Your role:
    - Read and analyze raw data files (CSV, JSON, Excel, etc.)
    - Analyze data structure and quality
    - Suggest data cleaning and preprocessing steps
    - Identify missing values and outliers
    - Recommend feature engineering approaches
    - Prepare data in the correct format for XGBoost training
    - Save processed data to files for later use
    
    Always provide step-by-step guidance for data preparation.
    """,
    servers=["xgboost", "filesystem"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)

@fast.agent(
    name="model_trainer",
    instruction="""
    You are an XGBoost model training specialist.
    
    Your expertise:
    - Training XGBoost models with optimal parameters
    - Selecting appropriate model types (regression/classification)
    - Configuring hyperparameters for best performance
    - Handling different data types and sizes
    - Providing training progress and results
    
    Focus on creating robust, well-performing models.
    """,
    servers=["xgboost"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)

@fast.agent(
    name="model_evaluator",
    instruction="""
    You are a model evaluation and analysis expert.
    
    Your responsibilities:
    - Analyzing trained model performance
    - Interpreting feature importance results
    - Identifying model strengths and weaknesses
    - Suggesting improvements and optimizations
    - Providing clear, actionable insights
    - Reading evaluation data from files
    - Generating and saving comprehensive evaluation reports
    
    Always provide comprehensive evaluation with specific recommendations.
    """,
    servers=["xgboost", "filesystem"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)

@fast.agent(
    name="prediction_engine",
    instruction="""
    You are a prediction specialist for XGBoost models.
    
    Your focus:
    - Making accurate predictions with trained models
    - Interpreting prediction results and confidence
    - Handling prediction data formatting
    - Providing prediction summaries and insights
    - Explaining prediction reliability
    
    Help users understand both predictions and their implications.
    """,
    servers=["xgboost"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)

@fast.agent(
    name="ml_advisor",
    instruction="""
    You are a machine learning strategy advisor.
    
    Your expertise:
    - Providing high-level ML strategy guidance
    - Recommending appropriate approaches for different problems
    - Suggesting next steps in ML workflows
    - Interpreting results in business context
    - Coordinating between different ML specialists
    
    Focus on strategic guidance and workflow coordination.
    """,
    servers=["xgboost"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)

# Workflow Definitions

@fast.chain(
    name="ml_pipeline",
    sequence=["data_preprocessor", "model_trainer", "model_evaluator"],
    instruction="Complete ML pipeline from data preprocessing to model evaluation",
    cumulative=True,
    continue_with_final=True,
)
async def ml_pipeline():
    """Complete ML pipeline workflow"""
    async with fast.run() as agent:
        print("🚀 ML Pipeline Workflow")
        print("This workflow will guide you through:")
        print("  1. Data preprocessing and analysis")
        print("  2. Model training and optimization")
        print("  3. Model evaluation and insights")
        await agent.interactive()

@fast.chain(
    name="quick_predict",
    sequence=["prediction_engine"],
    instruction="Quick prediction workflow for existing models",
    continue_with_final=True,
)
async def quick_predict():
    """Quick prediction workflow"""
    async with fast.run() as agent:
        print("🔮 Quick Prediction Workflow")
        print("Make predictions with existing trained models.")
        await agent.interactive()

@fast.parallel(
    name="model_comparison",
    fan_out=["model_trainer", "model_trainer"],  # Train multiple models
    fan_in="model_evaluator",  # Compare results
    instruction="Train and compare multiple XGBoost models",
    include_request=True,
)
async def model_comparison():
    """Compare multiple model configurations"""
    async with fast.run() as agent:
        print("⚖️ Model Comparison Workflow")
        print("Train and compare multiple XGBoost models.")
        await agent.interactive()

@fast.evaluator_optimizer(
    name="model_optimizer",
    generator="model_trainer",
    evaluator="model_evaluator",
    min_rating="GOOD",
    max_refinements=3,
)
async def model_optimizer():
    """Iterative model optimization workflow"""
    async with fast.run() as agent:
        print("🎯 Model Optimizer Workflow")
        print("Iteratively train and optimize models until satisfied.")
        await agent.interactive()

@fast.router(
    name="ml_router",
    agents=["data_preprocessor", "model_trainer", "model_evaluator", "prediction_engine"],
)
async def ml_router():
    """Route ML requests to appropriate specialists"""
    async with fast.run() as agent:
        print("🗂️ ML Router")
        print("Routes your ML requests to the right specialist.")
        await agent.interactive()

@fast.orchestrator(
    name="ml_orchestrator",
    instruction="Coordinate complex ML projects with multiple agents",
    agents=["data_preprocessor", "model_trainer", "model_evaluator", "prediction_engine", "ml_advisor"],
    model="claude-3-opus-20240229",  # Use Opus for complex orchestration
    plan_type="iterative",
    plan_iterations=5,
)
async def ml_orchestrator():
    """Orchestrate complex ML workflows"""
    async with fast.run() as agent:
        print("🎭 ML Orchestrator")
        print("Coordinates complex ML projects with multiple specialists.")
        await agent.interactive()

# Advanced Workflow Examples

@fast.agent(
    name="experiment_tracker",
    instruction="""
    You are an ML experiment tracking specialist.
    
    Your role:
    - Track multiple model experiments
    - Compare model performance across runs
    - Maintain experiment history and metadata
    - Provide experiment summaries and insights
    - Suggest next experiments based on results
    
    Help users organize and learn from their ML experiments.
    """,
    servers=["xgboost", "memory"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)

@fast.chain(
    name="research_workflow",
    sequence=["ml_advisor", "experiment_tracker", "model_optimizer"],
    instruction="Research-oriented ML workflow for experimentation",
    cumulative=True,
)
async def research_workflow():
    """Research-oriented ML workflow"""
    async with fast.run() as agent:
        print("🔬 Research Workflow")
        print("Systematic ML research and experimentation.")
        await agent.interactive()

@fast.agent(
    name="production_deployer",
    instruction="""
    You are a model deployment specialist.
    
    Your focus:
    - Preparing models for production deployment
    - Validating model reliability and performance
    - Providing deployment recommendations
    - Monitoring and maintenance guidance
    - Production readiness assessment
    
    Ensure models are ready for real-world deployment.
    """,
    servers=["xgboost"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)

@fast.chain(
    name="production_pipeline",
    sequence=["model_optimizer", "production_deployer"],
    instruction="Production-ready ML pipeline",
    cumulative=True,
)
async def production_pipeline():
    """Production-ready ML pipeline"""
    async with fast.run() as agent:
        print("🏭 Production Pipeline")
        print("Develop and deploy production-ready ML models.")
        await agent.interactive()

# Main workflow selector
async def main():
    """Main workflow selector"""
    print("🤖 XGBoost ML Workflow Suite")
    print("\nAvailable workflows:")
    print("  1. ml_pipeline - Complete ML pipeline")
    print("  2. quick_predict - Quick predictions")
    print("  3. model_comparison - Compare models")
    print("  4. model_optimizer - Iterative optimization")
    print("  5. ml_router - Route to specialists")
    print("  6. ml_orchestrator - Complex project orchestration")
    print("  7. research_workflow - Research and experimentation")
    print("  8. production_pipeline - Production deployment")
    
    workflow_choice = input("\nSelect a workflow (1-8) or press Enter for interactive mode: ")
    
    workflows = {
        "1": ml_pipeline,
        "2": quick_predict,
        "3": model_comparison,
        "4": model_optimizer,
        "5": ml_router,
        "6": ml_orchestrator,
        "7": research_workflow,
        "8": production_pipeline,
    }
    
    if workflow_choice in workflows:
        await workflows[workflow_choice]()
    else:
        # Default to orchestrator for interactive mode
        await ml_orchestrator()

if __name__ == "__main__":
    asyncio.run(main()) 