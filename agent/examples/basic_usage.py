#!/usr/bin/env python3
"""
Basic XGBoost Agent Usage Examples

This script demonstrates simple usage patterns for the XGBoost agents.
Perfect for getting started and understanding the basic capabilities.

Usage:
    python basic_usage.py
"""

import asyncio
import json
from mcp_agent.core.fastagent import FastAgent

# Create the FastAgent application
fast = FastAgent("XGBoost Basic Examples")

@fast.agent(
    name="simple_ml",
    instruction="""
    You are a simple ML assistant that helps users with basic XGBoost operations.
    Keep responses concise and focused on the task at hand.
    """,
    servers=["xgboost"],
    model="claude-3-5-sonnet-20241022",
    use_history=False,  # Each interaction is independent
)

async def demonstrate_basic_operations():
    """Demonstrate basic XGBoost operations"""
    
    print("🚀 XGBoost Basic Operations Demo")
    print("=" * 50)
    
    async with fast.run() as agent:
        
        # 1. Get available models
        print("\n1. Checking available models...")
        models_response = await agent.simple_ml("List all available trained models")
        print(f"Models: {models_response}")
        
        # 2. Create sample data and train a model
        print("\n2. Training a simple regression model...")
        
        # Sample data for regression
        sample_data = {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            "target": [2.1, 4.2, 6.3, 8.4, 10.5, 12.6, 14.7, 16.8, 18.9, 21.0]
        }
        
        data_json = json.dumps(sample_data)
        training_message = f"""
        Please train a regression model called 'demo_model' using this data:
        {data_json}
        
        Use 'target' as the target column and keep the parameters simple.
        """
        
        train_response = await agent.simple_ml(training_message)
        print(f"Training result: {train_response}")
        
        # 3. Get model information
        print("\n3. Getting model information...")
        info_response = await agent.simple_ml("Get detailed information about the 'demo_model'")
        print(f"Model info: {info_response}")
        
        # 4. Make predictions
        print("\n4. Making predictions...")
        pred_data = {
            "feature1": [11.0, 12.0],
            "feature2": [10.5, 11.5]
        }
        pred_json = json.dumps(pred_data)
        
        pred_message = f"""
        Make predictions using the 'demo_model' with this data:
        {pred_json}
        """
        
        pred_response = await agent.simple_ml(pred_message)
        print(f"Predictions: {pred_response}")
        
        # 5. Get feature importance
        print("\n5. Analyzing feature importance...")
        importance_response = await agent.simple_ml("Show feature importance for the 'demo_model'")
        print(f"Feature importance: {importance_response}")

async def interactive_example():
    """Run an interactive example"""
    
    print("\n🎯 Interactive XGBoost Assistant")
    print("=" * 50)
    print("You can now interact with the XGBoost assistant.")
    print("Try commands like:")
    print("  • 'List all models'")
    print("  • 'Train a classification model'")
    print("  • 'Make predictions with <model_name>'")
    print("  • 'Show feature importance for <model_name>'")
    print("\nType 'exit' to quit.\n")
    
    async with fast.run() as agent:
        await agent.simple_ml.interactive()

async def main():
    """Main function to run examples"""
    
    choice = input("""
Choose an example:
1. Basic operations demo (automated)
2. Interactive assistant
Enter choice (1 or 2): """)
    
    if choice == "1":
        await demonstrate_basic_operations()
    elif choice == "2":
        await interactive_example()
    else:
        print("Invalid choice. Running interactive assistant...")
        await interactive_example()

if __name__ == "__main__":
    asyncio.run(main()) 