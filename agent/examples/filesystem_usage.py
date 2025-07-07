#!/usr/bin/env python3
"""
Filesystem Usage Examples for XGBoost Agents

This script demonstrates how to use the filesystem MCP server with XGBoost agents
for data handling, model persistence, and result management.

Usage:
    python filesystem_usage.py
"""

import asyncio
import json
import csv
import os
from pathlib import Path
from mcp_agent.core.fastagent import FastAgent

# Create the FastAgent application
fast = FastAgent("XGBoost Filesystem Examples")

@fast.agent(
    name="file_ml_agent",
    instruction="""
    You are a machine learning agent with file system access for data management.
    
    Your capabilities:
    - Read data from CSV, JSON, and other file formats
    - Process and analyze data files
    - Train XGBoost models on file-based datasets
    - Save model results and predictions to files
    - Generate analysis reports and save them
    - Manage data workflows with file operations
    
    Always help users work with their data files effectively.
    """,
    servers=["xgboost", "filesystem"],
    model="claude-3-5-sonnet-20241022",
    use_history=True,
)

async def create_sample_data():
    """Create sample data files for demonstration"""
    
    print("📁 Creating sample data files...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample regression data
    regression_data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        "feature3": [2.0, 1.8, 2.2, 1.9, 2.1, 2.3, 1.7, 2.4, 2.0, 1.6],
        "target": [2.1, 4.2, 6.3, 8.4, 10.5, 12.6, 14.7, 16.8, 18.9, 21.0]
    }
    
    # Save as JSON
    with open(data_dir / "regression_data.json", "w") as f:
        json.dump(regression_data, f, indent=2)
    
    # Save as CSV
    with open(data_dir / "regression_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature1", "feature2", "feature3", "target"])
        for i in range(len(regression_data["feature1"])):
            writer.writerow([
                regression_data["feature1"][i],
                regression_data["feature2"][i],
                regression_data["feature3"][i],
                regression_data["target"][i]
            ])
    
    # Sample classification data
    classification_data = {
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "income": [30000, 45000, 60000, 75000, 90000, 105000, 120000, 135000, 150000, 165000],
        "education": [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
        "approved": [0, 0, 1, 1, 1, 1, 1, 0, 0, 1]
    }
    
    # Save classification data as JSON
    with open(data_dir / "classification_data.json", "w") as f:
        json.dump(classification_data, f, indent=2)
    
    # Save as CSV
    with open(data_dir / "classification_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["age", "income", "education", "approved"])
        for i in range(len(classification_data["age"])):
            writer.writerow([
                classification_data["age"][i],
                classification_data["income"][i],
                classification_data["education"][i],
                classification_data["approved"][i]
            ])
    
    print("✅ Sample data files created in 'data' directory:")
    print("  - regression_data.json")
    print("  - regression_data.csv")
    print("  - classification_data.json")
    print("  - classification_data.csv")

async def demonstrate_file_operations():
    """Demonstrate file operations with the ML agent"""
    
    print("\n🤖 Demonstrating file operations with ML agent")
    print("=" * 60)
    
    async with fast.run() as agent:
        
        # 1. List available data files
        print("\n1. Listing available data files...")
        files_response = await agent.file_ml_agent(
            "List all files in the data directory and tell me what data files are available"
        )
        print(f"Files: {files_response}")
        
        # 2. Read and analyze a CSV file
        print("\n2. Reading and analyzing CSV data...")
        csv_analysis = await agent.file_ml_agent(
            "Read the regression_data.csv file and analyze its structure. "
            "Tell me about the columns, data types, and any patterns you notice."
        )
        print(f"CSV Analysis: {csv_analysis}")
        
        # 3. Train a model using file data
        print("\n3. Training model using file data...")
        training_response = await agent.file_ml_agent(
            "Read the regression_data.json file and train an XGBoost regression model "
            "called 'file_model' using the data. Use 'target' as the target column."
        )
        print(f"Training Result: {training_response}")
        
        # 4. Save model information to file
        print("\n4. Saving model information to file...")
        save_info_response = await agent.file_ml_agent(
            "Get detailed information about the 'file_model' and save it to a file "
            "called 'model_info.txt' in the data directory."
        )
        print(f"Save Info Result: {save_info_response}")
        
        # 5. Make predictions and save results
        print("\n5. Making predictions and saving results...")
        prediction_response = await agent.file_ml_agent(
            "Use the 'file_model' to make predictions on new data: "
            "feature1=[11, 12], feature2=[10.5, 11.5], feature3=[2.2, 1.8]. "
            "Save the predictions to a file called 'predictions.json' in the data directory."
        )
        print(f"Prediction Result: {prediction_response}")
        
        # 6. Generate analysis report
        print("\n6. Generating analysis report...")
        report_response = await agent.file_ml_agent(
            "Create a comprehensive analysis report that includes: "
            "1. Summary of the regression_data.csv dataset "
            "2. Model performance details "
            "3. Feature importance analysis "
            "4. Prediction results "
            "Save this report as 'analysis_report.txt' in the data directory."
        )
        print(f"Report Result: {report_response}")

async def demonstrate_workflow_with_files():
    """Demonstrate a complete ML workflow using files"""
    
    print("\n🔄 Demonstrating complete ML workflow with files")
    print("=" * 60)
    
    async with fast.run() as agent:
        
        # Complete workflow using files
        workflow_response = await agent.file_ml_agent(
            "I want to do a complete ML workflow using the files in the data directory. "
            "Please: "
            "1. Read the classification_data.csv file "
            "2. Analyze the data and create a data summary "
            "3. Train a classification model called 'classification_model' "
            "4. Analyze the model's feature importance "
            "5. Save all results to a workflow_summary.json file "
            "Walk me through each step and show me what you're doing."
        )
        print(f"Workflow Result: {workflow_response}")

async def interactive_file_agent():
    """Run an interactive session with file capabilities"""
    
    print("\n🎯 Interactive File-Enabled ML Agent")
    print("=" * 60)
    print("You can now interact with the ML agent that has file system access.")
    print("Try commands like:")
    print("  • 'List all files in the data directory'")
    print("  • 'Read the regression_data.csv file and show me the first few rows'")
    print("  • 'Train a model using the classification_data.json file'")
    print("  • 'Save the model results to a file'")
    print("  • 'Create a data analysis report'")
    print("\nType 'exit' to quit.\n")
    
    async with fast.run() as agent:
        await agent.file_ml_agent.interactive()

async def main():
    """Main function to run filesystem examples"""
    
    print("📁 XGBoost Filesystem Integration Examples")
    print("=" * 60)
    
    # Create sample data first
    await create_sample_data()
    
    choice = input("""
Choose an example:
1. Demonstrate file operations
2. Complete ML workflow with files
3. Interactive file-enabled agent
Enter choice (1-3): """)
    
    if choice == "1":
        await demonstrate_file_operations()
    elif choice == "2":
        await demonstrate_workflow_with_files()
    elif choice == "3":
        await interactive_file_agent()
    else:
        print("Invalid choice. Running interactive agent...")
        await interactive_file_agent()

if __name__ == "__main__":
    asyncio.run(main()) 