"""
Data Fetcher Agent - Direct implementation using OpenAI function calling
No complex LangChain dependencies - production stable
"""

import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from typing import Dict, Any, List
import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

from tools.stock_data import StockDataTool

# Setup
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcherAgent:
    """
    Financial data fetcher agent using OpenAI function calling.
    
    This implementation uses direct OpenAI API calls for stability
    instead of LangChain/LangGraph which have frequent breaking changes.
    """

    def __init__(self):
        """Initialize the Data Fetcher Agent with OpenAI client and tools"""
        
        # Initialize OpenAI client (works with OpenRouter)
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        
        self.model = "qwen/qwen-2.5-72b-instruct"
        self.stock_tool = StockDataTool()
        
    def _get_function_definitions(self) -> List[Dict]:
        """
        Define available functions for the LLM.
        
        Returns:
            List of function definitions in OpenAI format
        """
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "fetch_stock_info",
                    "description": "Fetch current stock information including price, high, low, open, and previous close for a given ticker symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')"
                            }
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_quarterly_revenue",
                    "description": "Fetch quarterly revenue data for a company for a specific year and quarter",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., 'AAPL')"
                            },
                            "year": {
                                "type": "integer",
                                "description": "Year (e.g., 2024, 2025)"
                            },
                            "quarter": {
                                "type": "integer",
                                "description": "Quarter number (1, 2, 3, or 4)",
                                "enum": [1, 2, 3, 4]
                            }
                        },
                        "required": ["ticker", "year", "quarter"]
                    }
                }
            }
        ]
    
    def _execute_function(self, function_name: str, arguments: Dict) -> Dict:
        """
        Execute the requested function with arguments.
        
        Args:
            function_name: Name of the function to execute
            arguments: Dictionary of arguments
            
        Returns:
            Function execution result
        """
        
        logger.info(f"Executing {function_name} with args: {arguments}")
        
        if function_name == "fetch_stock_info":
            return self.stock_tool.get_stock_info(arguments["ticker"])
        
        elif function_name == "fetch_quarterly_revenue":
            return self.stock_tool.get_quarterly_revenue(
                arguments["ticker"],
                arguments["year"],
                arguments["quarter"]
            )
        
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    def run(self, task: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Execute a data fetching task using function calling.
        
        Args:
            task: Natural language description of the task
            max_iterations: Maximum number of function calls allowed
            
        Returns:
            Dictionary with execution results
            
        Example:
            >>> agent = DataFetcherAgent()
            >>> result = agent.run("What is Apple's stock price?")
            >>> print(result["output"])
        """
        try:
            logger.info(f"DataFetcherAgent executing: {task}")
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a financial data fetcher agent specialized in retrieving stock market data.

Your capabilities:
- Fetch current stock prices and market data using fetch_stock_info
- Retrieve quarterly revenue information using fetch_quarterly_revenue
- Provide accurate, structured financial data

Guidelines:
1. Always use the available functions to fetch real-time data
2. Return clean, structured responses
3. If data is unavailable, clearly state that
4. Format numbers appropriately (e.g., $123.45 for prices)
5. Be concise and accurate

When asked to compare stocks, fetch data for each one separately and then compare them.
Always call the appropriate function when the user asks for stock information."""
                },
                {
                    "role": "user",
                    "content": task
                }
            ]
            
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # Call LLM
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._get_function_definitions(),
                    tool_choice="auto"
                )
                
                response_message = response.choices[0].message
                
                # Check if LLM wants to call a function
                if response_message.tool_calls:
                    # Add assistant's response to messages
                    messages.append(response_message)
                    
                    # Execute each function call
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Execute the function
                        function_response = self._execute_function(
                            function_name,
                            function_args
                        )
                        
                        # Add function response to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(function_response)
                        })
                    
                    # Continue the loop to get final answer
                    continue
                
                else:
                    # No more function calls, we have the final answer
                    final_content = response_message.content
                    
                    return {
                        "success": True,
                        "output": final_content,
                        "agent": "DataFetcherAgent",
                        "iterations": iteration
                    }
            
            # Max iterations reached
            return {
                "success": False,
                "error": f"Maximum iterations ({max_iterations}) reached without final answer",
                "agent": "DataFetcherAgent"
            }
            
        except Exception as e:
            logger.error(f"DataFetcherAgent error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "agent": "DataFetcherAgent"
            }


# ==============================
# TESTING
# ==============================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Fetcher Agent")
    print("=" * 60)
    
    agent = DataFetcherAgent()
    
    # Test 1: Basic stock info
    print("\nTest 1: Fetch Apple stock info")
    print("-" * 60)
    result = agent.run("What is the current price of Apple stock (AAPL)?")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Iterations: {result.get('iterations', 'N/A')}")
        print(f"Output:\n{result['output']}")
    else:
        print(f"Error: {result.get('error')}")
    
    # Test 2: Multiple stocks
    print("\n" + "=" * 60)
    print("Test 2: Compare two stocks")
    print("-" * 60)
    result = agent.run("Compare the current prices of Apple (AAPL) and Microsoft (MSFT)")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Iterations: {result.get('iterations', 'N/A')}")
        print(f"Output:\n{result['output']}")
    else:
        print(f"Error: {result.get('error')}")
    
    # Test 3: Quarterly revenue
    print("\n" + "=" * 60)
    print("Test 3: Fetch quarterly revenue")
    print("-" * 60)
    result = agent.run("What was Apple's revenue in Q4 2024?")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Iterations: {result.get('iterations', 'N/A')}")
        print(f"Output:\n{result['output']}")
    else:
        print(f"Error: {result.get('error')}")
    
    print("\n" + "=" * 60)