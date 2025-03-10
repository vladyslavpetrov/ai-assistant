import json

from open_ai_manager import OpenAiManager
from tool_manager import ToolManager

SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the {YOUR_DATASET}.
"""


class Router:

    def __init__(self):
        self.open_ai_manager = OpenAiManager()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_sales_data",
                    "description": "Look up data from {YOUR_DATASET}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "The unchanged prompt that the user provided."}
                        },
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_sales_data",
                    "description": "Analyze sales data to extract insights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {"type": "string", "description": "The lookup_sales_data tool's output."},
                            "prompt": {"type": "string", "description": "The unchanged prompt that the user provided."}
                        },
                        "required": ["data", "prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_visualization",
                    "description": "Generate Python code to create data visualizations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {"type": "string", "description": "The lookup_sales_data tool's output."},
                            "visualization_goal": {"type": "string", "description": "The goal of the visualization."}
                        },
                        "required": ["data", "visualization_goal"]
                    }
                }
            }
        ]

    def define_tools(self):
        tool_manager = ToolManager(self.open_ai_manager)
        # Dictionary mapping function names to their implementations
        tool_implementations = {
            "lookup_sales_data": tool_manager.lookup_sales_data,
            "analyze_sales_data": tool_manager.analyze_sales_data,
            "generate_visualization": tool_manager.generate_visualization
        }

        return tool_implementations

    def handle_tool_calls(self, tool_calls, messages):
        for tool_call in tool_calls:
            function = self.define_tools()[tool_call.function.name]
            function_args = json.loads(tool_call.function.arguments)
            result = function(**function_args)
            messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})

        return messages

    def run_agent(self, messages: str) -> str:
        print("Running agent with messages:", messages)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Check and add system prompt if needed
        if not any(
                isinstance(message, dict) and message.get("role") == "system" for message in messages
        ):
            system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
            messages.append(system_prompt)

        while True:
            print("Making router call to OpenAI")
            response = self.open_ai_manager.client.chat.completions.create(
                model=self.open_ai_manager.model,
                messages=messages,
                tools=self.tools,
            )
            messages.append(response.choices[0].message)
            tool_calls = response.choices[0].message.tool_calls
            print("Received response with tool calls:", bool(tool_calls))

            # if the model decides to call function(s), call handle_tool_calls
            if tool_calls:
                print("Processing tool calls")
                messages = self.handle_tool_calls(tool_calls, messages)
            else:
                print("No tool calls, returning final response")
                return response.choices[0].message.content
