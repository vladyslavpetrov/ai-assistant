import pandas as pd
import duckdb
from pydantic import BaseModel, Field
import os

# define the path to the transactional data file
TRANSACTION_DATA_FILE_PATH = ''
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""

CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
"""


class ToolManager:
    def __init__(self, open_ai_manager):
        self.TRANSACTION_DATA_FILE_PATH = os.getenv("TRANSACTION_DATA_FILE_PATH", "default_path.parquet")

        if not TRANSACTION_DATA_FILE_PATH:
            raise ValueError("TRANSACTION_DATA_FILE_PATH is missing from the environment.")

        self.open_ai_manager = open_ai_manager

    def generate_sql_query(self, prompt: str, columns: list, table_name: str) -> str:
        formatted_prompt = SQL_GENERATION_PROMPT.format(prompt=prompt,
                                                        columns=columns,
                                                        table_name=table_name)
        response = self.open_ai_manager.generate_response(formatted_prompt)

        return response.choices[0].message.content

    def lookup_sales_data(self, prompt: str) -> str:
        try:

            # define the table name
            table_name = ""

            # step 1: read the data file into a DuckDB table
            df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
            duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

            # step 2: generate the SQL code
            sql_query = self.generate_sql_query(prompt, df.columns, table_name)
            # clean the response to make sure it only includes the SQL code
            sql_query = sql_query.strip()
            sql_query = sql_query.replace("```sql", "").replace("```", "")

            # step 3: execute the SQL query
            result = duckdb.sql(sql_query).df()

            return result.to_string()
        except Exception as e:
            return f"Error accessing data: {str(e)}"

    def analyze_sales_data(self, prompt: str, data: str) -> str:
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

        response = self.open_ai_manager.generate_response(formatted_prompt)

        analysis = response.choices[0].message.content
        return analysis if analysis else "No analysis could be generated"

    def extract_chart_config(self, data: str, visualization_goal: str) -> dict:
        formatted_prompt = CHART_CONFIGURATION_PROMPT.format(data=data,
                                                             visualization_goal=visualization_goal)

        response = self.open_ai_manager.generate_response(formatted_prompt)

        try:
            # Extract axis and title info from response
            content = response.choices[0].message.content

            # Return structured chart config
            return {
                "chart_type": content.chart_type,
                "x_axis": content.x_axis,
                "y_axis": content.y_axis,
                "title": content.title,
                "data": data
            }
        except Exception:
            return {
                "chart_type": "line",
                "x_axis": "date",
                "y_axis": "value",
                "title": visualization_goal,
                "data": data
            }

    # code for step 2 of tool 3
    def create_chart(self, config: dict) -> str:
        formatted_prompt = CREATE_CHART_PROMPT.format(config=config)

        response = self.open_ai_manager.generate_response(formatted_prompt)

        code = response.choices[0].message.content
        code = code.replace("```python", "").replace("```", "")
        code = code.strip()

        return code

    def generate_visualization(self, data: str, visualization_goal: str) -> str:
        config = self.extract_chart_config(data, visualization_goal)
        code = self.create_chart(config)
        return code


class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")
