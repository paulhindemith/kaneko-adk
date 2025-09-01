"""
Show a Vega-Lite Chart
"""
import os
from typing import Any, Callable, Optional

from google.adk.tools import FunctionTool
from google.genai import types
import jsonschema
import pandas as pd
from typing_extensions import override

DIR_PATH = os.path.dirname(__file__)


class ShowChartTool(FunctionTool):
    """Show a Vega-Lite chart."""

    def __init__(self, func: Callable[..., Any], schema: types.Schema):
        self.schema = schema
        super().__init__(func=func)

    @override
    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:

        return types.FunctionDeclaration(
            name=self.func.__name__,
            description=self.func.__doc__,
            parameters=self.schema,
        )


def build_tool(schema_name: str) -> ShowChartTool:
    """
    Build a tool for showing a Vega-Lite chart.
    Args:
        schema_name (str): The name of the schema to use for the chart.
    Returns:
        ShowChartTool : A function that takes a DuckDB backend connection and returns a Vega-Lite chart.
    """

    file_name = f"_auto_generated_{schema_name}_schema.json"
    with open(os.path.join(DIR_PATH, file_name), "r", encoding="utf-8") as f:
        json_schema = types.JSONSchema.model_validate_json(f.read())
        schema = types.Schema.from_json_schema(json_schema=json_schema,
                                               api_option="VERTEX_AI")

    def show_chart(**kwargs) -> Callable:
        """
        Show a Vega-Lite chart.
        """
        try:
            jsonschema.validate(instance=kwargs,
                                schema=json_schema.model_dump(mode="json"))

            data = kwargs["data"]

            if not data.get("format"):
                raise ValueError("Missing data format")
            format_info = data["format"]
            if format_info.get("type") == "csv":
                csv_url = data.get("url")
                if not csv_url:
                    raise ValueError("Missing CSV URL")
                pd.read_csv(csv_url, nrows=5)
            else:
                raise ValueError("Invalid data format")
            return {"message": "OK"}
        except Exception as e:
            return {"error": str(e)}

    return ShowChartTool(func=show_chart, schema=schema)
