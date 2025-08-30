"""
Show a Vega-Lite Chart
"""
import os
from typing import Any, Callable, Dict, Optional

from google.adk.tools import FunctionTool
from google.genai import types
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
        schema = types.Schema.model_validate_json(f.read())

    def show_chart(data: Dict[str, Any], description: str, encoding: Dict[str,
                                                                          Any],
                   mark: Dict[str, Any], title: str) -> Callable:
        """
        Show a Vega-Lite chart.
        """
        # 全体的に引数を出力
        print(f"{data=}")
        print(f"{description=}")
        print(f"{encoding=}")
        print(f"{mark=}")
        print(f"{title=}")
        return {"message": "OK"}

    return ShowChartTool(func=show_chart, schema=schema)
