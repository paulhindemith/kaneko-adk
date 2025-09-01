"""
Show a Vega-Lite Chart
"""
import inspect
import logging
import os
from typing import Any, Callable, Dict, Optional

from google.adk.tools import FunctionTool
from google.adk.tools import ToolContext
from google.genai import types
import pandas as pd
from pydantic import ValidationError
from typing_extensions import override

from kaneko_adk.tools.show_chart.auto_generated_gemini_models import Model
from kaneko_adk.tools.show_chart.auto_generated_gemini_models import Type

DIR_PATH = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO)
default_logger = logging.getLogger(__name__)


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

    @override
    async def run_async(self, *, args: dict[str, Any],
                        tool_context: ToolContext) -> Any:
        args_to_call = args.copy()

        if (inspect.iscoroutinefunction(self.func)
                or hasattr(self.func, '__call__')
                and inspect.iscoroutinefunction(self.func.__call__)):
            return await self.func(args_to_call)
        else:
            return self.func(args_to_call)


def build_tool(schema_name: str,
               logger: logging.Logger = default_logger) -> ShowChartTool:
    """
    Build a tool for showing a Vega-Lite chart.
    Args:
        schema_name (str): The name of the schema to use for the chart.
        logger (logging.Logger): The logger to use for logging.

    Returns:
        ShowChartTool : A function that takes a DuckDB backend connection and returns a Vega-Lite chart.
    """

    file_name = f"auto_generated_{schema_name}_schema.json"
    with open(os.path.join(DIR_PATH, file_name), "r", encoding="utf-8") as f:
        json_schema = types.JSONSchema.model_validate_json(f.read())
        schema = types.Schema.from_json_schema(json_schema=json_schema,
                                               api_option="VERTEX_AI")

    def show_chart(args) -> Dict:
        """
        Show a Vega-Lite chart.
        """
        try:

            model = Model.model_validate(args)

            if model.data.format.type == Type.csv:
                pd.read_csv(model.data.url, nrows=5)
            else:
                raise ValueError("Invalid data format")
            return {"message": "OK"}
        except ValidationError as e:
            return {"error": e.json()}
        except Exception as e:
            return {"error": str(e)}

    return ShowChartTool(func=show_chart, schema=schema)
