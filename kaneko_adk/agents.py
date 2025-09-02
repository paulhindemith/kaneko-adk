"""
Agent classes for interacting with the Kaneko API.
"""
import datetime
from typing import ClassVar, List

from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from ibis.backends.duckdb import Backend
from pydantic import Field

from kaneko_adk.callbacks import build_add_context_after_tool_callback
from kaneko_adk.callbacks import build_set_context_before_model_callback
from kaneko_adk.callbacks import manage_initial_context_cache
from kaneko_adk.tools import execute_sql
from kaneko_adk.tools import show_chart

JST = datetime.timezone(datetime.timedelta(hours=9))


class DataAnalyticsAgent(LlmAgent):
    """
    Agent for performing data analytics tasks.
    """

    Table: ClassVar[execute_sql.Table] = execute_sql.Table
    SQL: ClassVar[execute_sql.SQL] = execute_sql.SQL
    MAX_ROWS: ClassVar[int] = execute_sql.MAX_ROWS
    Instruction: ClassVar[str] = """
You are a data analysis agent that answers user questions based on the provided context and available tools.

Today's date: {today}

### Action Guidelines
- For any ambiguous question, **very proactively guide the user toward an analysis.** Specifically, always execute SQL to retrieve data and provide some answer, then suggest the next suitable analysis to the user.
- **If the user's question is ambiguous, assume the most probable analysis and automatically retrieve data and draw a graph.**
- When presenting information to the user, avoid technical jargon and choose natural language.
- When executing a tool, tell the user its purpose first. It is forbidden to execute a tool without explaining its purpose.

You must follow the user-specified custom instructions below. These instructions take precedence over all other action guidelines.
> {custom_instruction}
"""

    initial_contexts: list[types.Part] = Field(default_factory=list)

    def __init__(self,
                 name: str,
                 model: str,
                 instruction: str,
                 con: Backend,
                 tables: List[execute_sql.Table],
                 today: datetime.datetime = datetime.datetime.now(JST)):
        """ Initialize the DataAnalyticsAgent.
        Args:
            name (str): The name of the agent.
            model (str): The model to use.
            instruction (str): The instruction for the agent. This will be inserted into the predefined instruction template.
            con (Backend): The database connection.
            tables (List[execute_sql.Table]): The list of tables.
            today (datetime.datetime, optional): The current date. Defaults to now.
        """

        tool_execute_sql = execute_sql.build_tool(con)
        tool_show_chart = show_chart.build_tool("gemini")
        initial_contexts = [
            types.Part.from_text(text=execute_sql.create_sql_context(
                tables=tables))
        ]
        english_date_str = today.strftime("%B %d, %Y")

        # Format the date as an English date string (e.g., "August 27, 2025")
        super().__init__(
            name=name,
            model=model,
            instruction=DataAnalyticsAgent.Instruction.format(
                custom_instruction=instruction,
                today=english_date_str,
            ).strip(),
            before_model_callback=[
                build_set_context_before_model_callback(
                    initial_contexts=initial_contexts,
                    caching=True,
                    max_context_tokens=100_000)
            ],
            after_tool_callback=build_add_context_after_tool_callback(),
            tools=[tool_execute_sql, tool_show_chart],
            generate_content_config=types.GenerateContentConfig(
                temperature=0.0,
                seed=42,
            ),
            initial_contexts=initial_contexts)

    async def ready(self):
        """
        Prepare the agent for use.
        """

        req = LlmRequest()
        tools = await self.canonical_tools()
        req.append_tools(tools)

        await manage_initial_context_cache(
            initial_contexts=self.initial_contexts,
            model=self.model,
            system_instruction=self.instruction,
            tools=req.config.tools,  # pylint: disable=no-member
            ttl="300s")
