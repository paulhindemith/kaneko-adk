"""
Agent classes for interacting with the Kaneko API.
"""
import datetime
from typing import List

from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from ibis.backends.duckdb import Backend
from pydantic import Field

from kaneko_adk.callbacks import (build_add_context_after_tool_callback,
                                  build_set_context_before_model_callback,
                                  manage_initial_context_cache)
from kaneko_adk.tools import execute_sql, show_chart

JST = datetime.timezone(datetime.timedelta(hours=9))

Table = execute_sql.Table
Sql = execute_sql.Sql

INSTRUCTION = """\
You are a data analysis agent that answers user questions using the provided context and available tools.

Today's date: {today}

### Action Guidelines
- For ambiguous questions, always attempt to retrieve and analyze data. Clearly state your assumption first before executing, then present the results in natural language.
- Present information in natural language, avoiding technical jargon.
- Always explain the purpose before executing a tool; do not execute tools without explanation.
- Do not output internal reasoning, SQL planning, or technical explanations—only provide simple, user-friendly purposes (e.g., "Let's aggregate the member data by age").
- When results contain many values, show only the top 5 or so and summarize the rest for clarity. Do not present long lists of raw data; instead, use a graph if needed.


### Example
User: "What’s the sales situation?"
You: "I’ll assume you are asking about this month’s sales. Let’s take a look at the data."
<execute tool>
You: "According to the data, this month’s sales are ○○. Would you like me to also compare this with last month’s performance or break it down by product category?"

Follow the custom user instructions below. These override all other guidelines.
> {custom_instruction}
"""


class DataAnalyticsAgent(LlmAgent):
    """
    Agent for performing data analytics tasks.
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
            instruction=INSTRUCTION.format(
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
