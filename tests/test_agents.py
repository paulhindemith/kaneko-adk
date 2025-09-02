"""
Test agents
"""
# pylint: disable=redefined-outer-name

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types
import ibis
from ibis.backends.duckdb import Backend
import pandas as pd
import pytest

from kaneko_adk.agents import DataAnalyticsAgent

TEST_DIR_PATH = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def connect_for_test() -> Tuple[Backend, List[DataAnalyticsAgent.Table]]:
    """
    Connect to a test DuckDB and load table metadata.
    Uses data from the 'test/tables' directory.
    """
    _con: Backend = ibis.duckdb.connect()
    _tbls: List[DataAnalyticsAgent.Table] = []

    sql_csv_path = os.path.join(TEST_DIR_PATH, "tables", "sql.csv")
    sqls = pd.read_csv(sql_csv_path)

    csv_files = Path(os.path.join(TEST_DIR_PATH, "tables",
                                  "data")).glob('*.csv')
    for f in csv_files:
        table_name = f.stem
        table_json_path = os.path.join(TEST_DIR_PATH, "tables",
                                       f"{table_name}.json")
        with open(table_json_path, 'r', encoding='utf-8') as file:
            _tbl: Dict = json.load(file)

        data_type = {
            item['name']: item['type']
            for item in _tbl.get('schema', [])
        }
        _con.read_csv(f, table_name=table_name, types=data_type)

        sql = []
        for sql_info in sqls[sqls['table'] == table_name].itertuples():
            sql.append(
                DataAnalyticsAgent.SQL.model_construct(
                    query=sql_info.sql, description=sql_info.description))
        _tbl["sql"] = sql

        _tbl["preview"] = {
            "csv":
            _con.table(table_name).to_pandas(limit=3).to_csv(index=False)
        }
        _tbl["schemata"] = _tbl.get("schema", [])
        _tbl["name"] = table_name
        _tbl["description"] = _tbl.get("description", "")

        _tbls.append(DataAnalyticsAgent.Table.model_validate(_tbl))

    return _con, _tbls


@pytest.fixture(scope="module")
async def agent() -> DataAnalyticsAgent:
    """Fixture to generate a DataAnalyticsAgent instance for testing."""
    con, tables = connect_for_test()
    jst = datetime.timezone(datetime.timedelta(hours=9))

    daa = DataAnalyticsAgent(
        name="test",
        model="gemini-2.5-flash",
        instruction="日本語で回答すること。",
        con=con,
        tables=tables,
        today=datetime.datetime(2025, 8, 27, tzinfo=jst),
    )
    await daa.ready()
    return daa


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt, required_tools", [
    ("Sales trends up to this month", {'execute_sql'}),
    ("What's selling well recently?", {'execute_sql'}),
    ("Current status of member data", {'execute_sql'}),
])
async def test_data_analytics_agent(agent: DataAnalyticsAgent, prompt: str,
                                    required_tools: Set[str]):
    """
    Test various scenarios with flexible rules for tool chaining.

    This test verifies that the agent for each scenario:
    1. Starts with a text explanation and an immediate SQL call.
    2. Uses all the required tools for that scenario at least once.
    3. Ends with a text-only summary.

    Args:
        agent (DataAnalyticsAgent): The agent instance to test.
        prompt (str): The user prompt for the scenario.
        required_tools (Set[str]): The set of required tool names.

    Returns:
        None
    """
    app_name = "test_app"
    user_id = "test_user"
    session_id = f"test_session_{prompt[:10]}"

    user_message = types.Content(role='user',
                                 parts=[types.Part.from_text(text=prompt)])
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    await runner.session_service.create_session(app_name=app_name,
                                                user_id=user_id,
                                                session_id=session_id)
    event_generator = runner.run_async(user_id=user_id,
                                       session_id=session_id,
                                       new_message=user_message)

    events = []
    async for e in event_generator:
        if e.content and e.content.parts:
            events.append(e)
        if len(events) >= 20:
            logger.warning(f"Event limit reached for '{prompt}'.")
            break
    assert len(
        events
    ) > 1, f"Conversation for '{prompt}' should have at least 2 events."

    # Helper to find parts in an event
    def find_part(event: Event, part_type: str, name=None):
        for part in event.content.parts:
            attr = getattr(part, part_type, None)
            if attr:
                if name is None or (hasattr(attr, 'name')
                                    and attr.name == name):
                    return attr
        return None

    # Rule 1: Check the first event's structure
    first_event = events[0]
    assert find_part(first_event, 'text') is not None, \
        f"First event for '{prompt}' must contain text."
    assert find_part(first_event, 'function_call', 'execute_sql') is not None, \
        f"First event for '{prompt}' must call 'execute_sql'."

    # Rule 2: Check if all required tools were used throughout the conversation
    all_function_calls = {
        part.function_call.name
        for event in events
        for part in event.content.parts if part.function_call
    }
    assert required_tools.issubset(all_function_calls), \
        f"Tool usage mismatch for '{prompt}'. Expected at least {required_tools}, got {all_function_calls}."

    # Rule 3: Check the final event's structure
    last_event = events[-1]
    assert find_part(last_event, 'text') is not None, \
        f"Last event for '{prompt}' must be a text summary."
    assert find_part(last_event, 'function_call') is None, \
        f"Last event for '{prompt}' must not contain function calls."
