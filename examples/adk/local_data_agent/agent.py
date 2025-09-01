"""
Local data agent
"""
import json
import os
from pathlib import Path
from typing import Dict, Tuple

from google.adk.agents import LlmAgent
from google.genai import types
import ibis
from ibis.backends.duckdb import Backend
import pandas as pd

from kaneko_adk.callbacks import build_add_context_after_tool_callback
from kaneko_adk.callbacks import build_set_context_before_model_callback
from kaneko_adk.tools import execute_sql
from kaneko_adk.tools import show_chart

DIR_PATH = os.path.dirname(__file__)
MAX_ROWS = execute_sql.MAX_ROWS


def connect() -> Tuple[Backend, list[execute_sql.Table]]:
    """
    Connect to DuckDB and return the connection and table metadata.
    """

    _con: Backend = ibis.duckdb.connect()
    _tbls: list[execute_sql.Table] = []

    sqls = pd.read_csv(os.path.join(DIR_PATH, "tables", "sql.csv"))

    csv_files = Path(os.path.join(DIR_PATH, "tables", "data")).glob('*.csv')
    for f in csv_files:
        table_name = f.stem
        table_name_path = os.path.join(DIR_PATH, "tables",
                                       f"{table_name}.json")
        with open(table_name_path, 'r', encoding='utf-8') as file:
            _tbl: Dict = json.load(file)
            del _tbl["full_table_id"]
        data_type = {}
        for item in _tbl['schema']:
            column_name = item['name']
            column_type = item['type']
            data_type[column_name] = column_type
        _con.read_csv(f, table_name=table_name, types=data_type)

        sql = []
        for sql_info in sqls[sqls['table'] == table_name].itertuples():
            description = sql_info.description
            query = sql_info.sql
            sql.append(
                execute_sql.SQL.model_construct(query=query,
                                                description=description))
        _tbl["sql"] = sql

        _tbl["preview"] = {
            "csv":
            _con.table(table_name).to_pandas(limit=3).to_csv(index=False)
        }
        _tbl["schemata"] = _tbl.get("schema", [])
        _tbl["name"] = table_name
        _tbl["description"] = _tbl.get("description", "")
        _tbl["num_rows"] = _tbl.get("num_rows", 0)
        _tbl["num_bytes"] = _tbl.get("num_bytes", 0)
        _tbl["created"] = _tbl.get("created", "")
        _tbl["modified"] = _tbl.get("modified", "")

        _tbls.append(execute_sql.Table.model_validate(_tbl))
    return _con, _tbls


con, tables = connect()
initial_contexts = [
    types.Part.from_text(text=execute_sql.create_sql_context(tables=tables))
]
tool_execute_sql = execute_sql.build_tool(con)
tool_show_chart = show_chart.build_tool("gemini")

INSTRUCTION = """
You are an assistant that helps users analyze data. Please provide all output in Japanese.

1. **Summarize only:** Communicate only the key points, conclusions, or insights from the analysis in natural conversational Japanese. Do not include raw data directly in your answers.

2. **Base answers only on provided data:** Always base your responses strictly on the data presented.

3. **Date:** Assume today's date is 2025-08-27.

When using tools, inform the user of the intended action beforehand.
"""

root_agent = LlmAgent(
    name="local_data_agent",
    model="gemini-2.5-flash",
    instruction=INSTRUCTION,
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
    ))
