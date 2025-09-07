"""
Local data agent
"""
import datetime
import json
import os
from pathlib import Path
from typing import Dict, Tuple

from google.adk.models.lite_llm import LiteLlm
import ibis
from ibis.backends.duckdb import Backend
import pandas as pd

from kaneko_adk.agents import DataAnalyticsAgent, Sql, Table

DIR_PATH = os.path.dirname(__file__)


def connect() -> Tuple[Backend, list[Table]]:
    """
    Connect to DuckDB and return the connection and table metadata.
    """

    _con: Backend = ibis.duckdb.connect()
    _tbls: list[DataAnalyticsAgent.Table] = []

    sqls = pd.read_csv(os.path.join(DIR_PATH, "tables", "sql.csv"))

    csv_files = Path(os.path.join(DIR_PATH, "tables", "data")).glob('*.csv')
    for f in csv_files:
        table_name = f.stem
        table_name_path = os.path.join(DIR_PATH, "tables", f"{table_name}.json")
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
                Sql.model_construct(query=query, description=description)
            )
        _tbl["sql"] = sql

        _tbl["preview"] = {
            "csv": _con.table(table_name).to_pandas(limit=3
                                                   ).to_csv(index=False)
        }
        _tbl["schemata"] = _tbl.get("schema", [])
        _tbl["name"] = table_name
        _tbl["description"] = _tbl.get("description", "")
        _tbl["num_rows"] = _tbl.get("num_rows", 0)
        _tbl["num_bytes"] = _tbl.get("num_bytes", 0)
        _tbl["created"] = _tbl.get("created", "")
        _tbl["modified"] = _tbl.get("modified", "")

        _tbls.append(Table.model_validate(_tbl))
    return _con, _tbls


def build_agent(
    con: Backend,
    tables: list[Table],
    instruction: str = "日本語で回答すること。",
    model: str | LiteLlm = "gemini-2.5-flash"
) -> DataAnalyticsAgent:
    """Returns the root_agent instance."""
    jst = datetime.timezone(datetime.timedelta(hours=9))

    return DataAnalyticsAgent(
        name="local_data_agent",
        model=model,
        instruction=instruction,
        con=con,
        tables=tables,
        today=datetime.datetime(2025, 8, 27, tzinfo=jst)
    )


con, tables = connect()
root_agent = build_agent(con, tables)
