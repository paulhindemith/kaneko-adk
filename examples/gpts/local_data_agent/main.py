"""
Fast API server for querying a local DuckDB database
"""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

from fastapi import Depends
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import status
import ibis
from ibis.backends.duckdb import Backend
import pandas as pd
from pydantic import BaseModel
from pydantic import Field

from kaneko_adk.tools import execute_sql

DB_CONN = None
API_KEY = os.environ["API_KEY"]
DIR_PATH = os.path.dirname(__file__)


class QueryRequest(BaseModel):
    """Request body schema defined in the OpenAPI specification"""
    query: str = Field(
        ...,
        description=
        "The SQL query to execute. If the query does not include a LIMIT clause, LIMIT 100 will be added automatically."
    )


class OpenAPIFileResponseItem(BaseModel):
    """ Result for SQL query execution """
    name: str = Field(..., description="Name of the file")
    mime_type: str = Field(..., description="MIME type of the file")
    content: str = Field(..., description="Base64-encoded content of the file")


class QueryResponse(BaseModel):
    """
    Response schema defined in the OpenAPI specification
    """
    openaiFileResponse: list[OpenAPIFileResponseItem]


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


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Connect to DuckDB when the server starts and close the connection when it shuts down.
    """
    try:
        global DB_CONN
        DB_CONN, _ = connect()
        print("Connected to DuckDB.")
        # Run startup events
        yield
    except Exception as e:
        print(f"Failed to connect to DuckDB: {e}", file=sys.stderr)
        sys.exit(1)


app = FastAPI(
    title="Database for THELook eCommerce",
    version="1.0.0",
    servers=[{
        "url":
        "https://test-kaneko-chatgpt-gpts-thelook-ecommerce-467677818012.asia-northeast1.run.app",
        "description": "middleware service"
    }],
    description="Query THELook eCommerce database",
    lifespan=lifespan)


def verify_api_key(x_api_key: str = Header(...)):
    """
    Args:
        x_api_key (str): API key extracted from the request header.

    Returns:
        bool: True if authentication is successful.

    Raises:
        HTTPException: Raises 401 error if the API key is invalid.
    """
    if x_api_key != API_KEY:
        print(f"Invalid API Key: {x_api_key}", file=sys.stderr)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Invalid API Key: {x_api_key}")
    return True


@app.post("/api/query",
          summary="Query THELook eCommerce database",
          operation_id="executeSQL",
          response_model=QueryResponse)
def call(body: QueryRequest, _: bool = Depends(verify_api_key)):
    """
    Executes an SQL query on DuckDB and returns the result as a string.
    Only the first 100 rows of the result are returned.
    """
    query = body.query

    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Query string is empty.")

    res = execute_sql.build_tool(DB_CONN)(query)

    if res.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {res.get('error')}")

    return {
        "openaiFileResponse": [{
            "name": res.get("path", ""),
            "mime_type": res.get("mime_type", ""),
            "content": res.get("content", "")
        }]
    }
