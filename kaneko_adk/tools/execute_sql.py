"""
Tool to execute SQL
"""
import json
import tempfile
from typing import Callable

from ibis.backends.duckdb import Backend as DuckdbBackend
import pandas as pd
from pydantic import BaseModel
from pydantic import Field

MAX_ROWS = 100


class Schema(BaseModel):
    """Table schema information"""

    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column type")
    description: str = Field(..., description="Column description")


class SQL(BaseModel):
    """SQL statement information"""

    query: str = Field(..., description="SQL query")
    description: str = Field(..., description="SQL description")


class Preview(BaseModel):
    """Table preview information. Only 3 samples"""

    csv: str = Field(..., description="Sample data for preview in CSV format")


class Table(BaseModel):
    """Table information"""

    name: str = Field(..., description="Table name")
    description: str = Field(..., description="Table description")
    num_rows: int = Field(..., description="Number of rows in the table")
    num_bytes: int = Field(..., description="Table size (in bytes)")
    created: str = Field(..., description="Table creation date")
    modified: str = Field(..., description="Table last modified date")
    sql: list[SQL] = Field(...,
                           description="SQL statements related to the table")
    schemata: list[Schema] = Field(..., description="Table schema information")
    preview: Preview = Field(..., description="Table preview information")


def create_sql_context(tables: list[Table]) -> str:
    """Create SQL context
    Args:
        tables (list[Table]): List of tables
    Returns:
        str: SQL context
    """

    context = []
    for table in tables:
        d = table.model_dump(mode="json", exclude={"sql", "preview"})

        # Convert SQL to CSV
        df = pd.DataFrame([sql.model_dump(mode="json") for sql in table.sql])
        d["sql"] = df.to_csv(index=False)

        # Convert preview to CSV
        df = pd.DataFrame([row for row in table.preview.csv])
        d["preview"] = df.head(3).to_csv(index=False)

        context.append(d)

    return json.dumps(context, ensure_ascii=False)


def build_tool(con: DuckdbBackend, add_context: bool = False) -> Callable:
    "Build a tool to execute SQL"

    def execute_sql(query: str) -> str:
        """
        Executes an SQL query on DuckDB and returns the result as a string.
        Only the first 100 rows of the result are returned.
        Args:
            query (str): The SQL query to execute. If the query does not include a LIMIT clause, LIMIT 100 will be added automatically.
        Returns:
            dict: A dictionary with keys:
            - 'result': The CSV string of the first 100 rows of the query result.
            - 'path': The file path where the CSV result is saved.
        """
        try:
            df = con.sql(query).execute()

            # Return only the first 100 rows as CSV
            res = df.head(MAX_ROWS).to_csv(index=False)

            file_path = ""
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=".csv") as tmpfile:
                tmpfile.write(res.encode())
                file_path = tmpfile.name

            res = {
                "path": file_path,
                "mime_type": "text/csv",
                "content": res,
            }
            if add_context:
                return {"_context": res}
            return res
        except Exception as e:
            return {"error": str(e)}

    return execute_sql
