"""
A module for performing statistical t-tests on data extracted via SQL queries
using a DuckDB connection. It uses a closure to maintain the database connection
and validates SQL queries before execution.
"""
from typing import Optional

from ibis.backends.duckdb import Backend
import numpy as np
from scipy import stats
import sqlglot
from sqlglot.errors import ParseError


def is_single_column_query(sql_query: str) -> bool:
    """
    Checks if an SQL query selects a single column (not a wildcard).

    Args:
        sql_query: The SQL query string to validate.

    Returns:
        bool: True if the query selects exactly one column and not a wildcard;
              False otherwise.
    """
    try:
        parsed_query = sqlglot.parse_one(sql_query)

        selects = parsed_query.find(sqlglot.exp.Select).expressions

        if len(selects) == 1:
            return not isinstance(selects[0], sqlglot.exp.Star)

    except ParseError:
        return False

    return False


def _get_single_column_name(sql_query: str) -> Optional[str]:
    """
    Parses an SQL query to extract the name of the single column being selected.

    This is a helper function to dynamically get the column name for the t-test query.

    Args:
        sql_query: The SQL query string.

    Returns:
        str: The name or alias of the selected column. Returns None if not a single column.
    """
    try:
        parsed_query = sqlglot.parse_one(sql_query)
        selects = parsed_query.find(sqlglot.exp.Select).expressions
        if len(selects) == 1:
            select_expression = selects[0]
            if isinstance(select_expression, sqlglot.exp.Alias):
                return select_expression.alias.name
            elif isinstance(select_expression, sqlglot.exp.Column):
                return select_expression.name
            else:
                return "value"  # Default to 'value' if a simple name can't be extracted
    except ParseError:
        return None
    return None


def create_ttest_runner(conn: Backend):
    """
    Creates a closure that performs a t-test using a pre-configured DuckDB connection.

    This function returns an inner function that can be called repeatedly
    with different SQL queries, without needing to pass the connection object each time.

    Args:
        conn: The DuckDB database connection object.

    Returns:
        function: A function that takes two SQL queries and returns the p-value of the t-test.
    """

    def ttest_runner(sql_query_a: str, sql_query_b: str):
        """
        Executes a t-test by first retrieving summary statistics from DuckDB and
        then performing the final calculation in Python.

        Args:
            sql_query_a (str): The SQL query to extract data for group A. Must return a single column.
            sql_query_b (str): The SQL query to extract data for group B. Must return a single column.

        Returns:
            dict: A dictionary containing the p-value or an error message.
        """
        # ユーザーの元の関数を使用してバリデーションチェックを行います
        if not is_single_column_query(sql_query_a):
            return {
                "error":
                    f"Group A query must select a single column: {sql_query_a}"
            }
        if not is_single_column_query(sql_query_b):
            return {
                "error":
                    f"Group B query must select a single column: {sql_query_b}"
            }

        # クエリから列名を動的に取得します
        col_name_a = _get_single_column_name(sql_query_a)
        col_name_b = _get_single_column_name(sql_query_b)

        if not col_name_a or not col_name_b:
            return {"error": "Could not determine column names from queries."}

        stats_query = f"""
        WITH
        group_a_stats AS (
            SELECT
                COUNT(*) as n,
                AVG({col_name_a}) as mean,
                STDDEV_SAMP({col_name_a}) as std_dev,
                'a' AS group_name
            FROM ({sql_query_a}) AS temp_a
        ),
        group_b_stats AS (
            SELECT
                COUNT(*) as n,
                AVG({col_name_b}) as mean,
                STDDEV_SAMP({col_name_b}) as std_dev,
                'b' AS group_name
            FROM ({sql_query_b}) AS temp_b
        )
        SELECT * FROM group_a_stats
        UNION ALL
        SELECT * FROM group_b_stats;
        """

        try:
            df = conn.sql(stats_query).execute()
        except Exception as e:
            return {"error": f"Query execution error: {str(e)}"}

        if len(df) != 2:
            return {"error": "Failed to retrieve statistics for both groups."}

        stats_a = df[df['group_name'] == 'a'].iloc[0]
        stats_b = df[df['group_name'] == 'b'].iloc[0]

        n_a, mean_a, std_dev_a = stats_a['n'], stats_a['mean'], stats_a[
            'std_dev']
        n_b, mean_b, std_dev_b = stats_b['n'], stats_b['mean'], stats_b[
            'std_dev']

        problem_groups = []
        if n_a < 2:
            problem_groups.append("Group A")
        if n_b < 2:
            problem_groups.append("Group B")
        if problem_groups:
            return {
                "error":
                    f"T-test requires a minimum of 2 observations per group. This was not met for {', '.join(problem_groups)}."
            }

        problem_groups = []
        if std_dev_a == 0:
            problem_groups.append("Group A")
        if std_dev_b == 0:
            problem_groups.append("Group B")
        if problem_groups:
            return {
                "error":
                    f"T-test cannot be performed due to zero standard deviation. This was observed in {', '.join(problem_groups)}."
            }

        s_a_sq = std_dev_a**2
        s_b_sq = std_dev_b**2

        numerator = mean_a - mean_b
        denominator_sq = (s_a_sq / n_a) + (s_b_sq / n_b)
        t_statistic = numerator / np.sqrt(denominator_sq)

        df_numerator = denominator_sq**2
        df_denominator_a = (s_a_sq / n_a)**2 / (n_a - 1)
        df_denominator_b = (s_b_sq / n_b)**2 / (n_b - 1)
        df_value = df_numerator / (df_denominator_a + df_denominator_b)

        p_value = stats.t.sf(np.abs(t_statistic), df=df_value) * 2

        if np.isnan(p_value):
            p_value = "NaN"
        else:
            p_value = round(float(p_value), 3)

        return {"p_value": p_value}

    return ttest_runner
