"""
This script downloads data and/or metadata from BigQuery tables and generates a JSONL file for local data agents.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional

from google.cloud import bigquery
import ibis
from ibis.backends.duckdb import Backend

from kaneko_adk.tools import execute_sql

TABLES: List[str] = [
    "users", "products", "orders", "order_items", "inventory_items", "events",
    "distribution_centers"
]


def copy_sql_csv_to_output(output_dir: str) -> None:
    """
    Copies the sql.csv file from the script's directory to the specified output directory.

    Args:
        output_dir (str): The destination directory to copy the file to.
    """
    try:
        script_dir = Path(__file__).parent
        source_path = script_dir / 'sql.csv'
        destination_path = Path(output_dir) / 'sql.csv'

        if not source_path.exists():
            print(
                f"❌ Error: 'sql.csv' not found at {source_path}. Skipping copy."
            )
            return

        shutil.copyfile(source_path, destination_path)
        print(f"✅ Successfully copied 'sql.csv' to {destination_path}")
    except Exception as e:
        print(f"❌ Error: Failed to copy 'sql.csv'. Details: {e}")


def download_table_data(client: bigquery.Client, output_dir: str,
                        table_name: str, sample_ids: Dict[str,
                                                          List[int]]) -> None:
    """
    Downloads table data from BigQuery and saves it as a CSV file,
    filtering by related IDs to ensure a consistent sample.

    Args:
        client (bigquery.Client): The authenticated BigQuery client instance.
        output_dir (str): The directory path to save the CSV file.
        table_name (str): The name of the table to download.
        sample_ids (Dict[str, List[int]]): A dictionary of lists of IDs to filter by.
    """
    print(f"Downloading table data: {table_name}")

    query: str = f"SELECT * FROM `dtt-gcp.test_kaneko_us.{table_name}`"
    filter_clause = ""

    # 各テーブルの関連IDに応じてフィルタリング
    if table_name == "users" and "user_ids" in sample_ids:
        ids_str = ', '.join(map(str, sample_ids["user_ids"]))
        filter_clause = f" WHERE id IN ({ids_str})"
    elif table_name in ["orders", "events"] and "user_ids" in sample_ids:
        ids_str = ', '.join(map(str, sample_ids["user_ids"]))
        filter_clause = f" WHERE user_id IN ({ids_str})"
    elif table_name == "order_items" and "order_ids" in sample_ids:
        ids_str = ', '.join(map(str, sample_ids["order_ids"]))
        filter_clause = f" WHERE order_id IN ({ids_str})"
    elif table_name == "products" and "product_ids" in sample_ids:
        ids_str = ', '.join(map(str, sample_ids["product_ids"]))
        filter_clause = f" WHERE id IN ({ids_str})"
    elif table_name == "inventory_items" and "product_ids" in sample_ids:
        ids_str = ', '.join(map(str, sample_ids["product_ids"]))
        filter_clause = f" WHERE product_id IN ({ids_str})"
    elif table_name == "distribution_centers" and "distribution_center_ids" in sample_ids:
        ids_str = ', '.join(map(str, sample_ids["distribution_center_ids"]))
        filter_clause = f" WHERE id IN ({ids_str})"

    query += filter_clause

    query_job: bigquery.QueryJob = client.query(query)

    try:
        output_path: str = os.path.join(output_dir, f"{table_name}.csv")
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            iterator = query_job.result()
            writer = csv.writer(csvfile)
            writer.writerow([field.name for field in iterator.schema])
            for row in iterator:
                writer.writerow(row.values())
        print(f"✅ Successfully saved {table_name}.csv to {output_path}")
    except Exception as e:
        print(
            f"❌ Error: Failed to download data for {table_name}. Details: {e}")


def download_metadata(client: bigquery.Client, output_dir: str,
                      table_name: str) -> None:
    """
    Retrieves and saves table metadata from BigQuery to a JSON file.

    Args:
        client (bigquery.Client): The authenticated BigQuery client instance.
        output_dir (str): The directory path to save the JSON file.
        table_name (str): The name of the table to retrieve metadata for.
    """
    table_id: str = f"dtt-gcp.test_kaneko_us.{table_name}"

    try:
        table: bigquery.Table = client.get_table(table_id)
        table_info: Dict = {
            "full_table_id": table.full_table_id,
            "description": table.description,
            "num_rows": table.num_rows,
            "num_bytes": table.num_bytes,
            "created": str(table.created),
            "modified": str(table.modified),
            "schema": []
        }

        for field in table.schema:
            field_type: str = field.field_type
            if field_type == "GEOGRAPHY":
                field_type = "GEOMETRY"

            table_info["schema"].append({
                "name": field.name,
                "type": field_type,
                "mode": field.mode,
                "description": field.description
            })

        file_path: str = os.path.join(output_dir, f"{table_name}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(table_info, f, indent=4, ensure_ascii=False)

        print(f"✅ Information exported to {file_path}.")

    except Exception as e:
        print(
            f"❌ Error: Failed to retrieve information for {table_id}. Details: {e}"
        )


def generate_table_info_jsonl(metadata_dir: str, data_dir: str) -> None:
    """
    Integrates downloaded data and metadata to generate a unified JSONL file for tables.

    This function loads metadata from JSON files and corresponding CSV data, then
    creates a single JSONL file with all combined information, including data previews.

    Args:
        metadata_dir (str): The directory containing the JSON metadata files.
        data_dir (str): The directory containing the CSV data files.
    """
    try:
        con: Backend = ibis.duckdb.connect()
        tables_list: List[execute_sql.Table] = []

        sql_csv_path = os.path.join(metadata_dir, 'sql.csv')
        if not os.path.exists(sql_csv_path):
            print(
                f"❌ Error: '{sql_csv_path}' not found. Cannot generate JSONL.")
            return

        sqls_table = con.read_csv(sql_csv_path)

        csv_files = Path(data_dir).glob('*.csv')

        for f in csv_files:
            table_name: str = f.stem

            metadata_path: str = os.path.join(metadata_dir,
                                              f"{table_name}.json")
            if not os.path.exists(metadata_path):
                print(
                    f"⚠️ Warning: Metadata file not found for {table_name}. Skipping JSONL generation for this table."
                )
                continue

            with open(metadata_path, 'r', encoding='utf-8') as file:
                table_info: Dict = json.load(file)

            data_type: Dict = {}
            for item in table_info['schema']:
                column_name: str = item['name']
                column_type: str = item['type']
                data_type[column_name] = column_type

            con.read_csv(f, table_name=table_name, types=data_type)

            sql: List[execute_sql.SQL] = []
            for sql_info in sqls_table.filter(sqls_table['table'] == table_name
                                              ).execute().to_dict('records'):
                sql.append(
                    execute_sql.SQL.model_validate({
                        'query':
                        sql_info['sql'],
                        'description':
                        sql_info['description']
                    }))
            table_info["sql"] = sql

            preview_rows = con.table(table_name).limit(3).execute().to_dict(
                'list')

            csv_buffer: io.StringIO = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=preview_rows.keys())
            writer.writeheader()
            rows: List[Dict] = [
                dict(zip(preview_rows.keys(), row_values))
                for row_values in zip(*preview_rows.values())
            ]
            writer.writerows(rows)

            table_info["preview"] = {"csv": csv_buffer.getvalue()}

            table_info["schemata"] = table_info.get("schema", [])
            table_info["name"] = table_name

            tables_list.append(execute_sql.Table.model_validate(table_info))
        with open(os.path.join(metadata_dir, 'tables.jsonl'),
                  'w',
                  encoding='utf-8') as f:
            for table in tables_list:
                f.write(
                    json.dumps(table.model_dump(mode="json"),
                               ensure_ascii=False) + '\n')

        print(f"✅ Successfully generated tables.jsonl in {metadata_dir}")

    except Exception as e:
        print(f"❌ Error during JSONL generation: {e}")


def main() -> None:
    """
    Main function to parse arguments and execute the download process based on the specified mode.
    """
    parser = argparse.ArgumentParser(
        description="Download BigQuery data or metadata.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["data", "metadata", "all", "info_only"],
        default="all",
        help=
        "Specify 'data' to download table data (CSV), 'metadata' to download table information (JSON), 'all' to do both and generate the final JSONL file, or 'info_only' to download metadata and generate the JSONL file. Default is 'all'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Required: The output directory to save the files.")
    parser.add_argument(
        "--limit-users",
        type=int,
        help=
        "The number of user IDs to sample. By default, it's unlimited and downloads all data."
    )

    args = parser.parse_args()

    client: bigquery.Client = bigquery.Client()

    metadata_dir: str = args.output_dir
    data_dir: str = os.path.join(metadata_dir, "data")

    sample_ids = {}

    try:
        # Step 1: ユーザーIDをサンプリング
        if args.limit_users:
            query_users = f"SELECT id FROM `dtt-gcp.test_kaneko_us.users` ORDER BY RAND() LIMIT {args.limit_users}"
            query_job_users = client.query(query_users)
            sample_ids["user_ids"] = [
                row['id'] for row in query_job_users.result()
            ]
            print(f"Sampled user IDs: {sample_ids['user_ids']}")

            # Step 2: ユーザーIDに基づいて注文IDをサンプリング
            if sample_ids["user_ids"]:
                users_str = ', '.join(map(str, sample_ids["user_ids"]))
                query_orders = f"SELECT order_id FROM `dtt-gcp.test_kaneko_us.orders` WHERE user_id IN ({users_str})"
                query_job_orders = client.query(query_orders)
                sample_ids["order_ids"] = [
                    row['order_id'] for row in query_job_orders.result()
                ]
                print(f"Sampled order IDs: {sample_ids['order_ids']}")

            # Step 3: 注文IDに基づいて商品IDをサンプリング
            if sample_ids.get("order_ids"):
                orders_str = ', '.join(map(str, sample_ids["order_ids"]))
                query_products = f"SELECT product_id FROM `dtt-gcp.test_kaneko_us.order_items` WHERE order_id IN ({orders_str})"
                query_job_products = client.query(query_products)
                sample_ids["product_ids"] = [
                    row['product_id'] for row in query_job_products.result()
                ]
                print(f"Sampled product IDs: {sample_ids['product_ids']}")

            # Step 4: 商品IDに基づいて流通センターIDをサンプリング
            if sample_ids.get("product_ids"):
                products_str = ', '.join(map(str, sample_ids["product_ids"]))
                query_centers = f"SELECT product_distribution_center_id FROM `dtt-gcp.test_kaneko_us.inventory_items` WHERE product_id IN ({products_str})"
                query_job_centers = client.query(query_centers)
                sample_ids["distribution_center_ids"] = [
                    row['product_distribution_center_id']
                    for row in query_job_centers.result()
                ]
                print(
                    f"Sampled distribution center IDs: {sample_ids['distribution_center_ids']}"
                )

    except Exception as e:
        print(f"❌ Error: Failed to sample IDs. Details: {e}")
        return

    copy_sql_csv_to_output(metadata_dir)

    if args.mode == "data" or args.mode == "all":
        os.makedirs(data_dir, exist_ok=True)
        for table_name in TABLES:
            download_table_data(client, data_dir, table_name, sample_ids)

    if args.mode == "metadata" or args.mode == "all":
        os.makedirs(metadata_dir, exist_ok=True)
        for table_name in TABLES:
            download_metadata(client, metadata_dir, table_name)

    if args.mode == "all" or args.mode == "info_only":
        print("\n--- Generating table information JSONL file ---")
        generate_table_info_jsonl(metadata_dir, data_dir)

    print("\n--- Process completed ---")


if __name__ == "__main__":
    main()
