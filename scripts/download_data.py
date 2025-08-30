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
from typing import Dict, List

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
                        table_name: str) -> None:
    """
    Downloads table data from BigQuery and saves it as a CSV file.

    Args:
        client (bigquery.Client): The authenticated BigQuery client instance.
        output_dir (str): The directory path to save the CSV file.
        table_name (str): The name of the table to download.
    """
    print(f"Downloading table data: {table_name}")

    query: str = f"SELECT * FROM `dtt-gcp.test_kaneko_us.{table_name}`"
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

        dir_path = os.path.dirname(os.path.abspath(__file__))
        sqls_table = con.read_csv(os.path.join(dir_path, 'sql.csv'))

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

    args = parser.parse_args()

    client: bigquery.Client = bigquery.Client()

    metadata_dir: str = args.output_dir
    data_dir: str = os.path.join(metadata_dir, "data")

    copy_sql_csv_to_output(metadata_dir)

    if args.mode == "data" or args.mode == "all":
        os.makedirs(data_dir, exist_ok=True)
        for table_name in TABLES:
            download_table_data(client, data_dir, table_name)

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
