import argparse
import json
import os
from pathlib import Path

from config import defs
from converter import JsonSchemaConverter


def main(resolve_refs: bool, source_file: str, output_dir: str):
    """
    Converts JSON Schema and generates Pydantic models from it.

    Args:
        resolve_refs (bool): Whether to resolve $ref references.
        source_file (str): The path to the source JSON schema file.
        output_dir (str): The directory to save the generated Pydantic models.
    """
    with open(source_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    converter = JsonSchemaConverter(
        includes=defs.get("includes", []),
        excludes=defs.get("excludes", []),
        unresolving_definitions=defs.get("unresolving_definitions", []),
        replace=defs.get("replace", {}),
        remove_fields=defs.get("remove_fields", []),
    )
    converted_schema = converter.convert(schema, resolve_refs=resolve_refs)

    output_json_dir = Path(output_dir)
    output_json_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_json_dir / f"_auto_generated_{defs['name']}_schema.json"
    with open(output_file_path, "w", encoding="utf-8") as temp_file:
        json.dump(converted_schema, temp_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSON Schema and generate Pydantic models.")
    parser.add_argument("--resolve_refs",
                        default=True,
                        action="store_true",
                        help="Resolve $ref references in the schema.")
    parser.add_argument("--source",
                        dest="source_file",
                        type=str,
                        required=True,
                        help="Source JSON schema file path. (required)")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated Pydantic models. (required)")
    args = parser.parse_args()
    main(resolve_refs=args.resolve_refs,
         source_file=args.source_file,
         output_dir=args.output_dir)
