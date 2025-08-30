"""
This script converts a Vega-Lite JSON Schema and generates Pydantic models.

It uses a custom JsonSchemaConverter to filter, resolve references, and simplify
the schema based on a configuration file. It then utilizes the
datamodel-code-generator library to produce Pydantic models from the processed schema.

Arguments:
    --resolve_refs (bool): Resolve $ref references in the schema.
    --claude_schema (bool): Convert the schema to a format suitable for Claude.
    --output_dir (str, required): Output directory for the generated Pydantic models.
"""
import argparse
import json
import os
from pathlib import Path

from config import bar
from converter import JsonSchemaConverter


def main(resolve_refs: bool, output_dir: str):
    """
    Converts JSON Schema and generates Pydantic models from it.

    Args:
        resolve_refs (bool): Whether to resolve $ref references.
        output_dir (str): The directory to save the generated Pydantic models.
    """
    script_dir = os.path.dirname(__file__)
    schema_path = os.path.join(script_dir, "vega-lite.v6.2.0.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    for defs in [bar.defs]:
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
        output_file_path = output_json_dir / f"_auto_generated_{defs['name']}.json"
        with open(output_file_path, "w", encoding="utf-8") as temp_file:
            json.dump(converted_schema,
                      temp_file,
                      indent=4,
                      ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSON Schema and generate Pydantic models.")
    parser.add_argument("--resolve_refs",
                        default=True,
                        action="store_true",
                        help="Resolve $ref references in the schema.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated Pydantic models. (required)")
    args = parser.parse_args()
    main(resolve_refs=args.resolve_refs, output_dir=args.output_dir)
