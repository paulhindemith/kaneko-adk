"""
JsonSchemaConverter is a class for converting JSON Schema documents.
"""

import copy
import random

from google.genai import types
from jsonpath_ng import Fields
from jsonpath_ng import Index
from jsonpath_ng.ext import parse
from referencing import Registry
from referencing import Resource
from referencing.jsonschema import DRAFT7


class JsonSchemaConverter:
    """Class for converting JSON Schema."""

    ALLOWED_TYPES = [
        "string", "number", "integer", "boolean", "array", "object"
    ]
    ALLOWED_SCHEMA_FIELDS = {
        "$id", "$schema", "$defs", "$ref", "type", "title", "description",
        "enum", "items", "anyOf", "properties", "required"
    }

    def __init__(self,
                 includes: list[str] = None,
                 excludes: list[str] = None,
                 unresolving_definitions: list[str] = None,
                 replace: dict[str, list[str]] = None,
                 remove_fields: list[str] = None):
        """
        Initialize the JsonSchemaConverter.

        Args:
            includes (list[str]): List of JSON paths to include.
            excludes (list[str]): List of JSON paths to exclude.
            unresolving_definitions (list[str]): List of unresolved JSON paths.
            replace (dict[str, list[str]]): Dictionary of replacement rules.
            remove_fields (list[str]): List of fields to remove.
        """
        self.includes = includes or []
        self.excludes = excludes or []
        self.unresolving_definitions = unresolving_definitions or []
        self.replace = replace or {}
        self.remove_fields = remove_fields or []

    def is_subpath(self, super_path: str, sub_path: str) -> bool:
        """
        Check if a JSON Path is a subpath of another JSON Path.

        Args:
            super_path (str): Parent JSON Path string.
            sub_path (str): JSON Path string to check as subpath.

        Returns:
            bool: True if sub_path is a subpath of super_path, otherwise False.
        """
        parsed_super = parse(super_path)
        parsed_sub = parse(sub_path)

        if parsed_sub == parsed_super:
            return True

        current_path = parsed_sub
        while hasattr(current_path, 'left') and hasattr(current_path, 'right'):
            if current_path.left == parsed_super:
                return True
            current_path = current_path.left

        if super_path == '$' and sub_path != '$':
            return True

        return False

    def _filter(self, source: dict) -> dict:
        """
        Filter schema based on specified paths.

        Args:
            source (dict): Original JSON.

        Returns:
            dict: Filtered JSON.
        """
        results = {}

        for path_str in self.includes:
            jsonpath_expr = parse(path_str)
            for match in jsonpath_expr.find(source):
                jsonpath_expr.update_or_create(results, match.value)
        for path_str in self.excludes:
            jsonpath_expr = parse(path_str)

            matches = sorted(jsonpath_expr.find(results),
                             key=lambda m: str(m.full_path),
                             reverse=True)
            for match in matches:
                parent_context = match.context
                if parent_context is None:
                    continue

                parent_value = parent_context.value
                path_part = match.path

                try:
                    if isinstance(parent_value, dict) and isinstance(
                            path_part, Fields):
                        key = path_part.fields[0]
                        del parent_value[key]
                    elif isinstance(parent_value, list) and isinstance(
                            path_part, Index):
                        index = path_part.index
                        if isinstance(parent_value[index], dict):
                            parent_value[index] = {}
                        elif isinstance(parent_value[index], list):
                            parent_value[index] = []
                        else:
                            parent_value[index] = None

                except (KeyError, IndexError, AttributeError):
                    pass
        return results

    def _replace(self, schema: dict) -> dict:
        """
        Replace values at specified JSON Paths.

        Args:
            schema (dict): Target JSON Schema.

        Returns:
            dict: JSON Schema after replacement.
        """
        for path_str, value in self.replace.items():
            jsonpath_expr = parse(path_str)
            jsonpath_expr.update_or_create(schema, value)
        return schema

    def _resolve(self, schema: dict) -> dict:
        """
        Selectively resolve $ref in JSON Schema (using jsonpath-ng and referencing).

        Args:
            schema (dict): Target JSON Schema.

        Returns:
            dict: New JSON Schema after processing.
        """
        schema_copy = copy.deepcopy(schema)

        insert_id = False
        if "$id" not in schema_copy:
            schema_copy["$id"] = "urn:local-schema"
            insert_id = True

        # Expand top-level $ref to avoid errors.
        if '$ref' in schema_copy:
            root_ref_path = schema_copy['$ref']
            definition_name = root_ref_path.split('/')[-1]
            definition_path_str = f"$.definitions.{definition_name}"
            ref = schema_copy["definitions"][definition_name]
            schema_copy.update(ref)
            del schema_copy['$ref']

        resource = Resource.from_contents(schema_copy,
                                          default_specification=DRAFT7)

        registry = resource @ Registry()
        resolver = registry.resolver(base_uri=schema_copy["$id"])

        while True:

            ref_finder_expr = parse("$..*")

            all_ref_contexts = [
                match.context for match in ref_finder_expr.find(schema_copy)
                if isinstance(match.value, str) and str(match.path) == "'$ref'"
            ]
            if not all_ref_contexts:
                break

            resolved_in_this_pass = False
            for context in all_ref_contexts:
                ref_node = context.value
                ref_string = ref_node['$ref']
                if not ref_string.startswith('#/definitions/'):
                    continue
                definition_name = ref_string.split('/')[-1]
                definition_path_str = f"$.definitions.{definition_name}"
                if definition_path_str in self.unresolving_definitions:
                    continue

                try:
                    resolved_content = resolver.lookup(ref_string).contents
                    merged_content = {**resolved_content, **ref_node}
                    del merged_content['$ref']

                    context.full_path.update(schema_copy,
                                             copy.deepcopy(merged_content))

                    resolved_in_this_pass = True
                    break
                except Exception as e:
                    print(
                        f"Warning: Could not resolve ref '{ref_string}': {e}")
            if not resolved_in_this_pass:
                break

        if insert_id:
            del schema_copy["$id"]

        # Remove definitions except those in unresolving_definitions.
        for definition_name in list(schema_copy.get("definitions", {}).keys()):
            definition_path_str = f"$.definitions.{definition_name}"
            if definition_path_str not in self.unresolving_definitions:
                del schema_copy["definitions"][definition_name]
        return schema_copy

    def _simplify(self, node):
        """
        Simplify the schema by cleaning up empty values and flattening anyOf.

        Args:
            node (dict | list): Node to simplify.

        Returns:
            dict | list: Simplified node.
        """
        if isinstance(node, dict):
            if 'anyOf' in node and isinstance(node['anyOf'], list):
                node['anyOf'] = [item for item in node['anyOf'] if item]
                if not node['anyOf']:
                    del node['anyOf']
            for key, value in list(node.items()):
                if value == {}:
                    del node[key]
                self._simplify(value)
            if 'anyOf' in node and isinstance(node['anyOf'], list) and len(
                    node['anyOf']) == 1:
                content = node.pop('anyOf')[0]
                if isinstance(content, dict):
                    node.update(content)
        elif isinstance(node, list):
            for item in node:
                self._simplify(item)
        return node

    def _convert_draft_2020_12(self, data: dict, current_json_path: str = '$'):
        """
        Convert old JSON Schema to Draft 2020-12 format.

        Args:
            data (dict): JSON Schema data to convert.
            current_json_path (str, optional): Current JSON path. Default is '$'.

        Returns:
            dict: Converted JSON Schema.
        """
        if current_json_path == '$':
            data["$schema"] = "http://json-schema.org/draft-2020-12/schema"
        if current_json_path == '$' and 'definitions' in data:
            data['$defs'] = data.pop('definitions')
        for key, value in list(data.items()):
            json_path = f"{current_json_path}.{key}"

            # Convert '#/definitions/' to '#/$defs/' in $ref fields.
            if key == '$ref' and isinstance(
                    value, str) and value.startswith('#/definitions/'):
                new_ref_value = value.replace('#/definitions/', '#/$defs/')
                data[key] = new_ref_value

            if isinstance(value, dict):
                self._convert_draft_2020_12(value, json_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._convert_draft_2020_12(
                            item, current_json_path=f"{json_path}[{i}]")

        return data

    def _convert_llm_schema(self,
                            data: dict,
                            root_schema: dict = None,
                            current_json_path: str = '$'):
        """
        Convert to LLM JSON Schema.

        Args:
            data (dict): JSON Schema data to convert.
            root_schema (dict, optional): Root schema. Default is None.
            current_json_path (str, optional): Current JSON path. Default is '$'.

        Returns:
            dict: Converted JSON Schema.
        """

        def _process_const(target_dict: dict):
            """
            Convert 'const' to 'enum' or remove it.

            Args:
                target_dict (dict): Target dictionary.
            """
            if "const" in target_dict:
                if "type" in target_dict and not isinstance(
                        target_dict["const"], list):
                    const_value = target_dict.pop("const")
                    target_dict["enum"] = [const_value]
                else:
                    target_dict.pop("const")

        def _process_enum_none_and_limit(target_dict: dict):
            """
            Remove None from 'enum' and limit to 50 elements.

            Args:
                target_dict (dict): Target dictionary.
            """
            if "enum" in target_dict and isinstance(target_dict["enum"], list):
                filtered_enum = [
                    item for item in target_dict["enum"] if item is not None
                ]

                if len(filtered_enum) > 50:
                    random.seed(42)
                    filtered_enum = random.sample(filtered_enum, 50)

                if not filtered_enum:
                    del target_dict["enum"]
                else:
                    target_dict["enum"] = filtered_enum

        def _process_ref_coexistence(target_dict: dict):
            """
            Remove fields other than $ref and fields starting with $ if $ref is set.

            Args:
                target_dict (dict): Target dictionary.
            """
            if "$ref" in target_dict:
                keys_to_delete = []
                for k in list(target_dict.keys()):
                    if k != "$ref" and not k.startswith('$'):
                        keys_to_delete.append(k)
                for k_del in keys_to_delete:
                    del target_dict[k_del]

        def _process_remove_fields(target_dict: dict):
            """
            Remove unnecessary fields.

            Args:
                target_dict (dict): Target dictionary.
            """
            for k in self.remove_fields:
                if k in target_dict:
                    del target_dict[k]

        def _process_type_fields(target_dict: dict, current_json_path: str):
            """
            Convert and clean up 'type' fields.

            Args:
                target_dict (dict): Target dictionary.
                current_json_path (str): Current JSON path.
            """
            path_segments = current_json_path.split('.')

            if "type" in target_dict:
                if len(path_segments
                       ) >= 2 and path_segments[-2] == 'properties':

                    modified = False
                    if isinstance(target_dict["type"], dict):
                        del target_dict["type"]
                        modified = True

                    elif isinstance(target_dict["type"], list):
                        new_types = []
                        for _, item_type in enumerate(target_dict["type"]):
                            if isinstance(item_type,
                                          str) and item_type == "null":
                                modified = True
                            elif isinstance(
                                    item_type,
                                    str) and item_type in self.ALLOWED_TYPES:
                                new_types.append(item_type)
                            else:
                                modified = True

                        if modified:
                            if new_types:
                                data[key] = new_types
                            else:
                                del data[key]

                    elif isinstance(target_dict["type"], str):
                        if target_dict["type"] == "null":
                            del data[key]
                            modified = True
                        elif target_dict["type"] not in self.ALLOWED_TYPES:
                            del data[key]
                            modified = True

        _process_const(data)
        _process_enum_none_and_limit(data)
        _process_ref_coexistence(data)
        _process_remove_fields(data)
        _process_type_fields(data, current_json_path)
        for key, value in list(data.items()):
            json_path = f"{current_json_path}.{key}"

            if isinstance(value, dict):
                self._convert_llm_schema(value, root_schema, json_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._convert_llm_schema(
                            item,
                            root_schema,
                            current_json_path=f"{json_path}[{i}]")
        return data

    def _validate_fields(self, data: dict, current_json_path: str = '$'):
        """
        Validate schema fields.

        Args:
            data (dict): Data to validate.
            current_json_path (str, optional): Current JSON path. Default is '$'.
        """
        if not isinstance(data, dict):
            return
        path_segments = current_json_path.split('.')
        if len(path_segments) >= 2 and path_segments[-1] in ('properties',
                                                             '$defs',
                                                             'definitions'):
            is_skippable = True
        else:
            is_skippable = False
        for key, value in data.items():
            json_path = f"{current_json_path}.{key}"
            if not is_skippable:
                if key not in self.ALLOWED_SCHEMA_FIELDS:
                    raise ValueError(
                        f"Error: Field '{key}' at path '{json_path}' is not allowed."
                    )
            if isinstance(value, dict):
                self._validate_fields(value, json_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_fields(item, f"{json_path}[{i}]")

    def sort_dict_recursively_simple(self, item):
        """
        Recursively sort dictionary.

        Args:
            item (dict | list): Dictionary or list to sort.

        Returns:
            dict | list | any: Sorted dictionary or list.
        """
        if isinstance(item, dict):
            return {
                k: self.sort_dict_recursively_simple(v)
                for k, v in sorted(item.items())
            }
        elif isinstance(item, list):
            return [self.sort_dict_recursively_simple(i) for i in item]
        else:
            return item

    def convert(
        self,
        source_schema: dict,
        resolve_refs: bool = True,
    ) -> dict:
        """
        Convert JSON Schema.

        Args:
            source_schema (dict): Source JSON Schema.
            resolve_refs (bool, optional): Whether to resolve $ref references. Default is True.

        Returns:
            dict: Converted JSON Schema.
        """
        filtered = self._filter(source_schema)
        replaced = self._replace(filtered)
        if resolve_refs:
            processed = self._resolve(replaced)
        else:
            processed = replaced
        simplified = self._simplify(processed)
        cleaned = self._convert_draft_2020_12(simplified)
        final_schema = self._convert_llm_schema(cleaned)
        srtd = self.sort_dict_recursively_simple(final_schema)
        self._validate_fields(srtd)
        res = types.JSONSchema.model_validate(srtd)
        return res.model_dump(
            mode="json",
            exclude_unset=True,
        )
