# Vega-Lite Schema

The Vega-Lite schema is extremely large and complex, making it difficult for generative AI to handle directly. Therefore, we provide a script that converts the schema into a more manageable format.

Currently, we are using the Vega-Lite v6.2.0 schema. You can obtain the schema from [here](https://vega.github.io/schema/vega-lite/v6.2.0.json).

### Usage

Please download the schema in advance and save it as `scripts/vega-lite/vega-lite.v6.2.0.json`.

```bash
python scripts/vega-lite/generate_code.py \
--source scripts/vega-lite/vega-lite.v6.2.0.json \
--output_dir kaneko_adk/tools/show_chart
```

This will generate a file named _auto_generated_gemini_schema.json in the specified directory.

### Development

You can change the settings in `scripts/vega-lite/config.py`.

- **name**: Specifies the name of the schema.
- **includes**: Specifies fields to include after conversion using JSONPath format.
- **excludes**: Specifies fields to exclude after conversion using JSONPath format.
- **unresolving_definitions**: Specifies `definitions` that should not be resolved. For example, if you specify `$.definitions.foo`, `$ref: '#/definitions/foo'` will remain unresolved.
- **replace**: Specifies fields to replace after conversion using JSONPath format and the value to replace with.
- **remove_fields**: Removes the specified fields from all JSON fields.

**Note:** All JSONPath expressions are evaluated against the schema specified as the source. Make sure your expressions match the structure of the downloaded schema file.
