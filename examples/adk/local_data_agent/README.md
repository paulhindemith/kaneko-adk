# Local Data Agent

This example demonstrates how to create an agent that can interact with a local DuckDB database using the Google ADK framework.

## Quick Start

If you haven't set up your environment variables yet, please refer to the [README.md](../../../README.md#environment-variables) in the project root for instructions on configuring them before proceeding.

Run the following command in this directory (`examples/adk/local_data_agent`) to install dependencies:

```bash
poetry install
```

### ADK Web UI
You can start the ADK web UI in the `examples/adk/` directory with the following command:

```bash
adk web
```

### Streamlit UI
You can start the Streamlit UI with the following command:

```bash
streamlit run main.py