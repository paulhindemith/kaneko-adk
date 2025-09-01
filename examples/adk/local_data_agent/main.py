"""
Streamlit Demo for Local Data Agent
"""
import asyncio
import copy
import random
import time
from typing import AsyncGenerator, AsyncIterable, Iterable, List
import uuid

from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types
import pandas as pd
from sqids import Sqids
import sqlparse
import streamlit as st

# pylint: disable=import-error
from agent import root_agent, MAX_ROWS  # isort: skip

APP_NAME_FOR_ADK = "local_data_agent"
ADK_SESSION_KEY = "adk_session_id"
USER_ID = "user_id"
INITIAL_STATE = {}


@st.cache_resource
def initialize_adk():
    """
    Initializes the Google ADK Runner and manages the ADK session.
    Uses Streamlit's cache_resource to ensure this runs only once per app load.
    """
    return InMemoryRunner(agent=root_agent, app_name=APP_NAME_FOR_ADK)


async def run_adk_async(runner: InMemoryRunner, user_id: str, session_id: str,
                        user_message_text: str) -> AsyncGenerator[Event, None]:
    """
    Asynchronously runs a single turn of the ADK agent conversation.
    """
    content = types.Content(role='user',
                            parts=[types.Part(text=user_message_text)])

    agent_event_generator = runner.run_async(user_id=user_id,
                                             session_id=session_id,
                                             new_message=content)
    async for event in agent_event_generator:
        yield event


def author_to_st_name(author: str) -> str:
    """
    Maps the author name to the Streamlit message sender name.
    """

    if author == "user":
        return "user"
    return "ai"


def show_chart(call: types.FunctionCall):
    """
    Displays a Vega-Lite chart.
    """

    df = pd.read_csv(call.args["data"]["url"])
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    spec = copy.deepcopy(call.args)
    del spec["data"]
    st.vega_lite_chart(df, spec)


def dialog(parts: List[types.Part]):
    """
    Displays the execution details for each part.
    Args:
        parts (List[types.Part]): The parts to display.
    """

    parts_iter = iter(parts)
    part = next(parts_iter, None)
    while part:
        if call := part.function_call:
            if call.name == "execute_sql":
                with st.expander(":green-badge[execute_sql]", expanded=True):
                    formatted_sql = sqlparse.format(call.args["query"],
                                                    reindent=True,
                                                    keyword_case='upper')
                    st.code(formatted_sql, language="sql", wrap_lines=True)
                    response_part: types.Part = None
                    for p in parts:
                        if p.function_response and part.function_call.id == p.function_response.id:
                            response_part = p
                            break
                    if response_part is None:
                        st.warning("No response found.",
                                   icon=":material/computer_cancel:")
                    elif "error" in response_part.function_response.response:
                        st.warning(
                            "LLM made a mistake during content generation.",
                            icon=":material/computer_cancel:")
                    else:
                        df = pd.read_csv(
                            response_part.function_response.response["path"])
                        st.dataframe(df)
                        st.caption(
                            f":material/info: maximum {MAX_ROWS} rows can be retrieved."
                        )

            elif part.function_call.name == "show_chart":
                with st.expander(":blue-badge[show_chart]", expanded=True):
                    response_part: types.Part = None
                    for p in parts:
                        if p.function_response and part.function_call.id == p.function_response.id:
                            response_part = p
                            break
                    if response_part is None:
                        st.warning("No response found.",
                                   icon=":material/computer_cancel:")
                    elif "error" in response_part.function_response.response:
                        st.warning(
                            "LLM made a mistake during content generation.",
                            icon=":material/computer_cancel:")
                    else:
                        show_chart(part.function_call)

            else:
                st.code(part, wrap_lines=True, language="python")
        part = next(parts_iter, None)


async def show_events(event_stream: AsyncIterable[Event]):
    """
    Displays the event information in the Streamlit app.
    Args:
        event_stream (Iterable[Event] | AsyncIterable[Event]): The events to display.
    """

    def button(empty, disabled=False):

        with empty.popover(":small[分析ツール]",
                           disabled=disabled,
                           icon=":material/analytics:"):
            dialog(parts)

    with st.spinner("Waiting for response..."):
        event: Event = await anext(event_stream, None)
    while event:
        parts: List[types.Part] = []
        current_author = event.author
        if event.content and event.content.parts:
            with st.container(gap=None):
                with st.container(horizontal_alignment="right"):
                    empty = st.empty()
                with st.chat_message(author_to_st_name(event.author)):
                    while event and event.author == current_author:
                        for part in event.content.parts:
                            if part.function_call:
                                button(empty, disabled=True)
                                parts.append(part)
                            if part.function_response:
                                parts.append(part)
                                if part.function_response.name == "show_chart" and "error" not in part.function_response.response:
                                    for p in parts:
                                        if p.function_call and p.function_call.id == part.function_response.id:
                                            show_chart(p.function_call)
                                            break
                            elif part.text:
                                st.markdown(part.text)
                        with st.spinner("Waiting for response..."):
                            event: Event = await anext(event_stream, None)
                if len(parts) > 0:
                    button(empty)
        elif event.actions and (event.actions.state_delta
                                or event.actions.artifact_delta):
            print("Type: State/Artifact Update")
        else:
            print(f"Unknown event type: {event}")


async def main():
    """
    Main entry point for the Streamlit app.
    """

    runner = initialize_adk()

    if "user_id" not in st.session_state:
        sqids = Sqids()
        st.session_state["user_id"] = sqids.encode(
            [time.time_ns(), random.randint(1, 1_000_000)])
    if ADK_SESSION_KEY not in st.session_state:
        st.session_state[ADK_SESSION_KEY] = uuid.uuid4().hex
        await runner.session_service.create_session(
            app_name=APP_NAME_FOR_ADK,
            user_id=st.session_state["user_id"],
            session_id=st.session_state[ADK_SESSION_KEY],
        )
    with st.sidebar:
        if st.button("Clear"):
            del st.session_state[ADK_SESSION_KEY]
            st.rerun()
    session = await runner.session_service.get_session(
        app_name=APP_NAME_FOR_ADK,
        user_id=st.session_state["user_id"],
        session_id=st.session_state[ADK_SESSION_KEY])

    async def async_generator(sync_iterable: Iterable):
        for item in sync_iterable:
            yield item

    await show_events(async_generator(session.events))

    if prompt := st.chat_input("Ask for Local Data Agent"):
        event = Event(
            author='user',
            content=types.Content(role="user",
                                  parts=[types.Part.from_text(text=prompt)]),
        )
        await show_events(async_generator([event]))
        with st.spinner("Waiting for response..."):
            response = run_adk_async(
                runner=runner,
                user_id=st.session_state["user_id"],
                session_id=st.session_state[ADK_SESSION_KEY],
                user_message_text=prompt)
        await show_events(response)


if __name__ == "__main__":
    asyncio.run(main())
