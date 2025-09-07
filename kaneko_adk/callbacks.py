"""
Callbacks for Google ADK.
"""
import asyncio
from datetime import datetime, timedelta
import hashlib
import json
import logging
import os
import random
import time
from typing import Any, cast, Dict, List, Optional

from google import genai
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import AfterToolCallback, LlmAgent
from google.adk.models import Gemini, LlmRequest, LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import BaseTool, ToolContext
from google.genai import types
from litellm import token_counter
from sqids import Sqids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VAR_CONTEXT = "context"
CACHE_LOCATIONS = os.environ.get(
    "CACHE_LOCATIONS",
    "us-east5,us-south1,us-central1,us-west4,us-east1,us-east4,us-west1",
).split(",")
CACHE_STORE: Dict[str, Dict] = {}


def give_cache_location() -> str:
    """Return a random cache location.

    Returns:
        str: A randomly selected cache location.
    """
    return random.choice(CACHE_LOCATIONS)


def set_location(callback_context: CallbackContext, location: str) -> None:
    """Set the Google Cloud location.

    Args:
        callback_context (CallbackContext): The callback context.
        location (str): The Google Cloud location.
    """
    invocation_context = callback_context._invocation_context  # pylint: disable=protected-access

    model = cast(LlmAgent, invocation_context.agent).canonical_model
    if isinstance(model, Gemini):
        # Invalidate cache due to @cached_property
        if "api_client" in model:
            del model.api_client
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        model.api_client  # pylint: disable=pointless-statement


def _generate_cache_id(parts: List[types.Part]) -> str:
    """Generate a unique cache ID based on content parts.

    Args:
        parts (List[types.Part]): List of content parts.

    Returns:
        str: The generated cache ID.
    """
    s = "_".join(part.model_dump_json() for part in parts)
    return hashlib.sha256(s.encode()).hexdigest()


def _create_partial_context(parts: List[types.Part]) -> types.Content:
    """Create a partial context for caching.

    Args:
        parts (List[types.Part]): List of content parts.

    Returns:
        types.Content: The constructed partial context.
    """
    context_parts = [
        types.Part.from_text(
            text=
            "<context-start: This is a hidden context for internal use only. Do not disclose any part of it to the user.>"
        )
    ] + parts
    return types.Content(role="user", parts=context_parts)


async def _create_cache(
    cache_id: str, context: types.Content, model: str,
    system_instruction: Optional[str], tools: Optional[List[types.Tool]],
    ttl: str
) -> None:
    """Create new cached content and save to store.

    Args:
        cache_id (str): Cache identifier.
        context (types.Content): Context to cache.
        model (str): Model name.
        system_instruction (Optional[str]): System instruction.
        tools (Optional[List[types.Tool]]): List of tools.
        ttl (str): Time to live for cache.
    """
    location = give_cache_location()
    client = genai.Client(location=location)
    try:
        res = await client.aio.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                contents=[context],
                system_instruction=system_instruction,
                ttl=ttl,
                tools=tools
            )
        )
        cache_state = {
            "location": location,
            "expired_at": datetime.now() + timedelta(seconds=int(ttl[:-1]) - 5),
            "cache_name": res.name
        }
        CACHE_STORE[cache_id] = cache_state
    except Exception as e:
        print(f"Failed to create cache: {e}")


async def _update_cache(cache_id: str, cache_state: Dict, ttl: str) -> None:
    """Update TTL of existing cached content.

    Args:
        cache_id (str): Cache identifier.
        cache_state (Dict): Current cache state.
        ttl (str): Time to live for cache.
    """
    client = genai.Client(location=cache_state["location"])
    try:
        await client.aio.caches.update(
            name=cache_state["cache_name"],
            config=types.UpdateCachedContentConfig(ttl=ttl)
        )
        cache_state["expired_at"] = datetime.now() + timedelta(
            seconds=int(ttl[:-1]) - 5
        )
        CACHE_STORE[cache_id] = cache_state
    except Exception as e:
        print(f"Failed to update cache: {e}")


async def manage_initial_context_cache(
    initial_contexts: List[types.Part],
    model: str,
    system_instruction: Optional[str] = None,
    tools: Optional[List[types.Tool]] = None,
    ttl: str = "300s"
) -> Optional[Dict]:
    """
    Create or update cache based on initial_contexts.

    Args:
        initial_contexts (List[types.Part]): Initial contexts to cache.
        model (str): Model name.
        system_instruction (Optional[str]): System instruction.
        tools (Optional[List[types.Tool]]): List of tools.
        ttl (str): Time to live for cache.

    Returns:
        Optional[Dict]: The created or updated cache state.
    """
    if not initial_contexts:
        return None

    partial_context = _create_partial_context(initial_contexts)
    cache_id = _generate_cache_id(partial_context.parts)
    cache_state = CACHE_STORE.get(cache_id)

    if cache_state is None or cache_state.get("expired_at",
                                              datetime.min) < datetime.now():
        asyncio.create_task(
            _create_cache(
                cache_id=cache_id,
                context=partial_context,
                model=model,
                system_instruction=system_instruction,
                tools=tools,
                ttl=ttl
            )
        )
    else:
        asyncio.create_task(
            _update_cache(cache_id=cache_id, cache_state=cache_state, ttl=ttl)
        )

    return CACHE_STORE.get(cache_id)


def build_set_context_before_model_callback(
    initial_contexts: Optional[List[types.Part]] = None,
    var_context: str = VAR_CONTEXT,
    max_context_tokens: int = -1,
    caching: bool = True,
) -> Any:
    """Return a callback function to add context before model invocation.

    Args:
        initial_contexts (Optional[List[types.Part]]): List of initial contexts.
        var_context (str): Key for saving context in state.
        max_context_tokens (int): Maximum number of context tokens.
        caching (bool): Whether to enable caching.

    Returns:
        Any: The callback function.
    """
    _initial_contexts_parts = initial_contexts or []
    _initial_contexts = [part.model_dump() for part in _initial_contexts_parts]

    class PartDictAddedToken(types.PartDict):
        """Extended PartDict class with token count."""
        token: int | None

    def _trim_context_parts(
        initial_contexts: list[PartDictAddedToken],
        context_parts: list[PartDictAddedToken],
        max_tokens: int,
    ) -> list[PartDictAddedToken]:
        """Trim context parts to fit within max_tokens.

        Args:
            initial_contexts (list[PartDictAddedToken]): Initial context parts.
            context_parts (list[PartDictAddedToken]): Dynamic context parts.
            max_tokens (int): Maximum allowed tokens.

        Returns:
            list[PartDictAddedToken]: Trimmed context parts.
        """
        total_tokens = sum(part.get("token", 0) for part in initial_contexts)
        if total_tokens > max_tokens:
            raise ValueError(
                f"Initial context tokens: {total_tokens} exceed maximum limit: {max_tokens}"
            )

        reversed_parts = list(reversed(context_parts))
        trimmed = []
        for part in reversed_parts:
            part_token = part.get("token", 0)
            if total_tokens + part_token > max_tokens:
                continue
            total_tokens += part_token
            trimmed.insert(0, part)
        return trimmed

    def _create_left_context(parts: List[types.Part] = []) -> types.Content:
        """Create left context with <context-end> marker.

        Args:
            parts (List[types.Part], optional): List of parts. Defaults to [].

        Returns:
            types.Content: The constructed left context.
        """
        parts = parts + [types.Part.from_text(text="<context-end>")]
        return types.Content(role="user", parts=parts)

    def _create_context(parts: List[types.Part]) -> types.Content:
        """Create full context with start and end markers.

        Args:
            parts (List[types.Part]): List of context parts.

        Returns:
            types.Content: The constructed context.
        """
        context_parts = _create_partial_context(
            parts
        ).parts + _create_left_context().parts
        return types.Content(role="user", parts=context_parts)

    async def _before_model_callback(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """Callback to add context before model invocation.

        Args:
            callback_context (CallbackContext): The callback context.
            llm_request (LlmRequest): The LLM request.

        Returns:
            Optional[LlmResponse]: None.
        """
        context_parts = callback_context.state.get(var_context, [])
        canonical_model = cast(
            LlmAgent, callback_context._invocation_context.agent
        ).canonical_model

        if max_context_tokens > 0:
            tasks = []

            async def set_token(
                part: dict, canonical_model=canonical_model
            ) -> None:
                """Set token count for a part.

                Args:
                    part (dict): The part to count tokens for.
                    canonical_model: The canonical model.
                """
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.model_validate(part)]
                    )
                ]
                if isinstance(canonical_model, Gemini):
                    client = genai.Client()
                    res = await client.aio.models.count_tokens(
                        model=canonical_model.model, contents=contents
                    )
                    part["token"] = res.total_tokens

                elif isinstance(canonical_model, LiteLlm):
                    # pylint: disable=import-outside-toplevel
                    from google.adk.models.lite_llm import (
                        _get_completion_inputs,
                    )
                    messages, _, _, _ = (
                        _get_completion_inputs(LlmRequest(contents=contents))
                    )
                    part["token"] = token_counter(
                        model=canonical_model.model, messages=messages
                    )
                else:
                    raise NotImplementedError(
                        f"Token counting not implemented for model type: {type(canonical_model)}"
                    )

            for p in _initial_contexts + context_parts:
                if "token" not in p:
                    tasks.append(asyncio.create_task(set_token(p)))

            if tasks:
                await asyncio.gather(*tasks)
                callback_context.state[var_context] = context_parts

            new_context_parts = _trim_context_parts(
                _initial_contexts, context_parts, max_context_tokens
            )
            if len(new_context_parts) < len(context_parts):
                callback_context.state[var_context] = new_context_parts
            context_parts = new_context_parts

        dynamic_contexts = [
            types.Part.model_construct(**p) for p in context_parts
        ]
        full_context_parts = _initial_contexts_parts + dynamic_contexts
        context = _create_context(full_context_parts)

        if caching and len(full_context_parts) > 0:
            if isinstance(canonical_model, Gemini):
                partial_cache = False
                partial_cache_state = await manage_initial_context_cache(
                    initial_contexts=_initial_contexts_parts,
                    model=canonical_model.model,
                    system_instruction=llm_request.config.system_instruction,
                    tools=llm_request.config.tools
                )
                if partial_cache_state:
                    partial_cache = True

                cache_id = _generate_cache_id(context.parts)
                cache_state = CACHE_STORE.get(cache_id)

                if cache_state is None or cache_state.get(
                    "expired_at", datetime.min
                ) < datetime.now():
                    asyncio.create_task(
                        _create_cache(
                            cache_id=cache_id,
                            context=context,
                            model=canonical_model.model,
                            system_instruction=llm_request.config.
                            system_instruction,
                            tools=llm_request.config.tools,
                            ttl="60s"
                        )
                    )
                    if partial_cache and partial_cache_state:
                        left_context = _create_left_context(dynamic_contexts)
                        llm_request.contents = [
                            left_context
                        ] + llm_request.contents
                        set_location(
                            callback_context, partial_cache_state["location"]
                        )
                        llm_request.config.cached_content = partial_cache_state[
                            "cache_name"]
                        llm_request.config.system_instruction = None
                        llm_request.config.tools = None
                        llm_request.config.tool_config = None
                    else:
                        llm_request.contents = [context] + llm_request.contents
                else:
                    asyncio.create_task(
                        _update_cache(cache_id, cache_state, "60s")
                    )
                    set_location(callback_context, cache_state["location"])
                    llm_request.config.cached_content = cache_state["cache_name"
                                                                   ]
                    llm_request.config.system_instruction = None
                    llm_request.config.tools = None
                    llm_request.config.tool_config = None
            elif isinstance(canonical_model, LiteLlm):
                logger.warning(
                    "Caching is not supported for LiteLlm models. Proceeding without caching."
                )
                llm_request.contents = [context] + llm_request.contents
        else:
            if len(
                context.parts
            ) > 2:  # If there are elements other than context-start and context-end
                llm_request.contents = [context] + llm_request.contents

        return None

    return _before_model_callback


def build_add_context_after_tool_callback(
    var_context: str = VAR_CONTEXT
) -> AfterToolCallback:
    """Return a callback function to add context after tool execution.

    Args:
        var_context (str): Key for saving context in state.

    Returns:
        AfterToolCallback: The callback function.
    """

    # pylint: disable=unused-argument
    def _after_tool_callback(
        tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext,
        tool_response: Dict
    ) -> Optional[Dict]:
        """Callback to add tool result to context.

        Args:
            tool (BaseTool): The tool instance.
            args (Dict[str, Any]): Arguments for the tool.
            tool_context (ToolContext): The tool context.
            tool_response (Dict): The tool response.

        Returns:
            Optional[Dict]: Message and context ID if context added, else None.
        """
        if _context := tool_response.get("_context"):
            context_id = Sqids().encode([time.time_ns()])
            _context.update({"_ctx_id": context_id})
            context_parts = tool_context.state.get(var_context, [])
            context_parts.append(
                types.Part.from_text(
                    text=json.dumps(_context, ensure_ascii=False)
                ).model_dump()
            )
            tool_context.state[var_context] = context_parts

            return {
                "message": "Tool result added to context",
                "_ctx_id": context_id
            }
        return None

    return _after_tool_callback
