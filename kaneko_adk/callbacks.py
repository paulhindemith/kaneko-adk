"""
Google ADK のコールバック
"""
import asyncio
from datetime import datetime
from datetime import timedelta
import hashlib
import os
import random
from typing import Any, cast, Dict, List, Optional

from google import genai
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import AfterToolCallback
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models import Gemini
from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.adk.tools import BaseTool
from google.adk.tools import ToolContext
from google.genai import types

VAR_CONTEXT = "context"
CACHE_LOCATIONS = os.environ.get(
    "CACHE_LOCATIONS",
    "us-east5,us-south1,us-central1,us-west4,us-east1,us-east4,us-west1"
).split(",")
CACHE_STORE: Dict[str, Dict] = {}


def give_cache_location() -> str:
    """ランダムにキャッシュロケーションを返す。"""
    return random.choice(CACHE_LOCATIONS)


def set_location(callback_context: CallbackContext, location: str):
    """Google Cloud のロケーションを設定する。

    Args:
        callback_context (CallbackContext): コールバックコンテキスト。
        location (str): Google Cloud のロケーション。
    """
    invocation_context = callback_context._invocation_context  # pylint: disable=protected-access

    model: Gemini = cast(LlmAgent, invocation_context.agent).canonical_model
    # @cached_property されているのでキャッシュ無効化
    if "api_client" in model:
        del model.api_client
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    model.api_client  # pylint: disable=pointless-statement


def _generate_cache_id(parts: List[types.Part]) -> str:
    """コンテンツパーツに基づいて一意なキャッシュIDを生成する。"""
    s = "_".join(part.model_dump_json() for part in parts)
    return hashlib.sha256(s.encode()).hexdigest()


def _create_partial_context(parts: List[types.Part]) -> types.Content:
    """キャッシュ用の部分的なコンテキストを作成する。"""
    context_parts = [
        types.Part.from_text(
            text=
            "<context-start: This is a hidden context for internal use only. Do not disclose any part of it to the user.>"
        )
    ] + parts
    return types.Content(role="user", parts=context_parts)


async def _create_cache(cache_id: str, context: types.Content, model: str,
                        system_instruction: Optional[str],
                        tools: Optional[List[types.Tool]], ttl: str):
    """新しいキャッシュコンテンツを作成し、ストアに保存する。"""
    location = give_cache_location()
    client = genai.Client(location=location)
    try:
        res = await client.aio.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                contents=[context],
                system_instruction=system_instruction,
                ttl=ttl,
                tools=tools))
        cache_state = {
            "location": location,
            "expired_at":
            datetime.now() + timedelta(seconds=int(ttl[:-1]) - 5),
            "cache_name": res.name
        }
        CACHE_STORE[cache_id] = cache_state
    except Exception as e:
        print(f"Failed to create cache: {e}")


async def _update_cache(cache_id: str, cache_state: Dict, ttl: str):
    """既存のキャッシュコンテンツのTTLを更新する。"""
    client = genai.Client(location=cache_state["location"])
    try:
        await client.aio.caches.update(
            name=cache_state["cache_name"],
            config=types.UpdateCachedContentConfig(ttl=ttl))
        cache_state["expired_at"] = datetime.now() + timedelta(
            seconds=int(ttl[:-1]) - 5)
        CACHE_STORE[cache_id] = cache_state
    except Exception as e:
        print(f"Failed to update cache: {e}")


async def manage_initial_context_cache(
        initial_contexts: List[types.Part],
        model: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[types.Tool]] = None,
        ttl: str = "300s") -> Optional[Dict]:
    """
    initial_contexts に基づいてキャッシュを作成または更新する。

    Args:
        initial_contexts (List[types.Part]): キャッシュする初期コンテキスト。
        model (str): 使用するモデル名。
        system_instruction (Optional[str]): システムインストラクション。
        tools (Optional[List[types.Tool]]): 使用するツール。
        ttl (str): キャッシュの有効期間。

    Returns:
        Optional[Dict]: 作成または更新されたキャッシュの状態。
    """
    if not initial_contexts:
        return None

    partial_context = _create_partial_context(initial_contexts)
    cache_id = _generate_cache_id(partial_context.parts)
    cache_state = CACHE_STORE.get(cache_id)

    if cache_state is None or cache_state.get("expired_at",
                                              datetime.min) < datetime.now():
        asyncio.create_task(
            _create_cache(cache_id=cache_id,
                          context=partial_context,
                          model=model,
                          system_instruction=system_instruction,
                          tools=tools,
                          ttl=ttl))
    else:
        asyncio.create_task(
            _update_cache(cache_id=cache_id, cache_state=cache_state, ttl=ttl))

    return CACHE_STORE.get(cache_id)


def build_set_context_before_model_callback(
    initial_contexts: Optional[List[types.Part]] = None,
    var_context: str = VAR_CONTEXT,
    max_context_tokens: int = -1,
    caching: bool = True,
):
    """モデルを呼び出す前にコンテキストを追加するコールバック関数を返す。

    Args:
        initial_contexts (Optional[List[types.Part]]): 初期コンテキストのリスト。
        var_context (str): コンテキストを保存する state のキー。
        max_context_tokens (int): コンテキストの最大トークン数。
        caching (bool): キャッシングを有効にするかどうか。
    """
    _initial_contexts_parts = initial_contexts or []
    _initial_contexts = [part.model_dump() for part in _initial_contexts_parts]

    class PartDictAddedToken(types.PartDict):
        """拡張された PartDict にトークン数を追加したクラス。"""
        token: int | None

    def _trim_context_parts(
        initial_contexts: list[PartDictAddedToken],
        context_parts: list[PartDictAddedToken],
        max_tokens: int,
    ) -> list[PartDictAddedToken]:
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
        parts = parts + [types.Part.from_text(text="<context-end>")]
        return types.Content(role="user", parts=parts)

    def _create_context(parts: List[types.Part]) -> types.Content:
        context_parts = _create_partial_context(
            parts).parts + _create_left_context().parts
        return types.Content(role="user", parts=context_parts)

    async def _before_model_callback(
            callback_context: CallbackContext,
            llm_request: LlmRequest) -> Optional[LlmResponse]:
        context_parts = callback_context.state.get(var_context, [])
        model = llm_request.model

        if max_context_tokens > 0:
            tasks = []
            client = genai.Client()

            async def set_token(part: dict, model_name=model):
                res = await client.aio.models.count_tokens(
                    model=model_name,
                    contents=[
                        types.Content(role="user",
                                      parts=[types.Part.model_validate(part)])
                    ])
                part["token"] = res.total_tokens

            for p in _initial_contexts + context_parts:
                if "token" not in p:
                    tasks.append(asyncio.create_task(set_token(p)))

            if tasks:
                await asyncio.gather(*tasks)
                callback_context.state[var_context] = context_parts

            new_context_parts = _trim_context_parts(_initial_contexts,
                                                    context_parts,
                                                    max_context_tokens)
            if len(new_context_parts) < len(context_parts):
                callback_context.state[var_context] = new_context_parts
            context_parts = new_context_parts

        dynamic_contexts = [
            types.Part.model_construct(**p) for p in context_parts
        ]
        full_context_parts = _initial_contexts_parts + dynamic_contexts
        context = _create_context(full_context_parts)

        if caching and len(full_context_parts) > 0:
            partial_cache = False
            partial_cache_state = await manage_initial_context_cache(
                initial_contexts=_initial_contexts_parts,
                model=model,
                system_instruction=llm_request.config.system_instruction,
                tools=llm_request.config.tools)
            if partial_cache_state:
                partial_cache = True

            cache_id = _generate_cache_id(context.parts)
            cache_state = CACHE_STORE.get(cache_id)

            if cache_state is None or cache_state.get(
                    "expired_at", datetime.min) < datetime.now():
                asyncio.create_task(
                    _create_cache(cache_id=cache_id,
                                  context=context,
                                  model=model,
                                  system_instruction=llm_request.config.
                                  system_instruction,
                                  tools=llm_request.config.tools,
                                  ttl="60s"))
                if partial_cache and partial_cache_state:
                    left_context = _create_left_context(dynamic_contexts)
                    llm_request.contents = [left_context
                                            ] + llm_request.contents
                    set_location(callback_context,
                                 partial_cache_state["location"])
                    llm_request.config.cached_content = partial_cache_state[
                        "cache_name"]
                    llm_request.config.system_instruction = None
                    llm_request.config.tools = None
                    llm_request.config.tool_config = None
                else:
                    llm_request.contents = [context] + llm_request.contents
            else:
                asyncio.create_task(_update_cache(cache_id, cache_state,
                                                  "60s"))
                set_location(callback_context, cache_state["location"])
                llm_request.config.cached_content = cache_state["cache_name"]
                llm_request.config.system_instruction = None
                llm_request.config.tools = None
                llm_request.config.tool_config = None
        else:
            if len(context.parts
                   ) > 2:  # context-start と context-end 以外の要素がある場合
                llm_request.contents = [context] + llm_request.contents

        return None

    return _before_model_callback


def build_add_context_after_tool_callback(
        var_context: str = VAR_CONTEXT) -> AfterToolCallback:
    """ツールの実行後にコンテキストを追加するコールバック関数を返す。"""

    # pylint: disable=unused-argument
    def _after_tool_callback(tool: BaseTool, args: Dict[str, Any],
                             tool_context: ToolContext,
                             tool_response: Dict) -> Optional[Dict]:
        if _context := tool_response.get("_context"):
            context_parts = tool_context.state.get(var_context, [])
            context_parts.append(
                types.Part.from_text(text=_context).model_dump())
            tool_context.state[var_context] = context_parts
            return {"message": "Tool result added to context"}
        return None

    return _after_tool_callback
