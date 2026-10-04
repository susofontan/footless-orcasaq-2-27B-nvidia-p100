"""The serving API's wire schema: request parsing and response building.

This is the OpenAI-compatible surface `footless serve` speaks, shaped after the
one in vLLM's OpenAI entrypoints so the same clients work against both. It is
pure data: no engine, no model, no I/O.

A request field is honoured when the engine can execute it exactly. A field
that demands behaviour the engine cannot execute is accepted and ignored, as
vLLM does, and the request carries its name in `dropped` so the server can
report it; nothing fails for asking. What does fail is a request that is
wrong of itself — a mistyped field, a prompt past the context, an unknown
model — exactly the validations vLLM runs too.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from .decisions import QuestionView, check_option_names, choice_confidence, score_confidence

INT64 = (-(2**63), 2**63 - 1)
MAX_STOP_STRINGS = 4  # as in vLLM
MAX_TOP_LOGPROBS = 20  # as in OpenAI
MAX_PROMPTS = 1024  # as in vLLM
MAX_OPTIONS = 26  # /v1/decisions labels options A to Z
MAX_LEVELS = 10  # levels are labeled 0 to 9
MAX_CHOICE_OPTIONS = 255  # the System One API's documented maximum


class ApiError(Exception):
    """A request the server answers without running the model."""

    def __init__(self, message: str, code: int = 400, err_type: str = "BadRequestError",
                 param: str | None = None, detail: list | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.err_type = err_type
        self.param = param
        self.detail = detail

    def body(self) -> dict:
        """The response body: a detail list where one is set, the error envelope else."""
        if self.detail is not None:
            return {"detail": self.detail}
        return error(self.message, self.code, self.err_type, self.param)


def detail_error(entries: list[dict]) -> ApiError:
    """The System One API's documented 422: a FastAPI-style detail list.

    Each entry names where the request went wrong as `loc`, `msg`, and
    `type`; the optional input value is left out, since it can be any
    client value.
    """
    return ApiError("request body failed validation", code=422,
                    err_type="ValidationError", detail=entries)


# ---------------------------------------------------------------- requests


@dataclass
class Sampling:
    """What the request pins; the engine's sampling policy fills the rest."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    logprobs: int = 0  # top log-probabilities per step; 0 = off
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None


@dataclass
class Stop:
    """The request's detokenized and token-id stop rules."""

    strings: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    include_str_in_output: bool = False


@dataclass
class Prompt:
    """One completions prompt: text to encode, or the token ids already."""

    text: str | None = None
    ids: list[int] | None = None


@dataclass
class ChatRequest:
    model: str
    messages: list[dict]
    n: int
    max_tokens: int | None  # None: as much as fits in the context
    sampling: Sampling
    stop: Stop
    stream: bool
    include_usage: bool
    continuous_usage: bool
    thinking: bool
    level: str | None  # one of the package's thinking_levels; None: its default
    include_reasoning: bool
    return_token_ids: bool
    truncate_prompt_tokens: int | None
    truncation_side: str
    request_id: str | None
    dropped: list[str] = field(default_factory=list)  # asked for, not served
    tools: list[dict] | None = None  # rendered into the prompt; None: no tools
    read_calls: bool = False  # the answer's tool calls are read back out of it
    parallel_tool_calls: bool = True  # False: only the first call is returned


@dataclass
class CompletionRequest:
    model: str
    prompts: list[Prompt]
    n: int
    max_tokens: int
    echo: bool
    sampling: Sampling
    stop: Stop
    stream: bool
    include_usage: bool
    continuous_usage: bool
    return_token_ids: bool
    truncate_prompt_tokens: int | None
    truncation_side: str
    request_id: str | None
    dropped: list[str] = field(default_factory=list)  # asked for, not served


# -- typed decision routes ------------------------------------------------


@dataclass
class DecisionQuestion:
    """One /v1/decisions question; the fields its type does not use stay empty."""

    id: str
    type: str  # "choice", "score", or "yes_no"
    question: Any  # the question's own text: a string, an object, or a list
    options: list[tuple[str, Any]] = field(default_factory=list)  # (name, description)
    levels: list = field(default_factory=list)  # score level texts
    yes: Any = None  # yes_no descriptions
    no: Any = None


@dataclass
class DecisionsRequest:
    """A parsed POST /v1/decisions body."""

    input: Any  # rendered before every question: a string, an object, or a list
    questions: list  # DecisionQuestion, in request order
    temperature: float = 1.0  # scales the option probabilities only, not label_mass
    chat_template_kwargs: dict = field(default_factory=dict)  # only empty is served
    prompt_format_version: int | None = None  # a pin the serve layer compares
    return_prompt_token_ids: bool = False
    model: str = "default"
    request_id: str | None = None
    dropped: list[str] = field(default_factory=list)  # asked for, not served


@dataclass
class SystemOneQuestion:
    """One /v1/systemone question; criteria follow the type."""

    id: str  # may be blank here
    type: str  # "noul", "choice", or "score"
    instructions: Any = None  # the question's own text; blank drops the question line
    criteria: dict | list | None = None  # noul {true,false}, choice name map, score levels


@dataclass
class SystemOneRequest:
    """A parsed POST /v1/systemone body."""

    state: Any  # rendered before every question; may be empty
    questions: dict[str, SystemOneQuestion]  # id to question, in request order
    model: str
    request_id: str | None = None
    dropped: list[str] = field(default_factory=list)  # asked for, not served


# -- fields this engine cannot execute: accepted, ignored, reported -------

# name -> the value that means nothing to ask for
_NEUTRAL = {
    "length_penalty": 1.0,
    "watermarking": False,
    "min_tokens": 0,
    "ignore_eos": False,
    "use_beam_search": False,
    "best_of": 1,
    "skip_special_tokens": True,
    "spaces_between_special_tokens": True,
    "return_tokens_as_token_ids": False,
    "return_prompt_text": False,
    "return_token_offsets": False,
    "add_generation_prompt": True,
    "continue_final_message": False,
    "priority": 0,
    "routed_experts_prompt_start": 0,
    "suffix": None,
    "logit_bias": None,
    "prompt_logprobs": None,
    "logprob_token_ids": None,
    "allowed_token_ids": None,
    "bad_words": None,
    "tools": None,
    "tool_choice": None,
    "structured_outputs": None,
    "thinking_token_budget": None,
    "add_special_tokens": None,
    "chat_template": None,
    "chat_template_kwargs": None,
    "documents": None,
    "media_io_kwargs": None,
    "mm_processor_kwargs": None,
    "prompt_embeds": None,
    "cache_salt": None,
    "kv_transfer_params": None,
    "ec_transfer_params": None,
    "repetition_detection": None,
    "stream_interval": None,
    "guided_json": None,
    "guided_regex": None,
    "guided_choice": None,
    "guided_grammar": None,
    "guided_decoding_backend": None,
    "guided_whitespace_pattern": None,
}

# fields that carry no behaviour here; accepted and ignored, as in vLLM
_IGNORED = frozenset({"user", "session_id", "vllm_xargs", "parallel_tool_calls"})


def unknown_fields(body: dict, known) -> list[str]:
    """Body keys this parser does not know; the caller reports and drops them."""
    return sorted(key for key in body if key not in known and key not in _IGNORED)


def _unsupported(body: dict, served=()) -> list[str]:
    """The fields this engine cannot execute, by name.

    They are accepted and ignored — vLLM ignores what it does not serve the
    same way — and the names ride the request out as `dropped`, so the
    server can say what a response will not reflect. A field at its no-op
    value asks for nothing and is not worth a line. `served` names fields
    the caller executes for this request after all.
    """
    out = []
    for name, noop in _NEUTRAL.items():
        if name not in body or name in served:
            continue
        value = body[name]
        if name in ("logit_bias", "bad_words", "tools") and value in ([], {}):
            value = None
        if name == "tool_choice" and value == "none":
            value = None
        if value is None or value == noop:
            continue
        out.append(name)
    value = body.get("response_format")
    if value is not None and not (isinstance(value, dict) and value.get("type") == "text"):
        out.append("response_format")
    return out


# -- typed accessors -------------------------------------------------------


def _int(body: dict, name: str, param: str | None = None,
         lo: int | None = None, hi: int | None = None) -> int:
    value = body[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(f"{param or name} must be an integer", param=param or name)
    if lo is not None and value < lo:
        raise ApiError(f"{param or name} must be >= {lo}", param=param or name)
    if hi is not None and value > hi:
        raise ApiError(f"{param or name} must be <= {hi}", param=param or name)
    return value


def _float(body: dict, name: str, param: str | None = None,
           lo: float | None = None, hi: float | None = None,
           lo_open: bool = False) -> float:
    value = body[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(f"{param or name} must be a number", param=param or name)
    value = float(value)
    if lo is not None and (value <= lo if lo_open else value < lo):
        raise ApiError(
            f"{param or name} must be {'>' if lo_open else '>='} {lo}",
            param=param or name,
        )
    if hi is not None and value > hi:
        raise ApiError(f"{param or name} must be <= {hi}", param=param or name)
    return value


def _bool(body: dict, name: str, param: str | None = None) -> bool:
    value = body[name]
    if not isinstance(value, bool):
        raise ApiError(f"{param or name} must be a boolean", param=param or name)
    return value


def _bool_or(body: dict, name: str, default: bool, param: str | None = None) -> bool:
    """A request field that is absent or null carries no opinion."""
    if body.get(name) is None:
        return default
    return _bool(body, name, param)


def _string(value, param: str) -> str:
    if not isinstance(value, str):
        raise ApiError(f"{param} must be a string", param=param)
    return value


def _sampling(body: dict) -> Sampling:
    sampling = Sampling()
    if "temperature" in body and body["temperature"] is not None:
        sampling.temperature = _float(body, "temperature", lo=0.0)
    if "top_p" in body and body["top_p"] is not None:
        sampling.top_p = _float(body, "top_p", lo=0.0, hi=1.0, lo_open=True)
    if "top_k" in body and body["top_k"] is not None:
        sampling.top_k = _int(body, "top_k", lo=0)
    if "min_p" in body and body["min_p"] is not None:
        sampling.min_p = _float(body, "min_p", lo=0.0, hi=1.0)
    if "seed" in body and body["seed"] is not None:
        sampling.seed = _int(body, "seed", lo=INT64[0], hi=INT64[1])
    for name in ("presence_penalty", "frequency_penalty"):  # vLLM's ranges
        if body.get(name) is not None:
            setattr(sampling, name, _float(body, name, lo=-2.0, hi=2.0))
    if body.get("repetition_penalty") is not None:
        value = _float(body, "repetition_penalty")
        if not 0.0 < value < math.inf:
            raise ApiError("repetition_penalty must be greater than zero",
                           param="repetition_penalty")
        sampling.repetition_penalty = value
    return sampling


def _stop(body: dict) -> Stop:
    stop = Stop()
    raw = body.get("stop")
    if raw is None:
        pass
    elif isinstance(raw, str):
        if not raw:
            raise ApiError("stop strings must not be empty", param="stop")
        stop.strings = [raw]
    elif isinstance(raw, list):
        if len(raw) > MAX_STOP_STRINGS:
            raise ApiError(
                f"stop must hold at most {MAX_STOP_STRINGS} strings", param="stop"
            )
        for item in raw:
            if not isinstance(item, str) or not item:
                raise ApiError("stop strings must be non-empty strings", param="stop")
        stop.strings = list(raw)
    else:
        raise ApiError("stop must be a string or a list of strings", param="stop")
    ids = body.get("stop_token_ids")
    if ids is not None:
        if not isinstance(ids, list):
            raise ApiError("stop_token_ids must be a list of integers", param="stop_token_ids")
        for item in ids:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ApiError("stop_token_ids must be non-negative integers",
                               param="stop_token_ids")
        stop.token_ids = list(ids)
    if "include_stop_str_in_output" in body:
        stop.include_str_in_output = _bool_or(body, "include_stop_str_in_output", False)
    return stop


def _streaming(body: dict, dropped: list[str]) -> tuple[bool, bool, bool]:
    stream = _bool_or(body, "stream", False)
    include_usage = False
    continuous = False
    options = body.get("stream_options")
    if options is not None and not isinstance(options, dict):
        raise ApiError("stream_options must be an object", param="stream_options")
    if isinstance(options, dict):
        if not stream:
            raise ApiError(
                "stream_options is only allowed when stream is true",
                param="stream_options",
            )
        for key in options:
            if key not in ("include_usage", "continuous_usage_stats"):
                dropped.append(f"stream_options.{key}")
        if "include_usage" in options:
            include_usage = _bool_or(options, "include_usage", False,
                                     "stream_options.include_usage")
        if "continuous_usage_stats" in options:
            continuous = include_usage and _bool_or(options, "continuous_usage_stats",
                                                    False,
                                                    "stream_options.continuous_usage_stats")
    return stream, include_usage, continuous


def _model(body: dict, names) -> str:
    model = body.get("model")
    if model is None or model == "":
        return next(iter(sorted(names)))
    model = _string(model, "model")
    if model not in names:
        raise ApiError(
            f"The model `{model}` does not exist.",
            code=404, err_type="NotFoundError", param="model",
        )
    return model


def _count(body: dict, name: str, dropped: list[str]) -> int:
    """A top-logprobs count: 0 or more, where -1 means "as many as there are"."""
    value = _int(body, name, lo=-1)
    if value == -1:
        dropped.append(f"{name} (all is not a count; {MAX_TOP_LOGPROBS} used)")
        return MAX_TOP_LOGPROBS
    return value


def _logprobs(body: dict, capabilities, chat: bool, dropped: list[str]) -> Sampling:
    sampling = _sampling(body)
    if chat:
        if "logprobs" in body and body["logprobs"] is not None:
            if _bool(body, "logprobs"):
                sampling.logprobs = 1
        if body.get("top_logprobs") is not None:
            top = _count(body, "top_logprobs", dropped)
            if top > 0 and not sampling.logprobs:
                raise ApiError(
                    "top_logprobs requires logprobs to be true", param="top_logprobs"
                )
            sampling.logprobs = max(sampling.logprobs, top)
    elif body.get("logprobs") is not None:
        sampling.logprobs = _count(body, "logprobs", dropped)
    if sampling.logprobs and "logprobs" not in capabilities:
        dropped.append("logprobs (the package declares no logprobs capability)")
    return sampling


def _truncation(body: dict) -> tuple[int | None, str]:
    limit = None
    if body.get("truncate_prompt_tokens") is not None:
        limit = _int(body, "truncate_prompt_tokens", lo=-1, hi=INT64[1])
    side = body.get("truncation_side")
    if side is not None:
        side = _string(side, "truncation_side")
        if side not in ("left", "right"):
            raise ApiError("truncation_side must be 'left' or 'right'",
                           param="truncation_side")
    return limit, side or "right"


def _drop_seed(sampling: Sampling, choices: int, dropped: list[str]) -> None:
    """A seed identifies one per-request draw stream (AGENTS.md).

    Several choices would replay the same draws; the seed is dropped and the
    choices get fresh streams instead.
    """
    if sampling.seed is not None and choices > 1:
        dropped.append("seed")
        sampling.seed = None


def _tail(body: dict, dropped: list[str]) -> dict:
    stream, include_usage, continuous = _streaming(body, dropped)
    return {
        "n": _int(body, "n", lo=1) if body.get("n") is not None else 1,
        "stream": stream,
        "include_usage": include_usage,
        "continuous_usage": continuous,
        "return_token_ids": _bool_or(body, "return_token_ids", False),
        "stop": _stop(body),
        "request_id": body.get("request_id"),
    }


# -- function tools --------------------------------------------------------


def _tool(raw, index: int) -> dict:
    """One function tool, as vLLM hands it to the template.

    Only the fields vLLM's schema knows survive, in its order; `strict` and
    `defer_loading` only when set, the latter copied into the function too.
    """
    param = f"tools[{index}]"
    if not isinstance(raw, dict):
        raise ApiError(f"{param} must be an object", param=param)
    if raw.get("type", "function") != "function":
        raise ApiError(f"{param}.type must be 'function'", param=f"{param}.type")
    function = raw.get("function")
    if not isinstance(function, dict):
        raise ApiError(f"{param}.function must be an object", param=f"{param}.function")
    name = function.get("name")
    if not isinstance(name, str):
        raise ApiError(f"{param}.function.name must be a string",
                       param=f"{param}.function.name")
    description = function.get("description")
    if description is not None and not isinstance(description, str):
        raise ApiError(f"{param}.function.description must be a string",
                       param=f"{param}.function.description")
    parameters = function.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise ApiError(f"{param}.function.parameters must be an object",
                       param=f"{param}.function.parameters")
    out = {"name": name, "description": description, "parameters": parameters}
    for key in ("strict", "defer_loading"):
        for where, value in ((f"{param}.function.{key}", function.get(key)),
                             (f"{param}.{key}", raw.get(key))):
            if value is not None and not isinstance(value, bool):
                raise ApiError(f"{where} must be a boolean", param=where)
    if function.get("strict") is not None:
        out["strict"] = function["strict"]
    deferred = function.get("defer_loading")
    if deferred is None:
        deferred = raw.get("defer_loading")
    if deferred is not None:
        out["defer_loading"] = deferred
    tool = {"type": "function", "function": out}
    if raw.get("defer_loading") is not None:
        tool["defer_loading"] = raw["defer_loading"]
    return tool


def _tools(body: dict, dropped: list[str]) -> tuple[list[dict] | None, bool, bool]:
    """`tools`, `tool_choice` and `parallel_tool_calls`, checked as vLLM does.

    Returns the tools to render, whether the answer's calls are read back,
    and whether more than one call may be returned. "none" renders the tools
    and reads nothing back, as in vLLM. "required" and a named function need
    constrained decoding, which this engine does not have: they are served as
    "auto" and reported.
    """
    raw = body.get("tools")
    if raw == []:
        raise ApiError("`tools` must not be an empty array. Either provide at least "
                       "one tool or omit the field entirely.", param="tools")
    if raw is not None and not isinstance(raw, list):
        raise ApiError("tools must be a list", param="tools")
    tools = [_tool(item, index) for index, item in enumerate(raw)] if raw else None
    choice = body.get("tool_choice")
    if choice is None:
        choice = "auto" if tools else None
    elif choice != "none":
        if tools is None:
            raise ApiError("When using `tool_choice`, `tools` must be set.",
                           param="tool_choice")
        if isinstance(choice, dict):
            usage = 'Correct usage: `{"type": "function", "function": {"name": "my_function"}}`'
            function = choice.get("function")
            if not isinstance(function, dict):
                raise ApiError(f"Invalid value for `function`: `{function}` in "
                               f"`tool_choice`! {usage}", param="tool_choice.function")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ApiError(f"Invalid `name` in `function`: `{name}` in "
                               f"`tool_choice`! {usage}", param="tool_choice.function.name")
            if all(tool["function"]["name"] != name for tool in tools):
                raise ApiError("The tool specified in `tool_choice` does not match any "
                               "of the specified `tools`", param="tool_choice")
            dropped.append("tool_choice (a named function needs constrained decoding; "
                           "served as auto)")
        elif choice == "required":
            dropped.append("tool_choice (required needs constrained decoding; "
                           "served as auto)")
        elif choice != "auto":
            raise ApiError(f"Invalid value for `tool_choice`: {choice}! Only named tools, "
                           '"none", "auto" or "required" are supported.',
                           param="tool_choice")
    parallel = _bool_or(body, "parallel_tool_calls", True)
    return tools, tools is not None and choice != "none", parallel


# -- messages --------------------------------------------------------------


def _content(raw, param: str, dropped: list[str]) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        out = []
        for part in raw:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                out.append(_string(part.get("text"), f"{param} text"))
            else:
                kind = part.get("type", "unknown") if isinstance(part, dict) else "unknown"
                dropped.append(f"{param} {kind}")  # the packages are text-only
        return "".join(out)
    raise ApiError(f"{param} must be a string or a list of content parts", param=param)


def _messages(body: dict, dropped: list[str]) -> list[dict]:
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise ApiError("messages must be a non-empty list", param="messages")
    out = []
    for index, item in enumerate(raw):
        param = f"messages[{index}]"
        if not isinstance(item, dict):
            raise ApiError(f"{param} must be an object", param=param)
        role = _string(item.get("role"), f"{param}.role")
        role = "system" if role == "developer" else role
        message = {"role": role,
                   "content": _content(item.get("content"), f"{param}.content", dropped)}
        for key, value in item.items():
            if key not in ("role", "content"):
                message[key] = value  # the contract passes other keys to the template
        if role == "assistant":
            _assistant_turn(message, param)
        out.append(message)
    return out


def _assistant_turn(message: dict, param: str) -> None:
    """An earlier answer as templates take it, the way vLLM hands it over.

    Its `reasoning` is also `reasoning_content`; its tool calls' `arguments`,
    JSON text on the wire, become the mapping templates iterate — a text that
    is not a JSON object becomes an empty one, as in vLLM, so one malformed
    call in the history does not fail every later turn.
    """
    if message.get("reasoning") is not None:
        message["reasoning_content"] = message["reasoning"]
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return
    if not calls:
        del message["tool_calls"]
        return
    for index, call in enumerate(calls):
        where = f"{param}.tool_calls[{index}]"
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(call, dict) or call.get("type", "function") != "function" \
                or not isinstance(function, dict):
            raise ApiError("chat completions only support assistant tool_calls of type "
                           "'function'.", param=where)
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = None
        if not isinstance(arguments, dict):
            arguments = {}
        function["arguments"] = arguments


# -- prompts ---------------------------------------------------------------


def _prompts(body: dict) -> list[Prompt]:
    raw = body.get("prompt")
    if raw is None:
        raise ApiError("prompt is required", param="prompt")
    if isinstance(raw, str):
        return [Prompt(text=raw)]
    if not isinstance(raw, list) or not raw:
        raise ApiError("prompt must be a non-empty string or list", param="prompt")

    def ids(item) -> list[int] | None:
        if not isinstance(item, list):
            return None
        for value in item:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ApiError("prompt token ids must be non-negative integers",
                               param="prompt")
        return list(item)

    if all(isinstance(item, int) and not isinstance(item, bool) for item in raw):
        return [Prompt(ids=ids(raw))]
    if all(isinstance(item, list) for item in raw):
        if len(raw) > MAX_PROMPTS:
            raise ApiError(f"prompt holds more than {MAX_PROMPTS} prompts",
                           param="prompt")
        return [Prompt(ids=ids(item)) for item in raw]
    if all(isinstance(item, str) for item in raw):
        if len(raw) > MAX_PROMPTS:
            raise ApiError(f"prompt holds more than {MAX_PROMPTS} prompts",
                           param="prompt")
        return [Prompt(text=item) for item in raw]
    raise ApiError("prompt must be a string, a list of strings, or token ids",
                   param="prompt")


# -- entry points ----------------------------------------------------------


def _reasoning(body: dict, levels, dropped: list[str]) -> tuple[bool, str | None]:
    """`reasoning_effort` as vLLM serves it: the level goes to the template.

    `"none"` turns thinking off; any other value is the template's own level
    vocabulary, which the package declares. A value outside it is the
    template's rejection — a 400 — and a package that declares no levels
    never sees the kwarg at all, which is what vLLM does with a template
    that does not declare it.
    """
    effort = body.get("reasoning_effort")
    if effort is None:
        return True, None
    if effort == "none":
        return False, None
    if not levels:
        dropped.append("reasoning_effort (the package declares no levels)")
        return True, None
    if effort not in levels:
        raise ApiError(
            f"reasoning_effort {effort!r} is not one of this package's levels: "
            f"{', '.join(levels)} (or 'none' to turn thinking off)",
            param="reasoning_effort",
        )
    return True, effort


def parse_chat(body: dict, names, capabilities, levels=()) -> ChatRequest:
    # function tools are served by a package that declares `tool_calls`, and
    # dropped (reported) by any other
    served = ("tools", "tool_choice") if "tool_calls" in capabilities else ()
    dropped = _unsupported(body, served)
    tools, read_calls, parallel = _tools(body, dropped) if served else (None, False, True)
    thinking, level = _reasoning(body, levels, dropped)
    sampling = _logprobs(body, set(capabilities), chat=True, dropped=dropped)
    tail = _tail(body, dropped)
    _drop_seed(sampling, tail["n"], dropped)
    if body.get("max_completion_tokens") is not None:
        max_tokens = _int(body, "max_completion_tokens", lo=0)
    elif body.get("max_tokens") is not None:
        max_tokens = _int(body, "max_tokens", lo=0)
    else:
        max_tokens = None
    truncation = _truncation(body)
    return ChatRequest(
        model=_model(body, names),
        messages=_messages(body, dropped),
        n=tail["n"],
        max_tokens=max_tokens,
        sampling=sampling,
        stop=tail["stop"],
        stream=tail["stream"],
        include_usage=tail["include_usage"],
        continuous_usage=tail["continuous_usage"],
        thinking=thinking,
        level=level,
        include_reasoning=_bool_or(body, "include_reasoning", True),
        return_token_ids=tail["return_token_ids"],
        truncate_prompt_tokens=truncation[0],
        truncation_side=truncation[1],
        request_id=tail["request_id"],
        dropped=dropped,
        tools=tools,
        read_calls=read_calls,
        parallel_tool_calls=parallel,
    )


def parse_completions(body: dict, names, capabilities) -> CompletionRequest:
    dropped = _unsupported(body)
    sampling = _logprobs(body, set(capabilities), chat=False, dropped=dropped)
    if body.get("max_tokens") is None:
        max_tokens = 16  # as in OpenAI and vLLM
    else:
        max_tokens = _int(body, "max_tokens", lo=0)
    truncation = _truncation(body)
    tail = _tail(body, dropped)
    prompts = _prompts(body)
    _drop_seed(sampling, tail["n"] * len(prompts), dropped)
    return CompletionRequest(
        model=_model(body, names),
        prompts=prompts,
        n=tail["n"],
        max_tokens=max_tokens,
        echo=_bool_or(body, "echo", False),
        sampling=sampling,
        stop=tail["stop"],
        stream=tail["stream"],
        include_usage=tail["include_usage"],
        continuous_usage=tail["continuous_usage"],
        return_token_ids=tail["return_token_ids"],
        truncate_prompt_tokens=truncation[0],
        truncation_side=truncation[1],
        request_id=tail["request_id"],
        dropped=dropped,
    )


# -- typed decision routes: parsing ---------------------------------------


_DECISIONS_FIELDS = frozenset({
    "input", "questions", "temperature", "chat_template_kwargs",
    "prompt_format_version", "return_prompt_token_ids", "model",
})

_SYSTEMONE_FIELDS = frozenset({
    "state", "model", "questions", "chat_template_kwargs",
    "temperature", "prompt_format_version", "return_prompt_token_ids",
})

# /v1/decisions fields, refused by name on /v1/systemone: they would change
# the answers there whether honored or ignored.
_SYSTEMONE_DECISIONS_ONLY = (
    "temperature", "prompt_format_version", "return_prompt_token_ids",
)


def _blank(value) -> bool:
    """Blank as in decisions.render_question: a stripped-empty string, or an empty object or list."""
    return not (value.strip() if isinstance(value, str) else value)


def _text(value, param: str) -> Any:
    """A prompt text: a string, an object, or a list."""
    if not isinstance(value, (str, dict, list)):
        raise ApiError(f"{param} must be a string, an object, or a list", param=param)
    return value


def _nonblank(value, param: str) -> Any:
    """A prompt text the request cannot do without."""
    _text(value, param)
    if _blank(value):
        raise ApiError(f"{param} must not be blank", param=param)
    return value


def _text_or_none(value, param: str) -> Any:
    """An optional prompt text: a question's descriptions may come later or never."""
    if value is None:
        return None
    return _text(value, param)


def _invalid(loc: list, msg: str, kind: str = "value_error") -> ApiError:
    """One entry in the System One API's 422 detail list."""
    return detail_error([{"loc": loc, "msg": msg, "type": kind}])


def _detail_text(value, loc: list) -> Any:
    """A prompt text in the 422 vocabulary; blank is a value here, only not a text."""
    if not isinstance(value, (str, dict, list)):
        raise _invalid(loc, "Input should be a valid string, dictionary or list")
    return value


def _detail_nonblank(value, loc: list) -> Any:
    """A prompt text the request cannot do without, in the 422 vocabulary."""
    _detail_text(value, loc)
    if _blank(value):
        raise _invalid(loc, "Value error, must not be blank")
    return value


def _model_adapter(model: str, route: str) -> None:
    """Refuse the `base:adapter` form: this surface serves no LoRA."""
    if ":" in model:
        adapter = model.split(":", 1)[1]
        raise ApiError(
            f"model names the LoRA adapter {adapter!r}, which {route} does not "
            "support", param="model")


def _decision_options(raw: dict, param: str, qid: str) -> list[tuple[str, Any]]:
    options = raw.get("options")
    path = f"{param}.options"
    if options is None:
        raise ApiError(f"{path} is required", param=path)
    if not isinstance(options, list):
        raise ApiError(f"{path} must be a list", param=path)
    if len(options) < 2:
        raise ApiError(f"{path} must hold at least 2 options", param=path)
    if len(options) > MAX_OPTIONS:
        raise ApiError(f"{path} must hold at most {MAX_OPTIONS} options", param=path)
    out = []
    for index, option in enumerate(options):
        item = f"{path}[{index}]"
        if not isinstance(option, dict):
            raise ApiError(f"{item} must be an object", param=item)
        for key in option:
            if key not in ("name", "description"):
                raise ApiError(f"unknown field {key!r}", param=f"{item}.{key}")
        name_param = f"{item}.name"
        if option.get("name") is None:
            raise ApiError(f"{name_param} is required", param=name_param)
        name = _string(option["name"], name_param)
        description = _text_or_none(option.get("description"), f"{item}.description")
        out.append((name, description))
    try:
        check_option_names(name for name, _ in out)
    except ValueError as e:
        raise ApiError(f"question {qid!r}: {e}", param=path) from None
    return out


def _decision_levels(raw: dict, param: str) -> list:
    levels = raw.get("levels")
    path = f"{param}.levels"
    if levels is None:
        raise ApiError(f"{path} is required", param=path)
    if not isinstance(levels, list):
        raise ApiError(f"{path} must be a list", param=path)
    if len(levels) < 2:
        raise ApiError(f"{path} must hold at least 2 levels", param=path)
    if len(levels) > MAX_LEVELS:
        raise ApiError(f"{path} must hold at most {MAX_LEVELS} levels", param=path)
    return [_nonblank(level, f"{path}[{index}]")
            for index, level in enumerate(levels)]


def _decision_question(raw, index: int) -> DecisionQuestion:
    param = f"questions[{index}]"
    if not isinstance(raw, dict):
        raise ApiError(f"{param} must be an object", param=param)
    id_param = f"{param}.id"
    if raw.get("id") is None:
        raise ApiError(f"{id_param} is required", param=id_param)
    qid = _string(raw["id"], id_param)
    if not qid.strip():
        raise ApiError(f"{id_param} must not be blank", param=id_param)
    type_param = f"{param}.type"
    if raw.get("type") is None:
        raise ApiError(f"{type_param} is required", param=type_param)
    kind = _string(raw["type"], type_param)
    own = {"choice": {"options"}, "score": {"levels"},
           "yes_no": {"yes", "no"}}.get(kind)
    if own is None:
        raise ApiError(f"{type_param} must be 'choice', 'score', or 'yes_no'",
                       param=type_param)
    for key in raw:
        if key not in {"id", "type", "question"} | own:
            raise ApiError(f"unknown field {key!r}", param=f"{param}.{key}")
    text_param = f"{param}.question"
    if raw.get("question") is None:
        raise ApiError(f"{text_param} is required", param=text_param)
    question = _nonblank(raw["question"], text_param)
    if kind == "choice":
        return DecisionQuestion(id=qid, type=kind, question=question,
                                options=_decision_options(raw, param, qid))
    if kind == "score":
        return DecisionQuestion(id=qid, type=kind, question=question,
                                levels=_decision_levels(raw, param))
    return DecisionQuestion(id=qid, type=kind, question=question,
                            yes=_text_or_none(raw.get("yes"), f"{param}.yes"),
                            no=_text_or_none(raw.get("no"), f"{param}.no"))


def parse_decisions(body: dict, names: set[str]) -> DecisionsRequest:
    """Parse a POST /v1/decisions body strictly: an unknown field is refused.

    A request may name any model — clients carry their own aliases — so
    `names`, the served model names, keeps the parsers' shared call shape;
    the name is echoed back and only the `base:adapter` form is refused.
    """
    for key in body:
        if key not in _DECISIONS_FIELDS:
            raise ApiError(f"unknown field {key!r}", param=key)
    if body.get("input") is None:
        raise ApiError("input is required", param="input")
    text = _nonblank(body["input"], "input")
    raw_questions = body.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ApiError("questions must be a non-empty list", param="questions")
    questions, seen = [], set()
    for index, raw in enumerate(raw_questions):
        question = _decision_question(raw, index)
        if question.id in seen:
            raise ApiError(f"question id {question.id!r} repeats another question",
                           param=f"questions[{index}].id")
        seen.add(question.id)
        questions.append(question)
    temperature = 1.0
    if body.get("temperature") is not None:
        temperature = _float(body, "temperature", lo=0.0, lo_open=True)
        if not math.isfinite(temperature):
            raise ApiError("temperature must be finite", param="temperature")
    kwargs = body.get("chat_template_kwargs")
    if kwargs is None:
        kwargs = {}
    elif not isinstance(kwargs, dict):
        raise ApiError("chat_template_kwargs must be an object",
                       param="chat_template_kwargs")
    if kwargs:
        raise ApiError("chat_template_kwargs must be empty",
                       param="chat_template_kwargs")
    version = None
    if body.get("prompt_format_version") is not None:
        version = _int(body, "prompt_format_version")
    model = body.get("model")
    model = "default" if model is None else _string(model, "model")
    _model_adapter(model, "/v1/decisions")
    return DecisionsRequest(
        input=text,
        questions=questions,
        temperature=temperature,
        chat_template_kwargs=kwargs,
        prompt_format_version=version,
        return_prompt_token_ids=_bool_or(body, "return_prompt_token_ids", False),
        model=model,
    )


def _systemone_noul(qid: str, raw: dict, loc: list, instructions) -> SystemOneQuestion:
    criteria = raw.get("criteria")
    true = false = None
    if criteria is not None:
        path = loc + ["criteria"]
        if not isinstance(criteria, dict):
            raise _invalid(path, "Input should be a valid dictionary", "dict_type")
        for key in criteria:
            if key not in ("true", "false"):
                raise _invalid(path + [key], "Extra inputs are not permitted",
                               "extra_forbidden")
        if criteria.get("true") is not None:
            true = _detail_text(criteria["true"], path + ["true"])
        if criteria.get("false") is not None:
            false = _detail_text(criteria["false"], path + ["false"])
    if all(_blank(value) for value in (instructions, true, false)):
        raise _invalid(loc,
                       "Value error, a noul question needs instructions or a true "
                       "or false description to decide on")
    return SystemOneQuestion(id=qid, type="noul", instructions=instructions,
                             criteria=criteria)


def _systemone_question(qid: str, raw) -> SystemOneQuestion:
    loc = ["body", "questions", qid]
    if not isinstance(raw, dict):
        raise _invalid(loc, "Input should be a valid dictionary", "dict_type")
    if raw.get("type") is None:
        raise _invalid(loc + ["type"], "Field required", "missing")
    kind = raw["type"]
    if kind not in ("noul", "choice", "score"):
        raise _invalid(loc + ["type"], "Input should be 'noul', 'choice' or 'score'",
                       "literal_error")
    for key in raw:
        if key not in ("type", "instructions", "criteria"):
            raise _invalid(loc + [key], "Extra inputs are not permitted",
                           "extra_forbidden")
    instructions = raw.get("instructions")
    if instructions is not None:
        instructions = _detail_text(instructions, loc + ["instructions"])
    criteria, path = raw.get("criteria"), loc + ["criteria"]
    if kind == "noul":
        return _systemone_noul(qid, raw, loc, instructions)
    if kind == "choice":
        if criteria is None:
            raise _invalid(path, "Field required", "missing")
        if not isinstance(criteria, dict):
            raise _invalid(path, "Input should be a valid dictionary", "dict_type")
        if not criteria:
            raise _invalid(path, "Dictionary should have at least 1 item", "too_short")
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise _invalid(path,
                           f"Dictionary should have at most {MAX_CHOICE_OPTIONS} items",
                           "too_long")
        parsed = {name: _detail_text(description, path + [name])
                  if description is not None else None
                  for name, description in criteria.items()}
        try:
            check_option_names(parsed)
        except ValueError as e:
            raise _invalid(path, f"Value error, {e}") from None
        return SystemOneQuestion(id=qid, type=kind, instructions=instructions,
                                 criteria=parsed)
    if criteria is None:
        raise _invalid(path, "Field required", "missing")
    if not isinstance(criteria, list):
        raise _invalid(path, "Input should be a valid list", "list_type")
    if not criteria:
        raise _invalid(path, "List should have at least 1 item", "too_short")
    if len(criteria) > MAX_LEVELS:
        raise _invalid(path, f"List should have at most {MAX_LEVELS} items", "too_long")
    levels = [_detail_nonblank(level, path + [index])
              for index, level in enumerate(criteria)]
    return SystemOneQuestion(id=qid, type=kind, instructions=instructions,
                             criteria=levels)


def parse_systemone(body: dict, names: set[str]) -> SystemOneRequest:
    """Parse a POST /v1/systemone body: permissive on top, strict inside.

    Unknown top-level fields are accepted and ignored, their names in
    `dropped`, as the published schema allows; inside a question nothing is
    ignored, since a misspelled key would answer a different question. A
    request may name any model — the System One clients send their own
    aliases — so `names`, the served model names, keeps the parsers' shared
    call shape; only the `base:adapter` form is refused.
    """
    for name in _SYSTEMONE_DECISIONS_ONLY:
        if body.get(name) is not None:
            raise _invalid(["body", name],
                           f"Value error, {name} is not part of this API, "
                           "use /v1/decisions for it")
    dropped = unknown_fields(body, _SYSTEMONE_FIELDS)
    if body.get("state") is None:
        raise _invalid(["body", "state"], "Field required", "missing")
    state = _detail_text(body["state"], ["body", "state"])
    if body.get("model") is None:
        raise _invalid(["body", "model"], "Field required", "missing")
    model = body["model"]
    if not isinstance(model, str):
        raise _invalid(["body", "model"], "Input should be a valid string",
                       "string_type")
    _model_adapter(model, "/v1/systemone")
    questions = body.get("questions")
    loc = ["body", "questions"]
    if questions is None:
        raise _invalid(loc, "Field required", "missing")
    if not isinstance(questions, dict):
        raise _invalid(loc, "Input should be a valid dictionary", "dict_type")
    if not questions:
        raise _invalid(loc, "Dictionary should have at least 1 item", "too_short")
    parsed = {qid: _systemone_question(qid, raw) for qid, raw in questions.items()}
    kwargs = body.get("chat_template_kwargs")
    if kwargs is not None:
        path = ["body", "chat_template_kwargs"]
        if not isinstance(kwargs, dict):
            raise _invalid(path, "Input should be a valid dictionary", "dict_type")
        if kwargs:
            raise _invalid(path, "Value error, chat_template_kwargs must be empty")
    return SystemOneRequest(state=state, questions=parsed, model=model, dropped=dropped)


def decision_view(question: DecisionQuestion) -> QuestionView:
    """The question as the renderer and scorer see it, candidates in order."""
    if question.type == "choice":
        return QuestionView(
            kind="choice",
            question=question.question,
            names=[name for name, _ in question.options],
            details=[description for _, description in question.options],
        )
    if question.type == "score":
        return QuestionView(
            kind="score",
            question=question.question,
            names=[str(level) for level in range(len(question.levels))],
            details=list(question.levels),
        )
    return QuestionView(
        kind="yes_no",
        question=question.question,
        names=["yes", "no"],
        details=[question.yes, question.no],
    )


def systemone_view(question: SystemOneQuestion) -> QuestionView:
    """The question as the renderer and scorer see it; noul is a yes or no."""
    if question.type == "choice":
        return QuestionView(
            kind="choice",
            question=question.instructions,
            names=list(question.criteria),
            details=list(question.criteria.values()),
        )
    if question.type == "score":
        return QuestionView(
            kind="score",
            question=question.instructions,
            names=[str(level) for level in range(len(question.criteria))],
            details=list(question.criteria),
        )
    criteria = question.criteria or {}
    return QuestionView(
        kind="yes_no",
        question=question.instructions,
        names=["yes", "no"],
        details=[criteria.get("true"), criteria.get("false")],
    )


# ---------------------------------------------------------------- responses


def error(message: str, code: int = 400, err_type: str = "BadRequestError",
          param: str | None = None) -> dict:
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}


def usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int,
          reasoning_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def model_card(name: str, created: int, root: str, max_model_len: int) -> dict:
    return {
        "id": name,
        "object": "model",
        "created": created,
        "owned_by": "footless",
        "root": root,
        "max_model_len": max_model_len,
        "permission": [{
            "id": f"modelperm-{name}",
            "object": "model_permission",
            "created": created,
            "allow_create_engine": False,
            "allow_sampling": True,
            "allow_logprobs": True,
            "allow_search_indices": False,
            "allow_view": True,
            "allow_fine_tuning": False,
            "organization": "*",
            "group": None,
            "is_blocking": False,
        }],
    }


def model_list(cards: list[dict]) -> dict:
    return {"object": "list", "data": cards}


def _bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def chat_logprobs(rows) -> dict:
    content = []
    for row in rows:
        content.append({
            "token": row.text,
            "logprob": -9999.0 if row.logprob is None else row.logprob,
            "bytes": _bytes(row.text),
            "top_logprobs": [
                {"token": text, "logprob": logprob, "bytes": _bytes(text)}
                for _, text, logprob in row.top
            ],
        })
    return {"content": content}


def completion_logprobs(rows, text_offset: int = 0) -> dict:
    return {
        "text_offset": [text_offset + row.start for row in rows],
        "token_logprobs": [row.logprob for row in rows],
        "tokens": [row.text for row in rows],
        "top_logprobs": [
            {text: logprob for _, text, logprob in row.top} or None for row in rows
        ],
    }


def chat_message(role: str, content: str | None, reasoning: str | None,
                 tool_calls: list[dict] | None = None) -> dict:
    message = {"role": role}
    if reasoning is not None:
        message["reasoning"] = reasoning
    message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def tool_call_id() -> str:
    """A fresh call id, in vLLM's default form."""
    return f"chatcmpl-tool-{uuid.uuid4().hex}"


def tool_call(call_id: str, name: str, arguments: str, index: int | None = None) -> dict:
    """One call in a message, or with `index` in a streamed delta."""
    out = {"id": call_id, "type": "function"}
    if index is not None:
        out["index"] = index
    out["function"] = {"name": name, "arguments": arguments}
    return out


def chat_response(model: str, response_id: str, created: int, choices: list[dict],
                  usage_info: dict, prompt_token_ids=None) -> dict:
    response = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage_info,
    }
    if prompt_token_ids is not None:
        response["prompt_token_ids"] = prompt_token_ids
    return response


def chat_choice(index: int, message: dict, finish_reason: str,
                stop_reason=None, logprobs=None, token_ids=None) -> dict:
    choice = {"index": index, "message": message, "finish_reason": finish_reason}
    if stop_reason is not None:
        choice["stop_reason"] = stop_reason
    if logprobs is not None:
        choice["logprobs"] = logprobs
    if token_ids is not None:
        choice["token_ids"] = token_ids
    return choice


def chat_chunk(model: str, response_id: str, created: int, index: int | None = None,
               delta: dict | None = None, finish_reason: str | None = None,
               stop_reason=None, logprobs=None, usage_info=None) -> dict:
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    if index is None:
        chunk["choices"] = []
    else:
        choice = {"index": index, "delta": delta or {}}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
            if stop_reason is not None:
                choice["stop_reason"] = stop_reason
        if logprobs is not None:
            choice["logprobs"] = logprobs
        chunk["choices"] = [choice]
    if usage_info is not None:
        chunk["usage"] = usage_info
    return chunk


def completion_response(model: str, response_id: str, created: int, choices: list[dict],
                        usage_info: dict) -> dict:
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage_info,
    }


def completion_choice(index: int, text: str, finish_reason: str, stop_reason=None,
                      logprobs=None, token_ids=None, prompt_token_ids=None) -> dict:
    choice = {"index": index, "text": text, "finish_reason": finish_reason}
    if stop_reason is not None:
        choice["stop_reason"] = stop_reason
    if logprobs is not None:
        choice["logprobs"] = logprobs
    if token_ids is not None:
        choice["token_ids"] = token_ids
    if prompt_token_ids is not None:
        choice["prompt_token_ids"] = prompt_token_ids
    return choice


def completion_chunk(model: str, response_id: str, created: int,
                     index: int | None = None, text: str = "",
                     finish_reason: str | None = None, stop_reason=None, logprobs=None,
                     usage_info=None) -> dict:
    """One stream chunk; `index` None is the trailing usage chunk (no choices)."""
    chunk = {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model,
    }
    if index is None:
        chunk["choices"] = []
    else:
        choice = {"index": index, "text": text}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
            if stop_reason is not None:
                choice["stop_reason"] = stop_reason
        if logprobs is not None:
            choice["logprobs"] = logprobs
        chunk["choices"] = [choice]
    if usage_info is not None:
        chunk["usage"] = usage_info
    return chunk


# -- typed decision routes: answers ---------------------------------------


def decisions_answer(view: QuestionView, probs: list[float], mass: float,
                     prompt_token_ids=None, label_token_ids=None) -> dict:
    """One /v1/decisions answer: probabilities by name and the type's own value."""
    answer = {
        "type": view.kind,
        "probabilities": dict(zip(view.names, probs)),
        "label_mass": mass,
    }
    if view.kind == "choice":
        answer["choice"] = view.names[probs.index(max(probs))]
    elif view.kind == "score":
        answer["score"] = math.fsum(i * p for i, p in enumerate(probs))
    if prompt_token_ids is not None:
        answer["prompt_token_ids"] = prompt_token_ids
    if label_token_ids is not None:
        answer["label_token_ids"] = label_token_ids
    return answer


def systemone_answer(view: QuestionView, probs: list[float], mass: float) -> dict:
    """One /v1/systemone answer in the published shape.

    Probabilities are reported as scored — only confidence renormalizes
    them. Non-finite values are a server fault, never a client error.
    """
    if not all(math.isfinite(value) for value in [*probs, mass]):
        raise RuntimeError("the question scored non-finite values")
    if view.kind == "yes_no":
        return {"type": "noul", "noul": probs[0], "x_label_mass": mass}
    probabilities = dict(zip(view.names, probs))
    if view.kind == "choice":
        return {
            "type": "choice",
            "choice": view.names[probs.index(max(probs))],
            "confidence": choice_confidence(_normalized(probs)),
            "probabilities": probabilities,
            "x_label_mass": mass,
        }
    return {
        "type": "score",
        "score": math.fsum(i * p for i, p in enumerate(probs)),
        "confidence": score_confidence(_normalized(probs)),
        "legend": dict(zip(view.names, view.details)),
        "probabilities": probabilities,
        "x_label_mass": mass,
    }


def decisions_response(model: str, prompt_format_version: int, answers: dict,
                       prompt_tokens: int) -> dict:
    """The /v1/decisions response; usage counts every question's prompt."""
    return {
        "object": "decisions",
        "model": model,
        "prompt_format_version": prompt_format_version,
        "answers": answers,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        },
    }


def systemone_response(model: str, answers: dict, input_tokens: int) -> dict:
    """The /v1/systemone response, under the served model's name."""
    return {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }


def _normalized(probs: list[float]) -> list[float]:
    """Probabilities renormalized over the labels; confidence measures only these."""
    total = math.fsum(probs)
    if total <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]
