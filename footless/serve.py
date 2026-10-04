"""`footless serve`: the OpenAI-compatible HTTP front end.

Requests arrive over HTTP and leave as the wire shapes `wire` builds; between
the two, one generation at a time walks the engine's request lifecycle. The
routes are the ones vLLM's OpenAI entrypoints expose for inference — GET
/health, GET /v1/models and its single card, POST /v1/chat/completions and
POST /v1/completions, each with and without streaming — plus the typed
decision routes POST /v1/decisions and POST /v1/systemone, which answer
questions from label scores at the answer position, generate nothing, and
never stream.

Requests are served one at a time, in arrival order: the engine runs one step
at a time and this adds no batching of its own. A disconnected client cancels
its request at the next step boundary.
"""

from __future__ import annotations

import json
import queue
import select
import secrets
import socket
import threading
import time
import uuid
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from . import decisions, wire
from .engine import Engine
from .render import LogprobRow, Output, ToolCalls
from .sdk import LABEL_LOGPROBS, BudgetRefused, ContractError, SamplingSpec

DONE = b"data: [DONE]\n\n"


# ---------------------------------------------------------------- one request


class _Ctx:
    """The step context a request hands the model: cancelled ends it."""

    def __init__(self, cancelled: threading.Event) -> None:
        self._cancelled = cancelled

    def cancelled(self) -> bool:
        return self._cancelled.is_set()


class _Job:
    """One queued request; its events head for the socket that asked."""

    def __init__(self, request, stream: bool) -> None:
        self.request = request
        self.stream = stream
        self.events: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()
        self.ctx = _Ctx(self.cancelled)

    def push(self, kind: str, payload=None) -> None:
        self.events.put((kind, payload))


@dataclass
class _Usage:
    """The request's counters; streamed with every chunk when asked for."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def wire(self) -> dict:
        return wire.usage(
            self.prompt_tokens, self.completion_tokens,
            self.cached_tokens, self.reasoning_tokens,
        )


# ---------------------------------------------------------------- the front end


class FrontEnd:
    """One loaded model behind the wire, one request at a time.

    The request loop runs on one thread only — the thread that opened the
    engine: a model runtime's device context belongs to the thread that
    loaded its weights. `serve` runs the loop where it was called; `start`
    moves it to a thread of its own for runtimes that care less.
    """

    def __init__(self, engine: Engine, model_name: str, root: str = "",
                 api_key: str | None = None) -> None:
        self.engine = engine
        self.model_name = model_name
        self.root = root
        self.api_key = api_key
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def card(self) -> dict:
        """The one model card /v1/models hands out."""
        return wire.model_card(
            self.model_name, int(time.time()), self.root, self.engine.facts.max_context
        )

    def submit(self, job: _Job) -> None:
        self._queue.put(job)

    def start(self) -> None:
        """Run the request loop on a thread of its own."""
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- the one thread that touches the engine ----------------------------

    def run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._run(job)
            except wire.ApiError as exc:
                self._fail(job, exc.body(), exc.code)
            except (BudgetRefused, ContractError) as exc:
                self._log("error", f"{type(exc).__name__}: {exc}")
                self._fail(job, wire.error(str(exc), 500, "InternalServerError"), 500)
            except Exception as exc:  # noqa: BLE001 - one request never kills the server
                self._log("error", f"{type(exc).__name__}: {exc}")
                self._fail(job,
                           wire.error(f"{type(exc).__name__}: {exc}", 500,
                                      "InternalServerError"), 500)

    @staticmethod
    def _fail(job: _Job, body: dict, code: int) -> None:
        # through `ApiError.body()`: a 422 detail error keeps its {"detail":
        # [...]} shape, every other failure the {"error": {...}} envelope
        job.push("error", (body, code))

    def _run(self, job: _Job) -> None:
        request = job.request
        if isinstance(request, (wire.DecisionsRequest, wire.SystemOneRequest)):
            return self._run_decisions(job, request)
        prompts: list[tuple[list[int], str]] = []  # (token ids, echoed text)
        if isinstance(request, wire.ChatRequest):
            tokens = self.engine.chat_tokens(request.messages, request.thinking,
                                             request.level, request.tools)
            prompts.append((self._fit(tokens, request), ""))
        else:
            for prompt in request.prompts:
                tokens = prompt.ids if prompt.ids is not None \
                    else self.engine.runtime.encode(prompt.text or "")
                tokens = self._fit(tokens, request)
                echo = self.engine.decode(tokens) if request.echo else ""
                prompts.append((tokens, echo))
        job.push("started")
        response_id = ("chatcmpl-" if isinstance(request, wire.ChatRequest) else "cmpl-") \
            + str(request.request_id)
        created = int(time.time())
        usage = _Usage(prompt_tokens=sum(len(ids) for ids, _ in prompts))
        spec = self._spec(request.sampling)
        max_tokens = self._max_tokens(request, len(prompts[0][0]))
        choices = []
        index = 0
        for prompt_ids, echo in prompts:
            cached = 0
            for _ in range(request.n):
                output, finish, got, rows, calls = self._one_choice(
                    job, request, prompt_ids, echo, spec, max_tokens, usage,
                    response_id, created, index,
                )
                cached = max(cached, got)  # the choices share the prompt's cache
                choices.append(self._choice(request, output, rows, echo, index,
                                            prompt_ids, finish, calls))
                index += 1
            usage.cached_tokens += cached
        if job.stream:
            tail = self._usage_chunk(request, response_id, created, usage) \
                if request.include_usage or request.continuous_usage else None
            job.push("done", tail)
        else:
            fields = {}
            if request.return_token_ids and isinstance(request, wire.ChatRequest):
                fields["prompt_token_ids"] = prompts[0][0]
            job.push("response", wire.chat_response(
                request.model, response_id, created, choices, usage.wire(), **fields,
            ) if isinstance(request, wire.ChatRequest) else wire.completion_response(
                request.model, response_id, created, choices, usage.wire(),
            ))

    def _one_choice(self, job: _Job, request, prompt_ids: list[int], echo: str,
                    spec: SamplingSpec, max_tokens: int, usage: _Usage,
                    response_id: str, created: int, index: int):
        """One generation; its deltas stream out as they become safe."""
        # the fact describes the thinking mode: a prompt rendered outside it —
        # or a raw completion, which no template rendered — starts outside a block
        open_at_start = (
            self.engine.facts.thinking_open_at_start and request.thinking
            if isinstance(request, wire.ChatRequest) else False
        )
        output = Output(
            decode=self.engine.decode,
            thinking_markers=self.engine.facts.thinking_markers,
            open_at_start=open_at_start,
            stop=request.stop.strings,
            include_stop_str_in_output=request.stop.include_str_in_output,
            with_logprobs=request.sampling.logprobs > 0,
        )
        rows: list[LogprobRow] = []
        calls = None
        if isinstance(request, wire.ChatRequest) and request.read_calls:
            tools = request.tools
            calls = ToolCalls(lambda text, final: self.engine.tool_calls(text, tools, final),
                              None if request.parallel_tool_calls else 1)
        stream = _Stream(job, request, response_id, created, index, usage, echo,
                         prompt_ids, calls) if job.stream else None
        if stream is not None:
            stream.open()
        final_tokens = set(self.engine.facts.stop_tokens) | set(request.stop.token_ids)
        reported = set(request.stop.token_ids)
        base = (usage.completion_tokens, usage.reasoning_tokens)

        def count():
            usage.completion_tokens = base[0] + len(output.tokens)
            usage.reasoning_tokens = base[1] + output.reasoning_tokens

        def on_step(step, timing):
            token = step.token
            if token is None:
                return False
            if token in final_tokens:
                output.stop(token if token in reported else None)
                stop = True
            else:
                stop = output.push(token, step.top_logprobs)
            count()
            pieces, fresh = output.take()
            rows.extend(fresh)
            if stream is not None:
                stream.delta(pieces, fresh)
            return stop

        result = self.engine.generate_tokens(
            prompt_ids, max_tokens, sampling=spec, on_step=on_step, ctx=job.ctx,
        )
        pieces, fresh = output.close()
        rows.extend(fresh)
        finish = "stop" if result.finished else "length"
        if stream is not None:
            stream.close(pieces, fresh, finish, output.stop_reason)
        elif calls is not None:
            calls.push(output.content)  # the whole answer, read once
            calls.close()
        if calls is not None and calls.calls and finish == "stop":
            finish = "tool_calls"  # as in OpenAI and vLLM, for "auto"
        cached = len(prompt_ids) - result.prompt_tokens_fed
        return output, finish, cached, rows, calls

    def _choice(self, request, output: Output, rows: list[LogprobRow], echo: str,
                index: int, prompt_ids: list[int], finish: str,
                calls: ToolCalls | None = None) -> dict:
        logprobs = None
        if rows:
            if isinstance(request, wire.ChatRequest):
                logprobs = wire.chat_logprobs(rows)
            else:
                logprobs = wire.completion_logprobs(rows, len(echo))
        tokens = output.tokens if request.return_token_ids else None
        if isinstance(request, wire.ChatRequest):
            reasoning = output.reasoning if request.include_reasoning else None
            if not reasoning:
                reasoning = None
            content, made = output.content, None
            if calls is not None and calls.calls:
                # with calls, the content around them is trimmed, and may be none
                content = calls.content.strip() or None
                made = [wire.tool_call(wire.tool_call_id(), call["name"], call["arguments"])
                        for call in calls.calls]
            return wire.chat_choice(
                index, wire.chat_message("assistant", content, reasoning, made),
                finish, output.stop_reason, logprobs, tokens,
            )
        prompt_field = prompt_ids if request.return_token_ids else None
        return wire.completion_choice(
            index, echo + output.text, finish, output.stop_reason, logprobs,
            tokens, prompt_field,
        )

    @staticmethod
    def _usage_chunk(request, response_id: str, created: int,
                     usage: _Usage) -> dict:
        if isinstance(request, wire.ChatRequest):
            return wire.chat_chunk(request.model, response_id, created,
                                   usage_info=usage.wire())
        return wire.completion_chunk(request.model, response_id, created,
                                     usage_info=usage.wire())

    # -- the typed decision routes -----------------------------------------

    def _run_decisions(self, job: _Job, request) -> None:
        """A decision request: every question scored, nothing generated.

        One question is one user message rendered with thinking off, scored
        at the answer position through the package's `label_logprobs` verb.
        The capability is gated first (rule 4: an undeclared capability is
        refused with the reason, never quietly served without). Then each
        question walks its checks in request order: the reasoning block must
        be closed at the answer position, the prompt must fit the context
        whole — truncating would cut the answer position off — and the labels
        must resolve to distinct single tokens there, which the verb raises
        for. One question fails the whole request, naming the question. The
        questions are scored in ONE `label_logprobs_batch` call, after every
        question has passed its checks.
        """
        facts = self.engine.facts
        systemone = isinstance(request, wire.SystemOneRequest)
        if LABEL_LOGPROBS not in facts.capabilities:
            # the engine's own gate raises this as a contract fault (a 500);
            # on the wire it is a request this package cannot serve (rule 4)
            raise wire.ApiError(
                "label scoring was asked for, but this package declares no "
                f"{LABEL_LOGPROBS} capability (it declares: "
                f"{facts.capabilities or 'nothing'})")
        if not systemone and request.prompt_format_version is not None \
                and request.prompt_format_version != decisions.PROMPT_FORMAT_VERSION:
            raise wire.ApiError(
                f"prompt_format_version {request.prompt_format_version} is not "
                f"served, this server uses version "
                f"{decisions.PROMPT_FORMAT_VERSION}",
                param="prompt_format_version")
        text = decisions.render_text(request.state if systemone else request.input)
        items = list(request.questions.items()) if systemone else \
            [(question.id, question) for question in request.questions]
        answers: dict = {}
        prompt_tokens = 0
        prepared = []           # (qid, view, labels, prompt ids), request order
        for qid, question in items:
            view = wire.systemone_view(question) if systemone else \
                wire.decision_view(question)
            labels = decisions.default_labels(view)
            if systemone and view.kind == "choice" and len(view.names) > len(labels):
                # past A to Z the System One API labels the options with pairs
                labels = decisions.pair_labels()[: len(view.names)]
            if systemone and view.kind == "score":
                self._check_legend(qid, view)  # before scoring, not after
            content = decisions.render_question(text, view, labels)
            # thinking is forced off: there is no request field for it, and the
            # answer position must sit outside any reasoning block
            prompt_ids = self.engine.chat_tokens(
                [{"role": "user", "content": content}], thinking=False)
            self._check_reasoning_closed(qid, prompt_ids)
            if len(prompt_ids) > facts.max_context:
                # no truncation here: it would cut the answer position off
                raise wire.ApiError(
                    f"question {qid!r}: the prompt has {len(prompt_ids)} tokens, "
                    f"which does not fit the context length of {facts.max_context} "
                    "tokens", param="questions")
            prepared.append((qid, view, labels, prompt_ids))
        if job.ctx.cancelled():
            return  # the client went away: abandon quietly, push nothing
        # every question in one call: a package that can shares the prompts'
        # common head (template and state) instead of re-reading it per
        # question. The head that depends on the input text alone -- where the
        # prompts meet the input rendered with no question -- is the one later
        # requests about the same text repeat: the engine caches it.
        probe = self.engine.chat_tokens([{"role": "user", "content": text}], thinking=False)
        shared = len(probe)
        for _, _, _, prompt_ids in prepared:
            n = 0
            while n < min(shared, len(prompt_ids)) and prompt_ids[n] == probe[n]:
                n += 1
            shared = n
        try:
            scored_all = self.engine.label_logprobs_batch(
                [(prompt_ids, labels) for _, _, labels, prompt_ids in prepared], job.ctx,
                shared=shared)
        except ValueError as exc:  # a label, or a prompt that does not round-trip
            qid = prepared[getattr(exc, "item", 0)][0]
            raise wire.ApiError(f"question {qid!r}: {exc}",
                               param="questions") from None
        if scored_all is None or job.ctx.cancelled():
            return  # cancelled between questions: abandon quietly, push nothing
        for (qid, view, labels, prompt_ids), scored in zip(prepared, scored_all):
            logprobs = [logprob for _, logprob in scored]
            probs = decisions.probabilities(
                logprobs, 1.0 if systemone else request.temperature)
            mass = decisions.label_mass(logprobs)
            if systemone:
                # a non-finite score is a server fault: RuntimeError -> 500
                answers[qid] = wire.systemone_answer(view, probs, mass)
            elif request.return_prompt_token_ids:
                answers[qid] = wire.decisions_answer(
                    view, probs, mass, prompt_ids, [token for token, _ in scored])
            else:
                answers[qid] = wire.decisions_answer(view, probs, mass)
            prompt_tokens += len(prompt_ids)
        if systemone:
            # the served name answered, whatever name the request used
            body = wire.systemone_response(self.model_name, answers, prompt_tokens)
        else:
            body = wire.decisions_response(
                request.model, decisions.PROMPT_FORMAT_VERSION, answers,
                prompt_tokens)
        job.push("response", body)

    def _check_reasoning_closed(self, qid: str, prompt_ids: list[int]) -> None:
        """The thinking-off render must leave the answer outside any block.

        Generic over `Facts.thinking_markers`: scanning the prompt, the LAST
        open marker must come before the last close marker — or the prompt
        leaves a reasoning block open at the answer position. A package that
        declares no markers has no blocks to leave open. The marker ids come
        from the facts; no marker text is assumed here.
        """
        markers = self.engine.facts.thinking_markers
        if not markers:
            return
        opened = max((i for i, token in enumerate(prompt_ids)
                      if token == markers[0]), default=-1)
        closed = max((i for i, token in enumerate(prompt_ids)
                      if token == markers[1]), default=-1)
        if opened > closed:  # an open marker with no close after it is the same
            raise wire.ApiError(
                f"question {qid!r}: the chat template leaves a reasoning block "
                "open at the answer position, so this model is not supported",
                param="questions")

    def _check_legend(self, qid: str, view) -> None:
        """Refuse, before scoring, levels the legend cannot echo back.

        The response encoder refuses NaN and its kin all the same; refusing
        early keeps that fault from costing a prefill.
        """
        try:
            json.dumps(dict(zip(view.names, view.details)), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise wire.ApiError(
                f"question {qid!r}: a level cannot be returned in the legend: "
                f"{exc}", param="questions") from None

    # -- request fitting ---------------------------------------------------

    def _fit(self, prompt: list[int], request) -> list[int]:
        """Truncate as asked, then require the prompt to fit the context."""
        limit = request.truncate_prompt_tokens
        max_context = self.engine.facts.max_context
        if limit is not None:
            keep = max_context if limit == -1 else limit
            if len(prompt) > keep:
                prompt = prompt[-keep:] if request.truncation_side == "left" \
                    else prompt[:keep]
        if len(prompt) > max_context:
            raise wire.ApiError(
                f"Input length ({len(prompt)}) exceeds model's maximum context "
                f"length ({max_context}).",
                param="prompt",
            )
        return prompt

    def _max_tokens(self, request, prompt_len: int) -> int:
        room = max(0, self.engine.facts.max_context - prompt_len)
        return room if request.max_tokens is None else min(request.max_tokens, room)

    def _spec(self, sampling: wire.Sampling) -> SamplingSpec:
        pinned = {
            name: getattr(sampling, name)
            for name in ("temperature", "top_p", "top_k", "min_p", "seed",
                         "presence_penalty", "frequency_penalty", "repetition_penalty")
            if getattr(sampling, name) is not None
        }
        # a capability the package does not declare is never asked for
        pinned["logprobs"] = sampling.logprobs \
            if "logprobs" in self.engine.facts.capabilities else 0
        # what the request does not pin takes the package's own defaults (its
        # generation config), not the engine's: pinning `temperature` must not
        # smuggle in the engine's `top_k` where the package declares its own
        base = self.engine.facts.default_sampling
        return replace(base if base is not None else SamplingSpec(), **pinned)

    def _log(self, level: str, message: str) -> None:
        self.engine.services.log(level, message)


class _Stream:
    """One choice's chunks on the wire, deltas as they become safe."""

    def __init__(self, job: _Job, request, response_id: str, created: int, index: int,
                 usage: _Usage, echo: str, prompt_ids: list[int],
                 calls: ToolCalls | None = None) -> None:
        self.job = job
        self.calls = calls  # the answer's tool calls carved out as they complete
        self.request = request
        self.response_id = response_id
        self.created = created
        self.index = index
        self.usage = usage
        self.echo = echo
        self.prompt_ids = prompt_ids
        self.chat = isinstance(request, wire.ChatRequest)

    def open(self) -> None:
        """The chunk a client sees first: the role, or the echoed prompt."""
        if self.chat:
            chunk = wire.chat_chunk(self.request.model, self.response_id, self.created,
                                    self.index, {"role": "assistant", "content": ""})
        else:
            chunk = wire.completion_chunk(self.request.model, self.response_id,
                                          self.created, self.index, self.echo)
        if self.request.return_token_ids:
            chunk["prompt_token_ids"] = list(self.prompt_ids)
        self._send(chunk)

    def delta(self, pieces, rows: list[LogprobRow], final: bool = False) -> None:
        made = []
        if self.calls is not None:
            pieces, made = self._carve(pieces, final)
        if not pieces and not rows and not made:
            return
        if self.chat:
            fields: dict = {}
            for kind, text in pieces:
                fields[kind] = fields.get(kind, "") + text
            if made:
                fields["tool_calls"] = [
                    wire.tool_call(wire.tool_call_id(), call["name"], call["arguments"], i)
                    for i, call in made]
            chunk = wire.chat_chunk(
                self.request.model, self.response_id, self.created, self.index, fields,
                logprobs=wire.chat_logprobs(rows) if rows else None,
            )
        else:
            chunk = wire.completion_chunk(
                self.request.model, self.response_id, self.created, self.index,
                "".join(text for _, text in pieces),
                logprobs=wire.completion_logprobs(rows, len(self.echo)) if rows else None,
            )
        self._send(chunk)

    def _carve(self, pieces, final: bool):
        """The answer pieces through the call reader: content and new calls."""
        out, made = [], []
        for kind, text in pieces:
            if kind == "content":
                text, fresh = self.calls.push(text)
                made += fresh
            if text:
                out.append((kind, text))
        if final:
            text, fresh = self.calls.close()
            made += fresh
            if text:
                out.append(("content", text))
        return out, made

    def close(self, pieces, rows: list[LogprobRow], finish: str, stop_reason) -> None:
        self.delta(pieces, rows, final=True)
        if self.calls is not None and self.calls.calls and finish == "stop":
            finish = "tool_calls"  # as in OpenAI and vLLM, for "auto"
        if self.chat:
            chunk = wire.chat_chunk(self.request.model, self.response_id, self.created,
                                    self.index, {}, finish_reason=finish,
                                    stop_reason=stop_reason)
        else:
            chunk = wire.completion_chunk(self.request.model, self.response_id,
                                          self.created, self.index, "",
                                          finish_reason=finish, stop_reason=stop_reason)
        self._send(chunk)

    def _send(self, chunk: dict) -> None:
        if self.request.continuous_usage:
            chunk["usage"] = self.usage.wire()
        self.job.push("chunk", chunk)


# ---------------------------------------------------------------- the handler


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "footless"

    # -- verbs -------------------------------------------------------------

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            return self._send(200, b"", "text/plain")
        if not self._authorized(path):
            return self._unauthorized()
        front = self.server.front
        if path == "/v1/models":
            return self._send_json(200, wire.model_list([front.card()]))
        if path.startswith("/v1/models/"):
            name = unquote(path[len("/v1/models/"):])
            if name == front.model_name:
                return self._send_json(200, front.card())
            return self._send_json(404, wire.error(
                f"The model `{name}` does not exist.", 404, "NotFoundError", "model"))
        self._not_found(path)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in ("/v1/chat/completions", "/v1/completions",
                        "/v1/decisions", "/v1/systemone"):
            return self._not_found(path)
        if not self._authorized(path):
            return self._unauthorized()
        if self.headers.get("content-type", "").split(";", 1)[0].strip().lower() \
                != "application/json":
            return self._send_json(400, wire.error(
                "Unsupported Media Type: Only 'application/json' is allowed"))
        body = self._body()
        if body is None:
            return
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("the body must be a JSON object")
        except ValueError as exc:
            return self._send_json(400, wire.error(f"Invalid JSON: {exc}"))
        front = self.server.front
        try:
            if path == "/v1/chat/completions":
                request = wire.parse_chat(payload, {front.model_name},
                                          front.engine.facts.capabilities,
                                          front.engine.facts.thinking_levels)
            elif path == "/v1/completions":
                request = wire.parse_completions(payload, {front.model_name},
                                                 front.engine.facts.capabilities)
            elif path == "/v1/decisions":
                request = wire.parse_decisions(payload, {front.model_name})
            else:
                request = wire.parse_systemone(payload, {front.model_name})
        except wire.ApiError as exc:
            # body() keeps a 422 detail error its {"detail": [...]} shape and
            # every other refusal the {"error": {...}} envelope
            return self._send_json(exc.code, exc.body())
        if request.dropped:
            front._log("info", "ignoring what this engine cannot serve: "
                       + ", ".join(request.dropped))
        if request.request_id is None:
            request.request_id = self.headers.get("X-Request-Id") or uuid.uuid4().hex
        # the decision routes are non-streaming by design
        job = _Job(request, getattr(request, "stream", False))
        front.submit(job)
        if job.stream:
            self._stream(job)
        else:
            self._respond(job)

    # -- answers -----------------------------------------------------------

    def _respond(self, job: _Job) -> None:
        while True:
            event = self._next(job)
            if event is None:  # the client went away
                return
            kind, payload = event
            if kind == "response":
                return self._send_json(200, payload)
            if kind == "error":
                return self._send_json(payload[1], payload[0])

    def _stream(self, job: _Job) -> None:
        event = self._next(job)
        if event is None:
            return
        kind, payload = event
        if kind == "error":
            return self._send_json(payload[1], payload[0])
        # from here on the request is under way: failures ride the stream
        self._send(200, b"", "text/event-stream", chunked=True)
        try:
            if kind != "started":
                self._sse(payload)
            while True:
                event = self._next(job)
                if event is None:
                    return
                kind, payload = event
                if kind == "chunk":
                    self._sse(payload)
                elif kind == "done":
                    if payload is not None:
                        self._sse(payload)
                    self._frame(DONE)
                    return
                elif kind == "error":
                    self._sse(payload[0])
                    self._frame(DONE)
                    return
        except OSError:
            job.cancelled.set()
        finally:
            self._end_chunks()

    def _next(self, job: _Job):
        """The next event, or None once the client has gone away."""
        while True:
            try:
                return job.events.get(timeout=0.2)
            except queue.Empty:
                if self._gone():
                    job.cancelled.set()
                    return None

    # -- SSE over chunked transfer encoding --------------------------------

    def _sse(self, payload: dict) -> None:
        self._frame(b"data: " + json.dumps(payload).encode() + b"\n\n")

    def _frame(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")

    def _end_chunks(self) -> None:
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass

    # -- plumbing ----------------------------------------------------------

    def _body(self) -> bytes | None:
        length = self.headers.get("content-length")
        if length is None:
            self._send_json(400, wire.error("a body with Content-Length is required"))
            return None
        try:
            return self.rfile.read(int(length))
        except ValueError:
            self._send_json(400, wire.error("invalid Content-Length"))
            return None

    def _authorized(self, path: str) -> bool:
        front = getattr(self.server, "front", None)
        if front is None or front.api_key is None or not path.startswith("/v1"):
            return True
        header = self.headers.get("authorization", "")
        scheme, _, param = header.partition(" ")
        return scheme.lower() == "bearer" and secrets.compare_digest(
            param.encode(), front.api_key.encode()
        )

    def _unauthorized(self) -> None:
        self._send_json(401, {"error": "Unauthorized"})

    def _not_found(self, path: str) -> None:
        self._send_json(404, wire.error(f"Unknown path {path}", 404, "NotFoundError"))

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _send(self, code: int, body: bytes, content_type: str,
              chunked: bool = False) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Cache-Control", "no-cache")
            else:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)
        except OSError:
            pass

    def _gone(self) -> bool:
        """Has the peer hung up while we were busy?"""
        try:
            readable = bool(select.select([self.connection], [], [], 0)[0])
            return readable and not self.connection.recv(1, socket.MSG_PEEK)
        except OSError:
            return True

    def log_message(self, fmt: str, *args) -> None:
        front = getattr(self.server, "front", None)
        if front is not None:
            front._log("info", f"{self.address_string()} {fmt % args}")
        else:
            super().log_message(fmt, *args)


# ---------------------------------------------------------------- the server


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8000,
          model_name: str | None = None, root: str | None = None,
          api_key: str | None = None) -> None:
    """Answer the serving API on host:port until interrupted.

    The sockets are served on threads of their own; the requests run here,
    on the caller's thread — the one that opened the engine and holds the
    model's device context.
    """
    front = FrontEnd(engine, model_name or engine.model.name,
                     root if root is not None else "", api_key)
    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    server.front = front
    bound = server.server_address[:2]
    front._log("info", f"serving {front.model_name} on http://{bound[0]}:{bound[1]}/v1")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        front.run()
    except KeyboardInterrupt:
        front._log("info", "shutting down")
    finally:
        server.shutdown()
        server.server_close()
