"""The footless command line."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .engine import DEFAULT_BUDGET, DEFAULT_MAX_TOKENS, LOOP_GUARD_MODES, Engine
from .sdk import ContractError, SamplingSpec


def _add_speculative(parser) -> None:
    """`--speculative`, with `--mtp` as the alias people actually type.

    The canonical name is the neutral one: the engine must not carry an
    architecture's name (rule 1). `--mtp` is the same switch, because a drafter
    is the one implementation of speculative decoding this project has, and
    asking people to type the long name is asking for the wrong one.
    """
    parser.add_argument(
        "--speculative", "--mtp", dest="speculative", action="store_true",
        help="draft ahead and verify, if the package declares the capability",
    )


def _add_sampling(parser) -> None:
    """The sampling knobs a request may pin; the rest take the package's own
    defaults (its generation config), then the engine's own."""
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", dest="top_p", type=float, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--min-p", dest="min_p", type=float, default=None)
    parser.add_argument("--repetition-penalty", dest="repetition_penalty",
                        type=float, default=None,
                        help="HF scale over tokens seen in prompt+output "
                             "(vLLM semantics; > 1 discourages repeats)")
    parser.add_argument("--presence-penalty", dest="presence_penalty",
                        type=float, default=None,
                        help="minus this once per token seen in the output "
                             "(OpenAI semantics)")
    parser.add_argument("--frequency-penalty", dest="frequency_penalty",
                        type=float, default=None,
                        help="minus this x times-seen per output token "
                             "(OpenAI semantics)")
    parser.add_argument("--seed", type=int, default=None,
                        help="pin the request's draw stream (replays per turn)")
    parser.add_argument("--loop-guard", dest="loop_guard", nargs="?",
                        const="full", default="exact", choices=LOOP_GUARD_MODES,
                        help="stop a repeating generation and cut the output "
                             "to one cycle: 'exact' (default) on verbatim "
                             "copies only, 'full' (the bare flag) also on "
                             "near-copies -- which a rule over raw token ids "
                             "cannot tell from a table of iterations -- and "
                             "'off' never")
    parser.add_argument("--no-loop-guard", dest="loop_guard", action="store_const",
                        const="off", help="same as --loop-guard off")


def _sampling(args, defaults: SamplingSpec | None) -> SamplingSpec | None:
    """The spec the flags pin, over the package's own defaults.

    None means nothing is pinned: the request runs under
    `Facts.default_sampling` unchanged. What is pinned goes over THAT, not
    over the engine's own spec — pinning `--temperature` must not smuggle in
    the engine's `top_k` where the package declares its own.
    """
    names = ("temperature", "top_p", "top_k", "min_p", "seed",
             "repetition_penalty", "presence_penalty", "frequency_penalty")
    pinned = {n: getattr(args, n, None) for n in names
              if getattr(args, n, None) is not None}
    if not pinned:
        return None
    return replace(defaults if defaults is not None else SamplingSpec(), **pinned)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="footless")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="run inference on a model package")
    gen.add_argument("--model", required=True, help="model package directory")
    gen.add_argument("--backend", default=None, help="hardware folder to load")
    gen.add_argument("--prompt", action="append", required=True, help="prompt text")
    gen.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    gen.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="bytes")
    _add_speculative(gen)
    _add_sampling(gen)
    chat = sub.add_parser("chat", help="streaming chat on a model package")
    chat.add_argument("--model", required=True, help="model package directory")
    chat.add_argument("--backend", default=None, help="hardware folder to load")
    chat.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    chat.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="bytes")
    chat.add_argument("--no-thinking", action="store_true", help="skip thinking blocks")
    chat.add_argument("--reasoning", default=None, metavar="LEVEL",
                      help="reasoning level the package offers (its thinking_levels)")
    chat.add_argument("--system", default=None, metavar="TEXT",
                      help="system message for the conversation (e.g. for code: "
                           "'escribe cada función exactamente una vez' -- "
                           "measured to cut duplicated definitions 5-8x)")
    _add_speculative(chat)
    _add_sampling(chat)
    bench = sub.add_parser("bench", help="speed benchmark: prefill and decode rates")
    bench.add_argument("--model", required=True, help="model package directory")
    bench.add_argument("--backend", default=None, help="hardware folder to load")
    bench.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="bytes")
    bench.add_argument("--prompt-file", default=None, help="raw prompt text file")
    bench.add_argument("--chat-prompt-file", default=None, help="one chat user turn")
    bench.add_argument("--system", default="You are a helpful assistant.")
    bench.add_argument("--ctx-start", type=int, default=2048, help="first frontier")
    bench.add_argument("--ctx-max", type=int, default=32768, help="last frontier")
    bench.add_argument("--step-incr", type=int, default=2048, help="frontier step")
    bench.add_argument("--step-mul", type=float, default=1.0, help="frontier multiplier")
    bench.add_argument("-n", "--gen-tokens", "--tokens", dest="gen_tokens",
                       type=int, default=128, help="greedy tokens per frontier; 0 = prefill only")
    bench.add_argument("--csv", default=None, help="CSV file (default stdout)")
    bench.add_argument("--show-output", action="store_true", help="decoded text per frontier")
    serve = sub.add_parser("serve", help="OpenAI-compatible serving API")
    serve.add_argument("--model", required=True, help="model package directory")
    serve.add_argument("--backend", default=None, help="hardware folder to load")
    serve.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="bytes")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--api-key", default=None,
                       help="require Authorization: Bearer <key> on /v1 routes")
    serve.add_argument("--served-model-name", default=None,
                       help="model name the API answers to (default: the manifest's)")
    _add_speculative(serve)
    args = parser.parse_args(argv)

    try:
        engine = Engine.open(args.model, backend=args.backend, budget=args.budget)
        # Before anything runs, so a package that cannot speculate says so here
        # and not half a generation later. `bench` takes no such flag: it is a
        # measurement. On `serve` the flag is a process-wide default (2026-10-01):
        # every request runs under it.
        engine.speculative(getattr(args, "speculative", False))
        try:
            if args.command == "generate":
                for prompt in args.prompt:
                    result = engine.generate(
                        prompt, max_tokens=args.max_tokens,
                        sampling=_sampling(args, engine.facts.default_sampling),
                        loop_guard=args.loop_guard,
                    )
                    print(result.text)
                    if result.looped:
                        print("footless: repetition loop, output cut to one "
                              "cycle", file=sys.stderr)
            elif args.command == "chat":
                _chat_loop(engine, args)
            elif args.command == "serve":
                _serve(engine, args)
            else:
                _bench_run(engine, args)
        finally:
            engine.close()
    except ContractError as exc:
        print(f"footless: {exc}", file=sys.stderr)
        return 2
    return 0


def _serve(engine: Engine, args) -> None:
    from .serve import serve

    serve(engine, host=args.host, port=args.port,
          model_name=args.served_model_name, root=args.model,
          api_key=args.api_key)


def _bench_run(engine: Engine, args) -> None:
    from pathlib import Path

    from .bench import frontiers, run

    if bool(args.prompt_file) == bool(args.chat_prompt_file):
        raise ContractError("exactly one of --prompt-file or --chat-prompt-file is required")
    text = Path(args.prompt_file or args.chat_prompt_file).read_text()
    if args.prompt_file:
        tokens = engine.runtime.encode(text)
    else:
        tokens = engine.chat_tokens(
            [
                {"role": "system", "content": args.system},
                {"role": "user", "content": text},
            ],
            thinking=False,
        )
    sweep = frontiers(args.ctx_start, args.ctx_max, args.step_incr, args.step_mul)
    out = open(args.csv, "w") if args.csv else sys.stdout
    try:
        run(engine, tokens, sweep, args.gen_tokens, args.show_output, out)
    finally:
        if args.csv:
            out.close()


def _turn_block(render: str, anchor: str, question: str) -> str | None:
    """The text this turn adds after the previous answer's own text, or None.

    `render` is the whole conversation re-rendered with the new question in
    place; the block is everything the template puts AFTER the previous
    answer. Anchored on that text because nothing longer is stable -- the
    template closes a LAST assistant turn differently from a mid one, so a
    length cut against the previous render eats the head of the block (it ate
    "implementa di" once, and the model answered a greeting).

    Occurrences are walked BACKWARD and accepted only when the remainder still
    carries the question. The answer's text routinely occurs EARLIER in the
    render -- a short answer sits inside the reasoning that drafted it, or
    inside a question that asked for exactly that text -- and taking the first
    match splices a duplicate of the thinking (or of the question) into the
    stream: the model then sees its own previous turn twice and the next
    answer is confabulation. When no occurrence qualifies, return None and let
    the caller fall back to a full re-render: a slower turn, never a broken
    one.
    """
    if not anchor:
        return None
    pos = render.rfind(anchor)
    while pos >= 0:
        block = render[pos + len(anchor):]
        if question in block:
            return block
        pos = render.rfind(anchor, 0, pos)
    return None


def _chat_loop(engine: Engine, args) -> None:
    from .display import GRAY, RESET, StreamDisplay

    thinking = not args.no_thinking
    level = _reasoning_level(engine, args.reasoning, thinking)
    messages: list[dict] = []
    if getattr(args, "system", None):
        messages.append({"role": "system", "content": args.system})
    # the conversation as the RUN saw it: request ids plus generated ids.
    # Turns extend this stream instead of re-rendering the history from text:
    # a re-render re-tokenizes the generated turns at the template seams, the
    # prefix cache then matches only PARTIALLY, and a partial hit truncates the
    # state -- which zeroes the GDN recurrence and re-runs the whole kept
    # prefix inside the next prefill (truncate_state's replay). Extended in
    # place the hit is exact and the next prefill is just the new block.
    stream: list[int] = []
    prev_answer = ""
    prev_reasoning = ""
    while True:
        try:
            line = input("chat> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip():
            continue
        if line.strip() in ("/quit", "/exit"):
            break
        messages.append({"role": "user", "content": line})
        if stream:
            # The block this turn adds is everything the render puts AFTER the
            # previous answer's own text: the turn transition, the user
            # wrapper, the question, the generation prompt. The anchor is the
            # last thing the template puts inside the assistant block -- the
            # answer's text, or (when a turn burned its whole budget in
            # thinking and the answer is EMPTY) the reasoning text, which is
            # why one lost anchor used to cost a 453-token replay.
            r_full = engine.decode(engine.chat_tokens(messages, thinking, level))
            anchor = prev_answer or prev_reasoning
            block = _turn_block(r_full, anchor, line)
            _dbg = bool(__import__("os").environ.get("FOOTLESS_DEBUG"))
            if _dbg:
                import sys as _s
                print(f"[dbg cli] stream={len(stream)} prev_answer={len(prev_answer)} "
                      f"block={'ok' if block is not None else 'lost'} "
                      f"r_full={len(r_full)}", file=_s.stderr)
            if block is None:
                request = engine.chat_tokens(messages, thinking, level)
            else:
                request = stream + engine.runtime.encode(block)
            if _dbg:
                import sys as _s
                print(f"[dbg cli] request={len(request)}", file=_s.stderr)
        else:
            request = engine.chat_tokens(messages, thinking, level)
        display = StreamDisplay(
            sys.stdout, engine.decode, engine.facts.thinking_markers,
            # the fact describes the thinking mode: with thinking off the
            # template closes the block, so the answer starts outside one
            open_at_start=engine.facts.thinking_open_at_start and thinking,
        )
        result = engine.generate_tokens(
            request,
            args.max_tokens,
            on_token=display.token,
            sampling=_sampling(args, engine.facts.default_sampling),
            loop_guard=args.loop_guard,
        )
        display.close(result.timing)
        if result.looped:
            sys.stdout.write(f"{GRAY}[repetition loop: output cut to one "
                             f"cycle]{RESET}\n")
        stream = request + list(result.absorbed_tokens or result.tokens)
        assistant = _assistant_message(engine, result, thinking)
        prev_answer = assistant["content"]
        prev_reasoning = assistant["reasoning_content"]
        messages.append(assistant)


def _assistant_message(engine: Engine, result, thinking: bool) -> dict:
    """The turn as it goes into the conversation history: thinking and answer
    apart.

    `result.text` is the whole stream, reasoning and answer alike. Stored as
    `content`, the template's next render puts an empty thinking block before
    it and lets the reasoning chatter stand where the answer belongs -- and
    the model, seeing its turns shaped like that, answers with more chatter.
    The template wants the two apart: `reasoning_content` inside the thinking
    block, `content` outside it. `render.Output` is exactly that split (the
    markers delimit spans and never render), which is what `serve` already
    hands its clients.
    """
    from .render import Output

    output = Output(
        engine.decode, engine.facts.thinking_markers,
        # the fact describes the thinking mode: with thinking off the prompt
        # closed the block, so the answer starts outside one
        open_at_start=engine.facts.thinking_open_at_start and thinking,
    )
    for token in result.tokens:
        output.push(token)
    return {
        "role": "assistant",
        "reasoning_content": output.reasoning.strip(),
        "content": output.content.strip(),
    }


def _reasoning_level(engine: Engine, value: str | None, thinking: bool) -> str | None:
    """The `--reasoning` level, checked against what the package offers."""
    if value is None:
        return None
    if not thinking:
        raise ContractError("--reasoning and --no-thinking ask for opposite things")
    levels = engine.facts.thinking_levels
    if value not in levels:
        offered = ", ".join(levels) if levels else "none: this package has one mode"
        raise ContractError(
            f"reasoning level {value!r} is not offered by this package ({offered})"
        )
    return value


if __name__ == "__main__":
    sys.exit(main())
