"""Tokenization for the OrcaSAQ-2-27B package: a stdlib-only byte-level BPE reader.

No dependency, on purpose: this package runs against the system Python, which has
no `tokenizers` (the Rust library the checkpoint was converted with) and no
`transformers`. The checkpoint ships `tokenizer.json` -- 12.8 MB of JSON, which
`json.load` parses in 0.21 s and the rest of this module's load adds another
0.25 s to -- so it reads that file, builds the byte-level alphabet, the merge
ranks and the added-token table out of it, and reimplements the reference's
encoding pipeline against it:

  * normalizer: NFC (`unicodedata.normalize`), the one rule `tokenizer.json`
    carries.
  * pre-tokenizer: the Qwen split regex, transcribed by hand into `_match`
    (Python's `re` has no `\\p{L}`, `\\p{M}`, `\\p{N}`, and its `\\s` is not the
    regex crate's), then the GPT-2 `bytes_to_unicode` table.
  * model: BPE over the byte-level string. The reference merges the
    lowest-ranked adjacent pair (rank = position in `merges`), leftmost first on
    a tie, and re-offers the pairs a merge creates; `_bpe` does the same on a
    heap.
  * added tokens: the 33 tokens `tokenizer.json` lists are split out of the text
    by leftmost-longest match before normalization or pre-tokenization, exactly
    as the reference's `AddedVocabulary` does.

`chat()` renders `chat_template.jinja` by hand (Jinja is not in the standard
library), the way the Xing4.0 package does. The file must still hash to
`CHAT_TEMPLATE_SHA256`; the loader refuses a different one instead of silently
emitting a different prompt.

`decode` renders every id as text, including the special ones -- it is the
reference's `Tokenizer.decode(ids, skip_special_tokens=False)`, so
`decode(encode(text))` is `text` again for text holding a special token. An id
the vocabulary does not have is an error, not a silent drop (the Rust reference
drops it; a wrong id is worth hearing about). Note that `config.json` declares
248320 embedding rows while the tokenizer has `VOCAB_SIZE` 248077, so a sampler
must not emit the unused tail.

Verified against the reference -- `tokenizers` 0.23.2 (`Tokenizer.from_file`)
for `encode`/`decode`, `jinja2` 3.1.6 rendering this checkpoint's
`chat_template.jinja` for `chat` -- over the corpus `../tests/test_tokenizer.py`
describes.

    tok = Tokenizer(model_dir)      # reads tokenizer.json + tokenizer_config.json
    tok.encode("hello")             # [14556]
    tok.decode([14556])             # "hello"
    tok.chat([{"role": "user", "content": "hi"}], thinking=False)
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "Tokenizer",
    "ENDOFTEXT_ID", "IM_START_ID", "IM_END_ID", "THINK_ID", "THINK_END_ID",
    "TOOL_CALL_ID", "TOOL_CALL_END_ID", "TOOL_RESPONSE_ID", "TOOL_RESPONSE_END_ID",
    "THINKING_MARKERS", "VOCAB_SIZE",
]

# The ids the runtime needs, from tokenizer_config.json's added-token table
# (cross-checked against tokenizer.json's; see `_read_config`). `config.json`
# calls 248044 both `bos_token_id` and `eos_token_id`; `tokenizer_config.json`
# calls 248046 its `eos_token` and 248044 its `pad_token`. `encode` adds neither.
ENDOFTEXT_ID = 248044
IM_START_ID = 248045
IM_END_ID = 248046
TOOL_CALL_ID = 248058
TOOL_CALL_END_ID = 248059
TOOL_RESPONSE_ID = 248066
TOOL_RESPONSE_END_ID = 248067
THINK_ID = 248068
THINK_END_ID = 248069
# `Facts.thinking_markers`: index 0 opens a thinking block, index 1 closes one.
THINKING_MARKERS = (THINK_ID, THINK_END_ID)

# The reasoning efforts `chat_template.jinja` accepts, in the template's own
# order. `medium` is the one with nothing attached: the template's if/elif has
# no else, so it adds no reasoning instructions at all. The runtime reports this
# list in its `Facts` and the package's manifest declares it; the engine makes a
# disagreement between the two a hard error.
THINKING_LEVELS = ("xhigh", "medium", "low")
# The tokenizer's own vocabulary; config.json's vocab_size (248320) is larger,
# and its extra 243 embedding rows have no token behind them.
VOCAB_SIZE = 248077

# `chat()` renders chat_template.jinja by hand. That transcription is only known
# to be right for the template it was written from, so the file must still hash
# to this. If the template changes, re-check the renderer against it (and against
# Jinja) instead of silently emitting a different prompt.
CHAT_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"

# The pre-tokenizer pattern, pinned. tokenizer.json's pre_tokenizer is
# `Split(pattern, behavior=Isolated)` then `ByteLevel(use_regex=False)`, and
# tokenizer_config.json carries the same pattern as `pretokenize_regex`; the
# loader refuses a file whose pattern disagrees with this string.
PRETOKENIZE_REGEX = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

# The regex crate's `\s` is `\p{White_Space}`, which is not Python's `\s`: Python
# also calls U+001C..U+001F whitespace, and the reference does not. Spelled out so
# the scanner splits like the reference.
WHITESPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)

# BPE result cache: the reference caches a piece under the same limits
# (`DEFAULT_CACHE_CAPACITY` 10000, `MAX_LENGTH` 256), so this only makes the same
# tokenization faster.
CACHE_CAPACITY = 10_000
CACHE_MAX_LENGTH = 256


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-level alphabet: every byte as one printable character."""
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    spare = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + spare)
            spare += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_CHAR = _bytes_to_unicode()  # byte -> its byte-level character
CHAR_BYTE = {char: byte for byte, char in BYTE_CHAR.items()}  # and back


# --------------------------------------------------------------- pre-tokenizer


def _is_number(ch: str) -> bool:
    """`\\p{N}`, the general category N (Nd, Nl, No) -- not `str.isnumeric()`.

    Python's `isnumeric()` is the `Numeric_Type` property, which is true for 81
    ideographs of category `Lo` (U+3405, U+4E94, ... -- a CJK numeral is a letter
    that names a number) and those are `\\p{L}` to the reference. `isnumeric()`
    is false for every `\\p{N}`, so it is the cheap pre-filter; the category call
    only happens for the ideographs.
    """
    return ch.isnumeric() and unicodedata.category(ch)[0] == "N"


def _word_run(text: str, i: int, n: int) -> int:
    """End of the `[\\p{L}\\p{M}]+` run at `i` (may be `i`)."""
    while i < n:
        ch = text[i]
        if not ch.isalpha() and unicodedata.category(ch)[0] != "M":
            break
        i += 1
    return i


def _other_run(text: str, i: int, n: int) -> int:
    """End of the `[^\\s\\p{L}\\p{M}\\p{N}]+` run at `i` (may be `i`)."""
    while i < n:
        ch = text[i]
        if ch in WHITESPACE or ch.isalpha() or _is_number(ch):
            break
        if unicodedata.category(ch)[0] == "M":
            break
        i += 1
    return i


def _match(text: str, i: int, n: int) -> int | None:
    """One match of the reference pre-tokenizer regex at `text[i]`, or None.

    The pattern is

        (?i:'s|'t|'re|'ve|'m|'ll|'d)         1: English contractions
      | [^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+  2: a word, with a leading space
      | \\p{N}                                3: one number
      |  ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*  4: a run of anything else
      | \\s*[\\r\\n]+                          5: whitespace ending in a newline
      | \\s+(?!\\S)                            6: trailing whitespace
      | \\s+                                   7: whitespace

    The alternatives are tried in order, each greedy with backtracking, which is
    how the regex crate matches leftmost-first -- `re` would agree if it could
    spell those classes. The two backtracks that change the answer are marked.
    """
    ch = text[i]

    # 1. A contraction. `(?i)` in the regex crate is Unicode simple case folding,
    # so 't' also folds from U+017F... only 's' does: U+017F (LONG S) has the
    # single fold to 's', and it is the one non-ASCII character this alternative
    # can match.
    if ch == "'" and i + 1 < n:
        second = text[i + 1]
        if second in "sStT" or second in "mMdD" or second == "\u017f":
            return i + 2
        if i + 2 < n and text[i + 1 : i + 3].lower() in ("re", "ve", "ll"):
            return i + 3

    # 2. `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`: a word, with one leading non-word
    # character allowed to stick to it (the space in " hello").
    if ch.isalpha() or unicodedata.category(ch)[0] == "M":
        return _word_run(text, i, n)  # always past `i`, `ch` is a letter or mark
    if ch not in "\r\n" and not _is_number(ch):
        end = _word_run(text, i + 1, n)  # the optional character, consumed greedily
        if end > i + 1:  # ... and `[\p{L}\p{M}]+` still needs one character
            return end

    # 3. `\p{N}`: one numeric character.
    if _is_number(ch):
        return i + 1

    # 4. ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`: a run of punctuation or symbols, with
    # one leading space allowed to stick to it, then any newlines.
    if ch == " ":
        end = _other_run(text, i + 1, n)
        if end > i + 1:
            while end < n and text[end] in "\r\n":
                end += 1
            return end
    elif ch not in WHITESPACE:
        end = _other_run(text, i, n)
        while end < n and text[end] in "\r\n":
            end += 1
        return end

    # 5, 6 and 7 all start with whitespace, so a non-whitespace character is the
    # end of the pattern.
    if ch not in WHITESPACE:
        return None
    end = i + 1
    while end < n and text[end] in WHITESPACE:
        end += 1
    run_end = end

    # 5. `\s*[\r\n]+`: greedy `\s*` gives characters back until `[\r\n]+` can
    # match, so the match ends where the last newline run in the whitespace run
    # ends -- the second backtrack, and where a space after the newline is left
    # for the next match (" \n k" is " \n" then " k").
    stop = run_end
    while stop > i and text[stop - 1] not in "\r\n":
        stop -= 1
    if stop > i:
        end = stop
        while end < n and text[end] in "\r\n":
            end += 1
        return end

    # 6. `\s+(?!\S)`: the lookahead fails where the whitespace run ends (a
    # non-whitespace character follows it), so the regex matches one character
    # less. A one-character run has nothing to give back, so 6 fails and 7 takes
    # it.
    if run_end == n:
        return run_end
    if run_end - 1 > i:
        return run_end - 1

    # 7. `\s+`.
    return run_end


def pretokenize(text: str) -> list[str]:
    """Splits `text` the way the reference pre-tokenizer's `Split` does.

    The pieces are the pattern's matches. Between them the pattern matches
    almost everything, and `_match` never declines a character that another
    alternative could take (`../tests/test_tokenizer.py` checks a single
    character of every Unicode category); a run of characters that no
    alternative matches is kept as one piece, which is what
    `SplitDelimiterBehavior::Isolated` does with a gap.
    """
    pieces: list[str] = []
    n = len(text)
    index = 0
    gap = 0
    while index < n:
        end = _match(text, index, n)
        if end is None:
            index += 1
            continue
        if gap < index:
            pieces.append(text[gap:index])
        pieces.append(text[index:end])
        index = end
        gap = end
    if gap < n:
        pieces.append(text[gap:n])
    return pieces


# -------------------------------------------------------------------- template

# chat_template.jinja's literals, transcribed. Both the tool-call wording and the
# reasoning instructions are part of the prompt the model was trained on, so they
# are copied exactly; `../tests/test_tokenizer.py` compares every rendering
# against Jinja's, which is what keeps them copied exactly.
REASONING_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
REASONING_LOW = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration."
)
TOOLS_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOLS_TAIL = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the "
    "second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n"
    "</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may "
    "provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n- If there is no function call available, answer the "
    "question like normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>"
)
VISION_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
VISION_VIDEO = "<|vision_start|><|video_pad|><|vision_end|>"

MISSING = object()  # Jinja's `undefined`: absent key, absent attribute


def _attr(obj, name: str):
    """`obj.name` as Jinja reads it: an attribute first, then a mapping key."""
    try:
        return getattr(obj, name)
    except AttributeError:
        pass
    try:
        return obj[name]
    except (TypeError, LookupError, AttributeError):
        return MISSING


def _text(value) -> str:
    """`{{ value }}`: undefined renders as nothing, anything else as `str()`."""
    return "" if value is MISSING else str(value)


def _tojson(value) -> str:
    """Jinja's `tojson` filter: `json.dumps(sort_keys=True)`, HTML-escaped."""
    dumped = json.dumps(value, sort_keys=True)
    return (
        dumped.replace("<", "\\u003c").replace(">", "\\u003e")
        .replace("&", "\\u0026").replace("'", "\\u0027")
    )


def _char_bytes(text: str) -> bytes:
    """One token's characters back to bytes: `ByteLevel::decode_chain`'s rule.

    Every character is looked up in the byte-level alphabet and the resulting
    bytes are returned; a token with a character outside it (a special token's
    text) falls back to its own UTF-8 bytes, which is what the reference does and
    what makes `<|im_start|>` come back whole.
    """
    out = bytearray()
    for char in text:
        byte = CHAR_BYTE.get(char)
        if byte is None:
            return text.encode("utf-8")
        out.append(byte)
    return bytes(out)


# ------------------------------------------------------------------ tokenizer


class Tokenizer:
    """The model's tokenizer: `tokenizer.json` plus `tokenizer_config.json`."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "tokenizer_config.json").read_text("utf-8"))
        self._read_model(json.loads((model_dir / "tokenizer.json").read_text("utf-8")))
        self._read_config(self.config)
        self._check_template(model_dir)
        self._bpe_cache: dict[str, list[int]] = {}
        self._bytes_cache: dict[int, bytes] = {}

    # ------------------------------------------------------------------ load

    def _read_model(self, blob: dict) -> None:
        """The byte-level alphabet, the vocabulary, the merge ranks, the tokens."""
        if blob.get("normalizer") != {"type": "NFC"}:
            raise NotImplementedError(
                f"tokenizer.json normalizes with {blob.get('normalizer')!r}; this reader "
                "implements the NFC normalizer only"
            )
        pre = blob.get("pre_tokenizer") or {}
        steps = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
        if (
            len(steps or ()) != 2
            or steps[0] != {"type": "Split", "pattern": {"Regex": PRETOKENIZE_REGEX},
                            "behavior": "Isolated", "invert": False}
            or steps[1] != {"type": "ByteLevel", "add_prefix_space": False,
                            "trim_offsets": False, "use_regex": False}
        ):
            raise NotImplementedError(
                "tokenizer.json's pre_tokenizer is not the Qwen split + byte-level "
                f"sequence this reader was written for: {pre!r}"
            )
        if blob.get("decoder") != {"type": "ByteLevel", "add_prefix_space": False,
                                   "trim_offsets": False, "use_regex": False}:
            raise NotImplementedError(
                f"tokenizer.json decodes with {blob.get('decoder')!r}; this reader "
                "implements the byte-level decoder only"
            )
        # The post-processor must add no token: `encode` adds no bos/eos.
        post = blob.get("post_processor") or {}
        if post.get("type") != "ByteLevel" or post.get("add_prefix_space") is not False:
            raise NotImplementedError(
                f"tokenizer.json post-processes with {post!r}; this reader implements "
                "the byte-level post-processor (which adds no token) only"
            )

        model = blob["model"]
        if model.get("type") != "BPE":
            raise NotImplementedError(f"tokenizer.json holds a {model.get('type')} model")
        if (
            model.get("unk_token") is not None
            or model.get("byte_fallback")
            or model.get("fuse_unk")
        ):
            raise NotImplementedError(
                "tokenizer.json's BPE has an unk_token, byte_fallback or fuse_unk; this "
                "reader implements the byte-level BPE with none of them (every byte has "
                "a token of its own here, so none of them can be reached)"
            )
        if model.get("ignore_merges") or model.get("dropout") not in (None, 0.0):
            raise NotImplementedError("tokenizer.json's BPE has ignore_merges or dropout")
        if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
            raise NotImplementedError(
                "tokenizer.json's BPE prefixes or suffixes its pieces; this reader does not"
            )

        self.vocab: dict[str, int] = model["vocab"]
        self._vocab_r: dict[int, str] = {index: token for token, index in self.vocab.items()}
        if len(self._vocab_r) != len(self.vocab):
            raise ValueError("tokenizer.json's vocabulary repeats an id")
        # A piece starts out as its single byte-level characters, so all 256 of
        # them must have a token: the reference would otherwise drop the
        # character silently (no unk, no byte fallback to catch it).
        for byte, char in BYTE_CHAR.items():
            if char not in self.vocab:
                raise ValueError(f"tokenizer.json has no token for byte {byte:#04x}")

        # Ranks are the order of `merges`; a pair with no concatenated token
        # could not be encoded at all, and the reference refuses the file too
        # (`MergeTokenOutOfVocabulary`), so this does as well.
        self._ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(model["merges"]):
            left, _, right = merge.partition(" ")
            if left not in self.vocab or right not in self.vocab:
                raise ValueError(f"merges[{rank}] = {merge!r} names a token outside the vocabulary")
            if left + right not in self.vocab:
                raise ValueError(
                    f"merges[{rank}] = {merge!r} concatenates to a token outside the "
                    "vocabulary, which the reference refuses too"
                )
            self._ranks[(left, right)] = rank

        self.added_tokens: dict[int, str] = {}
        self._added_id: dict[str, int] = {}
        self._added_by_head: dict[str, list[str]] = {}
        self._added_special: dict[int, bool] = {}
        for token in blob["added_tokens"]:
            index, content = token["id"], token["content"]
            if token.get("lstrip") or token.get("rstrip") or token.get("single_word"):
                raise NotImplementedError(
                    f"added token {content!r} asks for lstrip/rstrip/single_word; this "
                    "reader splits added tokens by plain leftmost-longest match"
                )
            if token.get("normalized"):
                raise NotImplementedError(
                    f"added token {content!r} is matched against normalized text; this "
                    "reader extracts added tokens from the raw text only"
                )
            self.added_tokens[index] = content
            self._added_id[content] = index
            self._added_special[index] = bool(token.get("special"))
        for content in sorted(self._added_id, key=len, reverse=True):
            self._added_by_head.setdefault(content[0], []).append(content)

    def _read_config(self, config: dict) -> None:
        """Cross-checks the ids against `tokenizer_config.json`'s own table.

        That config is the file the reference reads its special tokens from, so a
        disagreement between the two files is a broken package: refuse it rather
        than pick one of them.
        """
        if not config.get("added_tokens_decoder"):
            raise ValueError(
                "tokenizer_config.json has no added_tokens_decoder table to cross-check "
                "tokenizer.json's added tokens against"
            )
        declared = {
            int(index): spec["content"]
            for index, spec in config["added_tokens_decoder"].items()
        }
        if declared != self.added_tokens:
            differing = {
                index: (declared.get(index), self.added_tokens.get(index))
                for index in declared.keys() | self.added_tokens.keys()
                if declared.get(index) != self.added_tokens.get(index)
            }
            raise ValueError(
                f"tokenizer_config.json and tokenizer.json disagree on added tokens: {differing}"
            )
        declared_special = {
            int(index): bool(spec.get("special"))
            for index, spec in config["added_tokens_decoder"].items()
        }
        if declared_special != self._added_special:
            differing_flags = {
                index: (declared_special[index], self._added_special[index])
                for index in declared_special
                if declared_special[index] != self._added_special[index]
            }
            raise ValueError(
                "tokenizer_config.json and tokenizer.json disagree on which added tokens "
                f"are special: {differing_flags}"
            )
        for name, index in (("bos_token", ENDOFTEXT_ID), ("eos_token", IM_END_ID),
                            ("pad_token", ENDOFTEXT_ID)):
            token = config.get(name)
            if token is not None and self._added_id.get(token) != index:
                raise ValueError(
                    f"tokenizer_config.json calls {token!r} `{name}`, and id {index} is "
                    f"{self.added_tokens.get(index)!r}"
                )
        if config.get("unk_token") is not None:
            raise NotImplementedError("tokenizer_config.json declares an unk_token")
        if config.get("add_bos_token") or config.get("add_eos_token"):
            raise NotImplementedError(
                "tokenizer_config.json asks encode() to add a bos/eos token; this "
                "reader's encode() is the text itself and nothing else"
            )
        if config.get("pretokenize_regex", PRETOKENIZE_REGEX) != PRETOKENIZE_REGEX:
            raise NotImplementedError(
                "tokenizer_config.json's pretokenize_regex is not the pattern this reader "
                f"was written for: {config['pretokenize_regex']!r}"
            )
        if config.get("split_special_tokens"):
            raise NotImplementedError(
                "tokenizer_config.json sets split_special_tokens, which tokenizes the "
                "special tokens as ordinary text; this reader always splits them out"
            )

    def _check_template(self, model_dir: Path) -> None:
        path = model_dir / "chat_template.jinja"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; chat() has nothing to render")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != CHAT_TEMPLATE_SHA256:
            raise ValueError(
                f"{path.name} has changed (sha256 {digest}, expected "
                f"{CHAT_TEMPLATE_SHA256}); chat() is a hand transcription of the old "
                "template and must be re-checked against the new one"
            )
        inline = self.config.get("chat_template")
        if inline is not None and inline != data.decode("utf-8"):
            raise ValueError(
                "tokenizer_config.json's chat_template does not match chat_template.jinja; "
                "the package holds two different templates"
            )

    # ---------------------------------------------------------------- encode

    def encode(self, text: str) -> list[int]:
        """Text to token ids, as the reference tokenizer does it.

        Added tokens are split out first and keep their own id (that is how
        `<|im_start|>`, `<think>` or `<tool_call>` come back as one token); every
        other run of text is NFC-normalized, pre-tokenized and BPE-encoded on its
        own. Nothing is added: this package's config declares no bos, and the
        template writes every marker it wants, so `chat()` needs no help here.
        """
        ids: list[int] = []
        for chunk, token_id in self._split_added(text):
            if token_id is not None:
                ids.append(token_id)
            else:
                for piece in pretokenize(unicodedata.normalize("NFC", chunk)):
                    ids.extend(self._bpe(piece))
        return ids

    def _split_added(self, text: str) -> list[tuple[str, int | None]]:
        """Splits text at added-token contents, leftmost longest first.

        That is the reference's added-token trie (`MatchKind::LeftmostLongest`)
        applied to this text. Every token in this package has `lstrip`, `rstrip`
        and `single_word` false and is matched un-normalized, so there is nothing
        else to honour.
        """
        out: list[tuple[str, int | None]] = []
        by_head = self._added_by_head
        added_id = self._added_id
        index, size, start = 0, len(text), 0
        while index < size:
            for candidate in by_head.get(text[index], ()):
                if text.startswith(candidate, index):
                    if start < index:
                        out.append((text[start:index], None))
                    out.append((candidate, added_id[candidate]))
                    index += len(candidate)
                    start = index
                    break
            else:
                index += 1
        if start < size:
            out.append((text[start:size], None))
        return out

    def _bpe(self, piece: str) -> list[int]:
        """Byte-level BPE of one pre-tokenized piece (no added token inside it).

        The reference's `merge_all`: pop the adjacent pair with the lowest merge
        rank (leftmost first on a tie, which is the order its heap compares in),
        join it, and offer the two pairs the join creates. A piece's symbols
        start one byte-level character wide, and every merge is a pair whose
        concatenation is a token, so each symbol is in the vocabulary.
        """
        cached = self._bpe_cache.get(piece)
        if cached is not None:
            return cached
        symbols = [BYTE_CHAR[byte] for byte in piece.encode("utf-8")]
        count = len(symbols)
        if count > 1:
            left_of = [i - 1 for i in range(count)]
            right_of = [i + 1 if i + 1 < count else -1 for i in range(count)]
            alive = [True] * count
            ranks = self._ranks
            agenda: list[tuple[int, int, int, int, int]] = []  # rank, left, right, lengths

            def offer(left: int, right: int) -> None:
                if left < 0 or right < 0:
                    return
                rank = ranks.get((symbols[left], symbols[right]))
                if rank is not None:
                    heapq.heappush(
                        agenda, (rank, left, right, len(symbols[left]), len(symbols[right]))
                    )

            for i in range(count - 1):
                offer(i, i + 1)
            while agenda:
                top = agenda[0]
                left, right = top[1], top[2]
                # Past its time: a side was merged away, or the symbols at those
                # positions are not the ones this entry was queued for. A merge
                # only makes a symbol longer, so the lengths identify it.
                if (
                    not alive[left] or not alive[right]
                    or len(symbols[left]) != top[3] or len(symbols[right]) != top[4]
                ):
                    heapq.heappop(agenda)
                    continue
                heapq.heappop(agenda)
                symbols[left] += symbols[right]
                alive[right] = False
                right_of[left] = right_of[right]
                if right_of[right] >= 0:
                    left_of[right_of[right]] = left
                offer(left_of[left], left)
                offer(left, right_of[left])
            ids = [self.vocab[symbols[i]] for i in range(count) if alive[i]]
        else:
            ids = [self.vocab[symbols[i]] for i in range(count)]
        if len(piece) < CACHE_MAX_LENGTH and len(self._bpe_cache) < CACHE_CAPACITY:
            self._bpe_cache[piece] = ids
        return ids

    # ---------------------------------------------------------------- decode

    def decode(self, tokens: list[int]) -> str:
        """Token ids to text, the reference's byte-level decoder, verbatim.

        Every id is rendered: an added token as its content, anything else as its
        byte-level characters' bytes. The bytes of the whole sequence are joined
        and decoded once with `replace`, so a character split across two tokens
        comes back whole and a broken run comes back as U+FFFD -- exactly
        `ByteLevel::decode_chain` and then `String::from_utf8_lossy`.

        That is the reference with `skip_special_tokens=False`; the Rust default
        drops the 21 tokens this file marks `special`, which would make
        `decode(encode(text))` lose them. An id the vocabulary does not have is an
        error here, not a silent drop.
        """
        buf = bytearray()
        cache = self._bytes_cache
        for token in tokens:
            piece = cache.get(token)
            if piece is None:
                content = self.added_tokens.get(token)
                if content is None:
                    content = self._vocab_r.get(token)
                    if content is None:
                        raise ValueError(f"invalid token id {token}")
                piece = cache[token] = _char_bytes(content)
            buf.extend(piece)
        return buf.decode("utf-8", "replace")

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: list[dict],
        thinking: bool = True,
        tools: list | None = None,
        add_generation_prompt: bool = True,
        reasoning_effort: str | None = None,
    ) -> list[int]:
        """Conversation to prompt tokens, via this package's `chat_template.jinja`.

        The template's text is transcribed in `_render` rather than rendered by
        Jinja, which is not in the standard library: it is one template for one
        model, and `../tests/test_tokenizer.py` pins the rendering, and the ids it
        encodes to, against a Jinja render of the template file.

        `reasoning_effort` is one of `THINKING_LEVELS`; `None` leaves the template
        its own `'xhigh'`. It only takes effect with `thinking` on, which is the
        template's own rule (`enable_thinking`).
        """
        return self.encode(self._render(messages, thinking, tools, add_generation_prompt,
                                        reasoning_effort))

    def _render(
        self,
        messages: list[dict],
        thinking: bool,
        tools: list | None,
        add_generation_prompt: bool,
        reasoning_effort: str | None = None,
        preserve_thinking: bool = True,
        add_vision_id: bool = False,
    ) -> str:
        """`chat_template.jinja`, transcribed -- no Jinja engine involved.

        Every `{%-` and `-%}` of that template is honoured, so this produces the
        string Jinja would: a control tag contributes no whitespace, a literal
        contributes its own. The arguments after `add_generation_prompt` are the
        other names the template reads; the runtime only uses the first four.
        `reasoning_effort` defaults to the template's own `'xhigh'`, which puts
        the reasoning instructions in the system turn whenever thinking is on.
        """
        if not messages:
            raise ValueError("No messages provided.")
        rows = list(messages)
        tools = list(tools) if tools else []
        effort = "xhigh" if reasoning_effort is None else reasoning_effort
        instructions = ""
        if thinking is not False:  # the template's `undefined or true`
            if effort not in THINKING_LEVELS:
                raise ValueError(
                    f"Unexpected reasoning effort {effort}. Supported types are "
                    "xhigh (default), medium, and low."
                )
            # xhigh and low carry an instruction; medium carries none (the
            # template's if/elif has no else) -- that absence is the feature
            instructions = REASONING_XHIGH if effort == "xhigh" else (
                REASONING_LOW if effort == "low" else ""
            )
        vision = {"image": 0, "video": 0}

        out: list[str] = []
        if tools:
            out.append("<|im_start|>system\n" + (instructions + "\n\n" if instructions else ""))
            out.append(TOOLS_HEAD)
            for tool in tools:
                out.append("\n" + _tojson(tool))
            out.append("\n</tools>" + TOOLS_TAIL)
            if _attr(rows[0], "role") == "system":
                system = self._content(rows[0], vision, system=True, count_vision=False)
                if system:
                    out.append("\n\n" + system)
            out.append("<|im_end|>\n")
        else:
            system = None
            if _attr(rows[0], "role") == "system":
                system = self._content(rows[0], vision, system=True, count_vision=False)
            if system:
                out.append(
                    "<|im_start|>system\n"
                    + (instructions + "\n\n" if instructions else "")
                    + system + "<|im_end|>\n"
                )
            elif instructions:
                out.append("<|im_start|>system\n" + instructions + "<|im_end|>\n")

        # The template records the index of the last user message that is not a
        # pure tool response: assistant turns after it are the only ones that get
        # a thinking block (when `preserve_thinking` is off).
        last_query = len(rows) - 1
        multi_step_tool = True
        for index in range(len(rows) - 1, -1, -1):
            if multi_step_tool and _attr(rows[index], "role") == "user":
                content = self._content(rows[index], vision, count_vision=False)
                tool_response = content.startswith("<tool_response>") and content.endswith(
                    "</tool_response>"
                )
                if not tool_response:
                    multi_step_tool = False
                    last_query = index
        if multi_step_tool:
            raise ValueError("No user query found in messages.")

        for index, message in enumerate(rows):
            role = _attr(message, "role")
            content = self._content(message, vision, add_vision_id=add_vision_id)
            if role == "system":
                if index != 0:
                    raise ValueError("System message must be at the beginning.")
            elif role == "user":
                out.append("<|im_start|>user\n" + content + "<|im_end|>\n")
            elif role == "assistant":
                out.append(self._assistant(message, content, index, last_query, preserve_thinking))
            elif role == "tool":
                if index > 0 and _attr(rows[index - 1], "role") != "tool":
                    out.append("<|im_start|>user")
                out.append("\n<tool_response>\n" + content + "\n</tool_response>")
                if index + 1 == len(rows) or _attr(rows[index + 1], "role") != "tool":
                    out.append("<|im_end|>\n")
            else:
                raise ValueError("Unexpected message role.")

        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
            out.append("<think>\n\n</think>\n\n" if thinking is False else "<think>\n")
        return "".join(out)

    def _assistant(
        self, message, content: str, index: int, last_query: int, preserve_thinking: bool
    ) -> str:
        """One assistant turn: its thinking block, its content, its tool calls."""
        reasoning = _attr(message, "reasoning_content")
        reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
        if preserve_thinking or index > last_query:
            out = ["<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content]
        else:
            out = ["<|im_start|>assistant\n" + content]

        calls = _attr(message, "tool_calls")
        # iterable and not a mapping, as the template tests it
        if calls is not MISSING and calls and not isinstance(calls, Mapping):
            for position, call in enumerate(calls):
                function = _attr(call, "function")
                if function is not MISSING:
                    call = function
                name = _attr(call, "name")
                if name is MISSING:
                    raise ValueError("a tool call has no `name` to write into the prompt")
                # the first call follows the content with a blank line; a later
                # one is a block of its own
                if position == 0:
                    lead = "\n\n" if content else ""
                else:
                    lead = "\n"
                out.append(lead + "<tool_call>\n<function=" + str(name) + ">\n")
                arguments = _attr(call, "arguments")
                if arguments is not MISSING and arguments != "":
                    items = getattr(arguments, "items", None)
                    if not callable(items):
                        raise ValueError(
                            "a tool call's `arguments` must be a mapping (the template "
                            f"iterates them with `|items`), not {arguments!r}"
                        )
                    for args_name, args_value in items():
                        rendered = (
                            args_value if isinstance(args_value, str) else _tojson(args_value)
                        )
                        out.append(f"<parameter={args_name}>\n" + rendered + "\n</parameter>\n")
                out.append("</function>\n</tool_call>")
        out.append("<|im_end|>\n")
        return "".join(out)

    def _content(
        self, message, vision: dict, system: bool = False, count_vision: bool = True,
        add_vision_id: bool = False,
    ) -> str:
        """`render_content(message.content, ...)|trim`, the template's own macro.

        A string is itself; a list is rendered item by item, vision branches
        included; None and an absent key render as nothing; a mapping or any
        other type is an error, as it is in the template.
        """
        content = _attr(message, "content")
        if isinstance(content, str):
            return content.strip()
        if content is MISSING or content is None:
            return ""
        if isinstance(content, Mapping) or not hasattr(content, "__iter__"):
            raise ValueError("Unexpected content type.")
        rendered: list[str] = []
        for item in content:
            if _in(item, "image") or _in(item, "image_url") or _attr(item, "type") == "image":
                if system:
                    raise ValueError("System message cannot contain images.")
                if count_vision:
                    vision["image"] += 1
                if add_vision_id:
                    rendered.append(f"Picture {vision['image']}: ")
                rendered.append(VISION_IMAGE)
            elif _in(item, "video") or _attr(item, "type") == "video":
                if system:
                    raise ValueError("System message cannot contain videos.")
                if count_vision:
                    vision["video"] += 1
                if add_vision_id:
                    rendered.append(f"Video {vision['video']}: ")
                rendered.append(VISION_VIDEO)
            elif _in(item, "text"):
                rendered.append(_text(_attr(item, "text")))
            else:
                raise ValueError("Unexpected item type in content.")
        return "".join(rendered).strip()


def _in(item, key: str) -> bool:
    """`key in item` as Jinja evaluates it: a substring test on a string."""
    try:
        return key in item
    except TypeError:
        raise ValueError(f"Unexpected content item {item!r}: {key!r} is not in it") from None


# ------------------------------------------------------------------ tool calls

# A call as the template asks for it (TOOLS_TAIL) and writes it back
# (`Tokenizer._assistant`):
#
#     <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
#
# read the way vLLM's qwen3 parser reads it (vllm/parser/qwen3.py, the
# `qwen3_coder`/`qwen3_xml` tool parsers): `<function=` opens a call even
# without `<tool_call>`, one block may hold several functions, a value loses
# one wrapping newline each side, and the tool's JSON schema types each value
# (`coerce_to_schema_type` in vllm/tool_parsers/utils.py). Stray text after a
# call goes back to the content rather than being lost.
TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"
FUNC_START, FUNC_END = "<function=", "</function>"
_PARAM = re.compile(r"<\s*parameter\s*=\s*([^>]*)>(.*?)"
                    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))", re.DOTALL)
_TYPE_ALIASES = {
    "str": "string", "text": "string", "varchar": "string", "char": "string",
    "enum": "string", "int": "integer", "int32": "integer", "int64": "integer",
    "uint": "integer", "uint32": "integer", "uint64": "integer", "long": "integer",
    "short": "integer", "unsigned": "integer", "float": "number", "float32": "number",
    "float64": "number", "double": "number", "bool": "boolean", "dict": "object",
    "arr": "array", "list": "array", "sequence": "array",
}


def tool_calls(text: str, tools: list, final: bool) -> tuple[str, list[dict]]:
    """The answer `text` as (content, calls), calls {"name", "arguments": JSON}.

    The `tool_calls` verb (footless.sdk.TOOL_CALLS). With `final` false the
    text may still grow: content stops short of a possible marker head, and a
    call counts once its `</function>` is in. With `final` true an unclosed
    call whose name is complete is read as far as its complete parameters go.
    """
    content: list[str] = []
    calls: list[dict] = []
    pos, n = 0, len(text)
    while True:
        # content: up to the next call
        hits = [(i, m) for m in (TOOL_START, FUNC_START) if (i := text.find(m, pos)) >= 0]
        if not hits:
            tail = text[pos:]
            content.append(tail if final else tail[: len(tail) - _marker_head(tail)])
            break
        start, mark = min(hits)
        content.append(text[pos:start])
        pos = start + len(TOOL_START) if mark == TOOL_START else start
        # a call block: functions, until it closes or something else follows
        while True:
            j = pos
            while j < n and text[j].isspace():
                j += 1
            if text.startswith(TOOL_END, j):
                pos = j + len(TOOL_END)
                break
            if text.startswith(TOOL_START, j):
                pos = j + len(TOOL_START)
                continue
            if text.startswith(FUNC_START, j):
                close = text.find(FUNC_END, j)
                body = text[j + len(FUNC_START): close if close >= 0 else n]
                if close < 0:
                    if final and ">" in body:
                        calls.append(_call(body, tools))
                    return "".join(content), calls
                calls.append(_call(body, tools))
                pos = close + len(FUNC_END)
                continue
            if j >= n or (not final and any(m.startswith(text[j:])
                                            for m in (TOOL_END, TOOL_START, FUNC_START))):
                return "".join(content), calls       # the block's end is not in yet
            break                                    # stray text: content again
    return "".join(content), calls


def _marker_head(tail: str) -> int:
    """How many trailing characters may be the head of a call marker."""
    for k in range(min(len(tail), len(TOOL_START) - 1), 0, -1):
        if TOOL_START.startswith(tail[-k:]) or FUNC_START.startswith(tail[-k:]):
            return k
    return 0


def _call(body: str, tools: list) -> dict:
    """`NAME>` then the parameters, as one call; values typed by the schema."""
    name, _, raw = body.partition(">")
    name = name.strip()
    args = {}
    for match in _PARAM.finditer(raw):
        value = match.group(2)
        value = value[1:] if value.startswith("\n") else value
        value = value[:-1] if value.endswith("\n") else value
        args[match.group(1)] = value
    properties = {}
    for tool in tools or ():
        function = tool.get("function", tool) if isinstance(tool, dict) else {}
        if isinstance(function, dict) and function.get("name") == name:
            properties = (function.get("parameters") or {}).get("properties") or {}
            break
    for key, value in args.items():
        schema = properties.get(key)
        if isinstance(schema, dict):
            args[key] = _coerce(value, _schema_types(schema))
    return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}


def _schema_types(schema) -> set[str]:
    """Every type a JSON schema allows: `type`, `enum` values, any/one/allOf."""
    if not isinstance(schema, dict):
        return {"string"}
    types: set[str] = set()
    declared = schema.get("type")
    for t in declared if isinstance(declared, list) else [declared]:
        if isinstance(t, str):
            types.add(t)
    enum = schema.get("enum")
    if isinstance(enum, list):
        for value in enum:
            types.add("null" if value is None else "boolean" if isinstance(value, bool)
                      else "integer" if isinstance(value, int)
                      else "number" if isinstance(value, float)
                      else "string" if isinstance(value, str)
                      else "array" if isinstance(value, list)
                      else "object" if isinstance(value, dict) else "string")
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            for choice in schema[key]:
                types |= _schema_types(choice)
    return types or {"string"}


def _finite(value) -> bool:
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False


def _coerce(value: str, types: set[str]):
    """A parameter's text as its schema's type, first that fits in vLLM's
    order (null, integer, number, boolean, object, array, string)."""
    types = {_TYPE_ALIASES.get(t.strip().lower(), t.strip().lower()) for t in types}
    for kind in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if kind not in types:
            continue
        if kind == "null" and value.lower() == "null":
            return None
        if kind == "string":
            return value
        if kind == "integer":
            try:
                return int(value)
            except ValueError:
                continue
        if kind == "number":
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                return number if number != int(number) else int(number)
        if kind == "boolean":
            if value.lower().strip() in ("true", "1"):
                return True
            if value.lower().strip() in ("false", "0"):
                return False
        if kind in ("object", "array"):
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if _finite(parsed):
                return parsed
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    return parsed if _finite(parsed) else value

