![Project banner](assets/banner_orca_p100.jpeg)

# OrcaSAQ-2-27B on a Tesla P100 — footless in Docker

Run **[OrcaSAQ2 27B](https://huggingface.co/orcarouter/OrcaSAQ-2-27B)** on a
**NVIDIA Tesla P100 16 GB**, a ten-year-old data-center card that sells used for
roughly **100–200 USD**, through an OpenAI-compatible API.

OrcaSAQ2 27B is a 3.21-bit quantization of Qwen3.8-27B: a current-generation
reasoning model with thinking, tool calling and a 262K-token architecture. Its
authors report 70.0% on SWE-bench Verified and 58.4% on Terminal-Bench 2.1. The
original checkpoint is 54 GB; this one is 12.3 GB. Their release targets vLLM on
recent GPUs. This project runs the same weights on Pascal instead, with
**footless**, a small inference engine with CUDA kernels written for this card.

On a P100 you get:

* about **20 tokens/s** of generation, and **~30 tokens/s with MTP speculative
  decoding**;
* **~200 tokens/s** of prompt processing;
* contexts up to **~100k tokens** (~80k with MTP);
* up to **4 requests at once**, ~44 tokens/s in total.

> [!WARNING]
> **Experimental.** footless is a young engine and this is its first packaged
> release. The kernels exist for one GPU model only. Expect rough edges, and
> do not use it where a failure would be costly.


**Contents:** [Prerequisites](#prerequisites) ·
[Setup and first start](#setup-and-first-start) · [Endpoints](#endpoints) ·
[Quality](#quality) · [Performance](#performance) · [Examples](#examples) ·
[Credits](#credits)

---

## Prerequisites

### Hardware

| | Requirement |
| --- | --- |
| GPU | **Tesla P100 16 GB** (compute capability 6.0): PCIe, which is tested, or SXM2, the same chip, untested. Not supported: the 12 GB P100, which cannot hold the weights, and every other GPU, since the kernels are built for `sm_60` only. |
| GPU memory | The card must be **free for this server**: under load the model uses ~15.8 of its 16 GB. |
| Cooling | The PCIe P100 has **no fan**. It needs server-chassis airflow, or a blower or fan shroud in a desktop. See [Performance](#thermal-throttling). |
| Power | 250 W board. The PCIe card takes an **8-pin CPU (EPS-12V)** connector, not a PCIe one. Most desktop PSUs need an adapter. |
| Motherboard | **Above 4G Decoding** enabled in the BIOS (needed for a 16 GB data-center card). |
| Host | x86-64 CPU, 8 GB of RAM or more. Disk: **13 GB** for the weights and 0.6 GB for the image. Building also needs **~12 GB** of temporary space: Docker caches the CUDA development image that compiles the kernels, and `docker builder prune` frees it afterwards. |

### Software

| | Requirement |
| --- | --- |
| OS | Linux, any distribution the NVIDIA Container Toolkit supports. |
| NVIDIA driver | **560 to 580 series**. The kernels are compiled with CUDA 12.6, so the driver must be 560 or newer. **580 is the last driver branch that supports Pascal**: newer branches do not see the P100. Tested with 580.173.02. |
| Docker | Docker Engine with the Compose v2 plugin (`docker compose`). Your user needs access to it (`docker` group). |
| NVIDIA Container Toolkit | Installed and configured for Docker, so that containers can use the GPU. |
| Network | Internet access to huggingface.co on first start. A Hugging Face token is optional. |

On Ubuntu, for example:

```bash
# NVIDIA driver (580 series), then reboot
sudo apt install nvidia-driver-580-server
```

```bash
# Docker Engine + Compose plugin: https://docs.docker.com/engine/install/
# then use docker without sudo (log out and in afterwards)
sudo usermod -aG docker $USER
```

```bash
# NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
sudo apt install nvidia-container-toolkit
```

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

### Checking the requirements

The checks run in two places:

1. **On the host, before building:** `./check-host.sh` checks the OS, Docker,
   Compose, the driver version, the Container Toolkit, the GPUs (and which one
   is the P100), the disk space and the port. It ends by running a CUDA container
   on the selected GPU. Add `--quick` to skip that last step.
2. **In the container, at every start, before anything else:**
   `docker/preflight.py` checks:
   * the driver is reachable;
   * it supports CUDA 12.6;
   * the GPU is a P100 with 16 GB;
   * enough GPU memory is free;
   * there is disk space for the download.

   If one of these fails, the log says which one and what to do. Nothing is
   downloaded, the server does not start, and the container tries again every
   minute until `docker compose down`. To run only these checks:
   `docker compose run --rm orcasaq check`. Stop the server first
   (`docker compose down`), since a running server holds the GPU memory that
   this check looks for.

Docker cannot see the GPU while it builds an image. That is why the hardware is
checked by the script and at start rather than during `docker build`.

---

## Setup and first start

```bash
cd footless-orcasaq-2-27B-nvidia-p100
```

```bash
./check-host.sh
```

`check-host.sh` creates `.env` from `.env.example` the first time, set to your
user. Review it, then build and start:

```bash
docker compose up -d --build
```

```bash
docker compose logs -f
```

What the first start does:

1. **Preflight**: the checks above, a few seconds.
2. **Download**: 12.3 GB from `orcarouter/OrcaSAQ-2-27B` into `MODELS_DIR`, at
   the revision this project was tested with. The log reports progress every
   30 s. If the download is interrupted, the next start keeps the files that
   finished and downloads the rest; a file cut off mid-way starts over.
3. **Verification**: SHA-256 of the weight files, once (about 30 s).
4. **Install**: `footless/` (runtime and kernels) is copied into
   `MODELS_DIR/OrcaSAQ-2-27B/`.
5. **Load**: weights onto the GPU. This takes a few seconds when the files are
   in the OS cache, and up to a couple of minutes from a slow disk. The server
   is ready when the log shows `serving OrcaSAQ-2-27B on
   http://0.0.0.0:8080/v1`.

On the test machine, with a ~90 MB/s connection, the whole first start took
about 3 minutes. Later starts skip steps 2 and 3: they find the files in
`MODELS_DIR` with their exact sizes, and need no network. On the test machine
a restart was serving again in about 15 s.

```bash
curl http://localhost:8080/health          # 200 when ready
```

```bash
docker compose ps                          # STATUS shows (healthy)
```

### Configuration (`.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_DEVICE` | `0` | GPU to use: index or UUID from `nvidia-smi -L`. |
| `HOST` | `0.0.0.0` | Host interface the API is published on. `0.0.0.0` means every interface, reachable from the network. `127.0.0.1` means this machine only. |
| `PORT` | `8080` | API port, on the host and in the container. |
| `API_KEY` | *(empty)* | If set, `/v1` routes require `Authorization: Bearer <API_KEY>`. |
| `MTP` | `1` | Speculative decoding with the model's MTP head: `1` on, `0` off. Off gives up ~1.5x generation speed and raises the context limit from ~80k to ~100k tokens. |
| `MODELS_DIR` | `./models` | Host folder that holds `OrcaSAQ-2-27B/`. |
| `HF_TOKEN` | *(empty)* | Optional [Hugging Face token](https://huggingface.co/settings/tokens) (read access) for faster downloads with higher rate limits. |
| `HF_REVISION` | *(empty)* | Checkpoint revision. Empty means the tested one. Another revision skips the size and hash checks. |
| `PUID` / `PGID` | `1000` | User the container runs as, so downloaded files belong to you (`id -u`, `id -g`). |

After editing `.env`, apply it:

```bash
docker compose up -d
```

### Hosts with several GPUs

Only the GPU named in `GPU_DEVICE` is passed into the container, so the other
cards stay free for other work. To find the P100:

```bash
nvidia-smi -L
```

```
GPU 0: NVIDIA GeForce RTX 3060 (UUID: GPU-8a1f...)
GPU 1: Tesla P100-PCIE-16GB (UUID: GPU-5ec6a13f-17fb-193c-e964-95a811c641c8)
```

For more detail:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap --format=csv
```

The P100 is the 16 GB card with compute capability `6.0`. Put its **UUID** in
`.env`. Indexes can change when cards are added, removed or reordered by the
BIOS; the UUID does not.

```ini
GPU_DEVICE=GPU-5ec6a13f-17fb-193c-e964-95a811c641c8
```

`./check-host.sh` lists every GPU, marks the P100, and fails with the line to
use if `GPU_DEVICE` points to another card. The container's preflight refuses
any GPU that is not a 16 GB P100.

### Using weights you already have

Point `MODELS_DIR` to the folder that **contains** `OrcaSAQ-2-27B/`. The
container checks every file's size and downloads only what is missing or
incomplete.

The container replaces `OrcaSAQ-2-27B/footless/` with its own copy at every
start, so the runtime always matches the image. A `footless/` folder that the
container did not create, such as a development copy, is never deleted: it is
renamed to `footless.backup-<date>`.

### Everyday commands

```bash
docker compose logs -f                        # follow the server log
```

```bash
docker compose restart                        # restart the server
```

```bash
docker compose down                           # stop and remove the container (weights stay)
```

```bash
docker compose up -d --build                  # rebuild after updating this project
```

```bash
docker compose run --rm orcasaq check         # run the preflight checks only
```

### If something fails

| Symptom | Cause and fix |
| --- | --- |
| `permission denied ... docker.sock` | Your user cannot use Docker. Run `sudo usermod -aG docker $USER`, then log out and in. |
| `could not select device driver "nvidia"` | The NVIDIA Container Toolkit is missing or not configured. Install it, run `sudo nvidia-ctk runtime configure --runtime=docker`, and restart Docker. |
| `nvidia-container-cli: device error: ...: unknown device` | `GPU_DEVICE` in `.env` names no GPU of this host. `./check-host.sh` prints the right value. |
| `[preflight] FAIL ...` or `[prepare] ERROR ...` in the log | A requirement is not met. The line says which one and what to do. The container tries again every minute until it is fixed: run `docker compose down` while you fix it. |
| `only X GiB of GPU memory free` | Something else is using the P100. `nvidia-smi` on the host lists the processes. |
| `cannot write to /app/models` | `PUID`/`PGID` in `.env` do not own `MODELS_DIR`. Use your `id -u`/`id -g`, or `chown` the folder. |
| HTTP 500 `budget ... cannot admit ...` | The conversation is past the context limit (~80k tokens with MTP, ~100k without). Start a new conversation, shorten it, or set `MTP=0`. |
| Download stops or fails | The container retries every minute, or start it again with `docker compose up -d`. Finished files are kept. A `HF_TOKEN` avoids anonymous rate limits. |

### Project layout

| Path | What it is |
| --- | --- |
| `footless/` | The footless inference engine and its OpenAI-compatible server. |
| `model/footless/` | The model package: `manifest`, plus `cuda_sm60/` with the runtime, the tokenizer and the CUDA kernel sources for the P100. |
| `docker/` | `preflight.py` (requirement checks), `prepare_model.py` (download, verification, install), `entrypoint.sh`, Python requirements. |
| `Dockerfile` | Compiles the kernels with CUDA 12.6, then builds a small runtime image without the compiler. |
| `compose.yaml`, `.env.example` | The service and its settings. |
| `check-host.sh` | The host-side requirement check. |
| `models/` | Default download folder for the weights. It is not part of the project. |

---

## Endpoints

Base URL: `http://<host>:<PORT>`. The served model name is **`OrcaSAQ-2-27B`**.
The API follows vLLM's OpenAI-compatible wire format, so OpenAI clients work
unchanged.

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/health` | 200 when the server is up. |
| `GET` | `/v1/models` | The served model, with `max_model_len`. This is always 102,400, the limit without MTP (see [Context limits](#context-limits)). |
| `POST` | `/v1/chat/completions` | Chat. Supports streaming (SSE), thinking, function tools, `n`, `seed`, `stop`, and the sampling and penalty knobs. |
| `POST` | `/v1/completions` | Raw-text completion, with or without streaming. |
| `POST` | `/v1/decisions` | **Typed questions without generation.** It scores a `choice`, `score` or `yes_no` question about an input and returns the probability of every option. |
| `POST` | `/v1/systemone` | The same scoring in the System One 0.2.0 request and response shapes. |

Things to know:

* **Several requests at once.** Up to 4 generate together, sharing each pass
  over the weights; more wait in arrival order. Each answer is exactly the one
  the request would get alone. See [Several requests at once](#several-requests-at-once).
* **Thinking** is on by default, at the template's `xhigh` level. Choose the
  level with `reasoning_effort`: `"xhigh"`, `"medium"` or `"low"`. `"none"`
  turns thinking off. The thinking text comes back in the `reasoning` field,
  next to `content`.
* **Multi-turn chats:** send each assistant turn back with its `reasoning`. The
  server then reuses the cached conversation, and a new turn only processes the
  new tokens: `usage.prompt_tokens_details.cached_tokens` shows how many were
  reused. Without it, the server re-reads the conversation from the first turn
  that lacks its reasoning.
* **Sampling defaults** are the model's own recommendation: `temperature` 1.0,
  `top_p` 0.95, `top_k` 20. Any field you send overrides them.
* **Fields the engine cannot execute** are accepted and ignored, as vLLM does,
  and the server log names them. Two cases to know:
  * `logprobs` / `top_logprobs`: this model package does not compute them. The
    response keeps the `logprobs` shape, but its values are placeholders
    (`-9999.0`, no top entries).
  * `tool_choice` `"required"` or a named function is served as `"auto"`.

  Fields that no OpenAI-compatible server knows are ignored without a log
  line.
* A generation that collapses into a verbatim loop is cut after one cycle.

### Chat (curl)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "OrcaSAQ-2-27B",
        "messages": [{"role": "user", "content": "Explain what a B-tree is in three sentences."}],
        "reasoning_effort": "low",
        "max_tokens": 2000
      }'
```

### Chat (Python, OpenAI client, streaming)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")  # or your API_KEY

stream = client.chat.completions.create(
    model="OrcaSAQ-2-27B",
    messages=[{"role": "user", "content": "Write a Python function that merges two sorted lists."}],
    reasoning_effort="none",          # answer directly, no thinking
    max_tokens=1000,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.content:
        print(delta.content, end="", flush=True)
```

### Decisions

Each question is scored at the answer position, so it costs one prompt pass and
no generated tokens.

```bash
curl http://localhost:8080/v1/decisions \
  -H "Content-Type: application/json" \
  -d '{
        "input": "I was charged twice this month and nobody answers my emails. Fix it today.",
        "questions": [
          {"id": "team", "type": "choice", "question": "Which team should handle this ticket?",
           "options": [{"name": "billing", "description": "Payment or subscription issues"},
                       {"name": "technical", "description": "Bugs or integration problems"},
                       {"name": "sales", "description": "Pricing or account questions"}]},
          {"id": "frustration", "type": "score", "question": "How frustrated is the customer?",
           "levels": ["Calm", "Frustrated but civil", "Very angry"]},
          {"id": "urgent", "type": "yes_no", "question": "The customer needs an answer today."}
        ]
      }'
```

Each answer has these fields:

* `probabilities`: one per option.
* `choice` for `choice` questions, and `score` (the expected level index) for
  `score` questions.
* `label_mass`: the share of the model's probability that went to the valid
  answers at all.

```json
{"object": "decisions", "model": "default", "prompt_format_version": 1,
 "answers": {
   "team":        {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.9994, "technical": 0.0003, "sales": 0.0003},
                   "label_mass": 0.997},
   "frustration": {"type": "score", "score": 1.78,
                   "probabilities": {"0": 0.016, "1": 0.191, "2": 0.793},
                   "label_mass": 0.961},
   "urgent":      {"type": "yes_no",
                   "probabilities": {"yes": 0.886, "no": 0.114},
                   "label_mass": 0.538}},
 "usage": {"prompt_tokens": 189, "completion_tokens": 0, "total_tokens": 189}}
```

(A real response from this server, rounded. `score` probabilities are keyed by
level index.)

### System One

`/v1/systemone` takes the same questions in the System One 0.2.0 shape:
* `state` instead of `input`;
* `questions` as a map from id to question;
* question types `noul` (yes/no), `choice` and `score`, each with
  `instructions` and `criteria`.

`model` is required, but any name is accepted and the response reports the
served one.

```bash
curl http://localhost:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
        "model": "OrcaSAQ-2-27B",
        "state": "I was charged twice this month and nobody answers my emails. Fix it today.",
        "questions": {
          "urgent": {"type": "noul", "instructions": "The customer needs an answer today."},
          "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                   "criteria": {"billing": "Payment or subscription issues",
                                "technical": "Bugs or integration problems",
                                "sales": "Pricing or account questions"}},
          "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                          "criteria": ["Calm", "Frustrated but civil", "Very angry"]}
        }
      }'
```

```json
{"model": "OrcaSAQ-2-27B",
 "answers": {
   "urgent":      {"type": "noul", "noul": 0.886, "x_label_mass": 0.538},
   "team":        {"type": "choice", "choice": "billing", "confidence": 0.999,
                   "probabilities": {"billing": 0.9994, "technical": 0.0003, "sales": 0.0003},
                   "x_label_mass": 0.997},
   "frustration": {"type": "score", "score": 1.78, "confidence": 0.665,
                   "legend": {"0": "Calm", "1": "Frustrated but civil", "2": "Very angry"},
                   "probabilities": {"0": 0.016, "1": 0.191, "2": 0.793},
                   "x_label_mass": 0.961}},
 "usage": {"input_tokens": 189, "output_tokens": 0}}
```

See [Performance](#decision-endpoints) for latencies.

---

## Quality

The weights are the published OrcaSAQ2 checkpoint, used as they are: nothing
is re-quantized. The kernels decode its mixed-precision trellis format (3 to 6
bits per weight) directly.
The choices this runtime makes on top of that were each measured against a
reference before being kept:

* an int8 KV cache;
* fp16 arithmetic in the prompt GEMM;
* fused and reordered kernels.

| Check | OrcaSAQ2 model card (vLLM) | This runtime (P100) |
| --- | --- | --- |
| WikiText-2 perplexity, 16,376 predicted tokens | BF16 5.6468 · OrcaSAQ2 5.6482 | **5.5425** (8 windows × 2,048 tokens) |
| Top-1 agreement / mean KLD vs BF16 | 93.2% / 0.031 | not measured: no BF16 reference fits on this card |
| int8 KV cache vs fp16 KV cache | — | PPL 5.5433 vs 5.5425 (+0.015%); KL at the level of a numerically neutral change |
| MTP speculative decoding | — | **lossless**: output identical to non-speculative decoding, greedy and sampled |
| Exact copy of 600 random words from the prompt | — | 600 / 600 |
| Long sampled generations (3 prompts × 3 seeds × 2,500 tokens, model recipe, MTP on) | — | 0 verbatim loops, distinct 8-grams 0.96–1.00 |

On the perplexity numbers: the model card does not publish its exact text
preparation and window placement. This project's 5.5425 comes from the
same-sized protocol, 8 × 2,048-token windows and 16,376 predicted tokens.
Read the two figures as "the same model, no sign of degradation", **not** as
this runtime being better than BF16.

Each speed-up was checked against the previous kernels:

* a teacher-forced pass through the generation path, with mean NLL
  2.60659 → 2.60657 after a 512-token prefix and 3.17144 → 3.17152 after
  30,000 tokens;
* a full-distribution KL divergence at the level of a change in summation order
  (~2–3·10⁻⁵).

The model inherits the capabilities, biases and limitations of Qwen3.8-27B and
of its quantization (see the
[model card](https://huggingface.co/orcarouter/OrcaSAQ-2-27B)). It is
text-only.

---

## Performance

Measured on a Tesla P100-PCIE-16GB with driver 580. "Context" is the number of
tokens already in the conversation.

### Generation (decode)

Short context, tokens per second:

| Mode | Greedy | Sampled |
| --- | --- | --- |
| Plain decoding (`MTP=0`) | 20.8 | 20.1 |
| MTP, writing code | 32.5 | 30.8 |
| MTP, writing prose (Spanish) | 29.7 | 27.5 |

MTP drafts two tokens ahead with the model's own MTP head and verifies them in
one pass. The speed-up depends on how predictable the text is: code gains the
most.

By context length:

| Context | Plain decoding, time per token | Plain decoding, tok/s | MTP, prose, default sampling (tok/s) |
| --- | --- | --- | --- |
| ~0.5–1k | 48 ms | 20.8 | 24.1 |
| ~10k | — | — | 27.3 |
| ~16k | 49.8 ms | ~20 | — |
| ~32k | 51.5 ms | ~19.4 | 19.3 |
| ~64–69k | 54.9 ms | ~18.2 | 17.7 |

The MTP column is one 256-token continuation of a WikiText article per row,
over HTTP. Its speed varies by about ±15% with the text being written.

### Prompt processing (prefill)

A new prompt, processed from scratch (MTP on, as served):

| Prompt | Time to first token | Throughput |
| --- | --- | --- |
| ~1k tokens | 4.9 s | 222 tok/s |
| ~10k tokens | 47 s | 212 tok/s |
| ~34k tokens | 197 s | 174 tok/s |
| ~69k tokens | 494 s (8 min) | 140 tok/s |
| ~81k tokens | 592 s (10 min) | 137 tok/s |

Attention cost grows with context, so very long prompts are slow from
scratch: 100k tokens take about 15 minutes. Conversations do not pay that on
every turn. The server keeps the processed conversation, and a follow-up turn
only processes its new tokens: about 4 s for the next turn of a
100k-token conversation.

### Several requests at once

Up to 4 requests generate together: each step reads the weights once for all
of them, so the total rate grows while each request slows down. Measured over
HTTP with `MTP=1`, 256 tokens a request, thinking off:

| Requests at once | Total | Each request |
| --- | --- | --- |
| 1 | 29.7 tok/s | 29.7 tok/s |
| 2 | 31.4 tok/s | 15.7 tok/s |
| 3 | 39.3 tok/s | 13.1 tok/s |
| 4 | 43.8 tok/s | 10.9 tok/s |

Every answer is token for token the one the request gets alone, greedy or
with a fixed `seed`. MTP speeds up a request only while it runs alone, which
is why 2 at once gain little over 1.

Prompts are processed one at a time. While others generate, a new prompt is
processed in chunks of 1024 tokens, and the others pause for each chunk:
about 5 s at a time. A 7k-token prompt that arrived while three requests were
generating took 32 s to its first token, the same as alone.

The requests share the memory of the context limits below. A request that
does not fit waits until one ends; long prompts at once are therefore
processed one after another. Decision requests run between generation steps
and add no wait of their own: 0.9 s for two questions, measured while two
requests were generating.

### Context limits

| | Maximum context (one conversation) |
| --- | --- |
| `MTP=0` | ~100k tokens (102,400) |
| `MTP=1` | ~80k tokens |

The limit counts the prompt and the generated tokens together, and requests
running at once share it. The KV cache is int8 at 34 KiB per token. These
limits assume the card is used by nothing else.

A conversation that grows past the limit fails with HTTP 500 and the message
`budget ... cannot admit ...`. A long prompt can fail only after it has been
processed for several minutes, so keep conversations under the limit, or set
`MTP=0` for the extra room.

### Decision endpoints

`/v1/decisions` and `/v1/systemone`, with yes/no questions, measured over HTTP
on the running server. "Cold" means the server has not seen the text before.
After a cold request, the processed text stays cached, and new questions about
the same text skip it:

| Text + question (tokens) | 1 question, cold | 3 questions, cold | Same text again: 1 question | 2 questions | 3 questions |
| --- | --- | --- | --- | --- | --- |
| ~200 | 1.2 s | 1.6 s | 0.24 s | 0.39 s | 0.64 s |
| ~900 | 4.3 s | 4.8 s | 0.24 s | 0.40 s | 0.65 s |
| ~2,600 | 11.8 s | 12.9 s | 0.26 s | 0.43 s | 0.68 s |

A cold request costs about one prompt pass over the text. Each extra question
about the same text adds ~0.2 s.

### Thermal throttling

The P100 slows itself down when it runs hot. Under sustained load a poorly
cooled card reaches ~79 °C and the driver holds the SM clock at ~1190 MHz
instead of ~1330 MHz, which costs **~10% of prompt-processing speed**. With
worse cooling it loses more.

* Give the card strong airflow: a server chassis, or a blower or fan shroud
  made for passive Tesla cards.
* Watch it under load:

  ```bash
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.active --format=csv -l 2
  ```

* The numbers above were taken on a card that spent part of the time
  clock-capped. A well-cooled card should match or beat them.

