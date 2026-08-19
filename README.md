# ComfyUI H3 Story → Sequences

A ComfyUI custom node that ports the **Story → Sequences** pipeline of the
*H3 Prompt Studio* desktop app into a single graph node.

Feed it up to **9 reference images**, and it drives three local LLM passes to
produce **10 ready-to-use MiniMax H3 Ref2VA prompts** — one per narrative
sequence — plus the generated story text, all without leaving ComfyUI.

Everything runs against a **local** LLM server (Ollama, LM Studio, or
llama.cpp). No cloud API keys, no external calls other than to your own
machine (or LAN) LLM backend.

---

## Table of contents

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Installation](#installation)
- [Node reference: `H3 Story → Sequences`](#node-reference-h3-story--sequences)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
- [How the pipeline works internally](#how-the-pipeline-works-internally)
- [Typical graph setup](#typical-graph-setup)
- [Reproducibility (seed)](#reproducibility-seed)
- [Backends](#backends)
- [Troubleshooting](#troubleshooting)
- [File layout](#file-layout)
- [Credits](#credits)

---

## What it does

Given a handful of reference images (characters, creatures, props, sets,
enemies...), the node:

1. Describes each image with a **vision LLM**.
2. Writes a short **story** from those descriptions + your premise, with a
   **story LLM**.
3. Splits that story into **N narrative sequences** (JSON) with a
   **sequence LLM**.
4. Expands each sequence into a **complete, standalone H3 Ref2VA prompt**
   (the same 6-section format used by MiniMax's Ref2VA model:
   `subject_definitions`, `summary`, `retention_analysis`,
   `detailed_description`, `overall_soundscape`, `non_diegetic_music`),
   using the same sequence LLM.

The result: 10 numbered STRING outputs, one fully-formed video prompt per
sequence, ready to route into whatever downstream node consumes your H3
Ref2VA prompts (text file writer, API call node, Show Text, etc.).

This node is a **direct port** of `sequence_pipeline.py` / `llm_client.py`
from the original Tkinter desktop app — same system prompts, same JSON
schema, same retry/repair logic for truncated LLM output, same final
Ref2VA system prompt. Nothing was reinvented; it was just re-wired for a
ComfyUI graph.

---

## Requirements

- ComfyUI (any reasonably recent version).
- Python packages (installed automatically if you use ComfyUI's embedded
  Python + `pip install -r requirements.txt`, or install manually):
  - `requests`
  - `numpy`
  - `Pillow`
- A running local LLM server, one of:
  - **[Ollama](https://ollama.com)** — default host `http://localhost:11434`
  - **[LM Studio](https://lmstudio.ai)** (Developer → Start Server) — default host `http://localhost:1234`
  - **llama.cpp** (`llama-server`) — default host `http://localhost:8080`
- At least one **vision-capable** model (e.g. `llava`, `qwen2-vl`,
  `minicpm-v`, `bakllava`) pulled/loaded for the image-description step,
  and at least one **text** model for the story/sequence/prompt passes.
  A single strong multimodal model can be used for all three roles.

---

## Installation

1. Download/clone this folder into your ComfyUI custom nodes directory:

   ```
   ComfyUI/custom_nodes/Comfyui_MinimaxH3_StoryToMultiprompts/
   ```

2. Install the Python dependencies (from ComfyUI's Python environment):

   ```bash
   pip install requests numpy Pillow
   ```

3. Start (or restart) ComfyUI.

4. Start your local LLM server (Ollama / LM Studio / llama.cpp) **before**
   opening the node's dropdowns — the `vision_model` / `story_model` /
   `sequence_model` combos are populated by querying
   `http://localhost:11434/api/tags` at graph-load time. If no server is
   reachable, the combos fall back to a placeholder
   `(saisir un nom de modele)` — the widget still accepts free text, you can
   just type the model name manually.

5. In ComfyUI, add the node via the search menu or right-click →
   `Add Node → H3 Prompt Studio → H3 Story → Sequences (9 refs → 10 prompts)`.

---

## Node reference: `H3 Story → Sequences`

### Inputs

#### Images (one socket per reference, not a batch)

Each reference is its own graph socket so you can wire 9 independent
`Load Image` nodes (or any other IMAGE-producing node) directly in:

| Input | Required | Notes |
|---|---|---|
| `image_1` | **Yes** | At least one reference image is mandatory. |
| `image_2` … `image_9` | No | Leave unconnected if you have fewer than 9 references. |
| `role_1` … `role_9` | No | Free-text label for what this image represents (e.g. `"main character"`, `"enemy mecha"`, `"set reference"`). Defaults to `"Reference N"` if left empty. Passed to the vision LLM as context and reused verbatim in every generated prompt's reference library. |

Images are converted to temporary PNG files, sent to the vision LLM for a
literal, detail-dense description, then deleted.

#### Backend / connection

| Input | Type | Default | Notes |
|---|---|---|---|
| `backend` | combo | `ollama` | `ollama`, `lmstudio`, or `llamacpp`. |
| `host` | STRING | `http://localhost:11434` | Base URL of your LLM server. Change it to match the backend you selected (e.g. `http://localhost:1234` for LM Studio, `http://localhost:8080` for llama.cpp). |

#### Model selection (one per pass)

| Input | Type | Notes |
|---|---|---|
| `vision_model` | combo | Model used to describe the 1–9 reference images. |
| `story_model` | combo | Model used to write the story (Pass A). |
| `sequence_model` | combo | Model used **both** to break the story into sequences (Pass B) **and** to write each final Ref2VA prompt (Pass C). This mirrors the original app, which also uses a single "chat model" for both jobs. |

All three combos are populated dynamically from your Ollama server's
`/api/tags` at node-definition time. They're plain text-editable widgets, so
you can also just type a model name that isn't in the list (useful for
LM Studio/llama.cpp, whose model catalogs aren't auto-probed).

#### Story parameters

| Input | Type | Default | Notes |
|---|---|---|---|
| `premise` | STRING (multiline) | *(empty)* | Your story premise / synopsis seed. |
| `language` | combo | `English` | Output language for the story and prompts. |
| `word_count` | INT | `350` | Target length of the generated story. |

#### Sequencing parameters

| Input | Type | Default | Notes |
|---|---|---|---|
| `n_sequences` | INT | `10` | How many sequences (and therefore how many non-empty prompt outputs) to generate. Range 1–10. |
| `duration_per_sequence` | INT | `8` | Target duration in seconds for each sequence, written into both the breakdown instructions and each final prompt's `TARGET DURATION`. |
| `style` | combo | `Cinematic` | Visual style hint injected into every sequence brief (`Cinematic`, `live-action`, `2D-animated`, `3D CG`, `claymation`, `watercolor`, `vintage film`, or `auto`). |
| `camera_motions` | STRING (multiline) | full built-in list | Comma-separated vocabulary the sequence LLM is allowed to pick camera movements from (Zoom In, Pan Left, Tracking Shot, Static Shot, etc.). Edit to restrict or extend the vocabulary. |
| `extra_instructions` | STRING (multiline) | *(empty)* | Free-text extra direction injected into both the breakdown pass and every final prompt (e.g. "keep dialogue in French", "avoid slow-motion"). |

#### Audio

| Input | Type | Default | Notes |
|---|---|---|---|
| `video_music` | STRING | *(empty)* | A single music note repeated identically across every sequence's `non_diegetic_music` section, for score consistency across the whole video. |
| `no_video_music` | BOOLEAN | `False` | If enabled, forces `non_diegetic_music: N/A` in every sequence and overrides `video_music`. |

#### Sampling / reproducibility

| Input | Type | Default | Notes |
|---|---|---|---|
| `seed` | INT | `0` | Passed to every LLM call (story, breakdown, and each final prompt pass) as the sampling seed. Comes with ComfyUI's native **fixed / randomize / increment / decrement** control widget (the same one used on KSampler), so you get "randomized vs fixed" seed behavior for free. See [Reproducibility](#reproducibility-seed) below for backend support caveats. |
| `temperature_story` | FLOAT | `0.85` | Sampling temperature for the story pass. Higher = more creative/divergent. |
| `temperature_sequence` | FLOAT | `0.6` | Sampling temperature for the JSON sequence-breakdown pass. Kept moderate since this pass must return strict, parseable JSON. |
| `temperature_prompt` | FLOAT | `0.6` | Sampling temperature for the final Ref2VA prompt-writing pass. This pass must follow a strict 6-section format with fixed field names and vocabulary, so avoid pushing this much above ~0.7. |

### Outputs

| Output | Type | Notes |
|---|---|---|
| `story` | STRING | The full generated story text. Wire a **Show Text** / **Display Text** node here to read it. |
| `prompt_1` … `prompt_10` | STRING | One complete H3 Ref2VA prompt per sequence, in order. If `n_sequences < 10`, the unused trailing outputs (e.g. `prompt_7`…`prompt_10` when `n_sequences = 6`) are empty strings — this is expected, not a bug. |

Each non-empty `prompt_N` is a complete, self-contained 6-section H3 Ref2VA
prompt: `subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music` — ready to
feed straight into a Ref2VA generation call.

---

## How the pipeline works internally

```
image_1..9 ──► [vision_model] ──► per-image description ──► reference library
                                                                   │
premise, language, word_count ────────────────────────────────────┤
                                                                   ▼
                                                        [story_model]  (Pass A)
                                                                   │
                                                                   ▼
                                                              story text
                                                                   │
     n_sequences, camera_motions, duration, extra_instructions ───┤
                                                                   ▼
                                                     [sequence_model]  (Pass B)
                                              (returns strict JSON array,
                                               retried/repaired once if the
                                               model truncates or returns the
                                               wrong sequence count)
                                                                   │
                                                     N sequence dicts (JSON)
                                                                   │
                                          for each sequence:
                                          sequence_to_brief() → intermediate
                                          "STORYBOARD SCENE i/N" text brief
                                                                   │
                                                                   ▼
                                                     [sequence_model]  (Pass C)
                                              system prompt = REF2VA_SYSTEM_PROMPT
                                                                   │
                                                                   ▼
                                            final 6-section H3 Ref2VA prompt
                                                                   │
                                                                   ▼
                                                        routed to prompt_N
```

Three separate LLM **passes**, not three separate models — you can point
`story_model` and `sequence_model` at the same or different models depending
on what your hardware can run.

**Pass B robustness**: the sequence-breakdown pass strictly validates that
the LLM returned *exactly* `n_sequences` JSON objects with all required
fields. If a local model truncates its output or returns the wrong count,
the node automatically retries once with a shorter/stricter instruction
before failing loudly (rather than silently returning fewer prompts than
requested).

---

## Typical graph setup

```
Load Image ──► image_1 ─┐
Load Image ──► image_2 ─┤
Load Image ──► image_3 ─┤
Load Image ──► image_4 ─┼──► [H3 Story → Sequences] ──► story ──────► Show Text
     ...                │                              prompt_1 ────► Show Text / Save Text
Load Image ──► image_9 ─┘                              prompt_2 ────► Show Text / Save Text
                                                         ...
                                                         prompt_10 ───► Show Text / Save Text
```

Only connect as many `image_N` sockets as you have references (1 to 9);
leave the rest unconnected. Only as many `prompt_N` outputs as
`n_sequences` will contain text — the rest are empty strings, so it's safe
to leave all 10 wired up even if you're generating fewer sequences.

---

## Reproducibility (seed)

The `seed` widget behaves like ComfyUI's standard sampler seed:

- **fixed** — reuse the exact same seed on every run → identical output for
  identical inputs (deterministic story, sequences, and prompts).
- **randomize** — a fresh seed is picked automatically after each run.
- **increment / decrement** — step the seed by ±1 each run.

The same seed value is sent to **every** LLM call in the pipeline (story,
breakdown, and all final prompt-writing calls).

> ⚠️ **Backend support varies.** Ollama and llama.cpp honor the `seed`
> parameter for genuinely deterministic sampling. LM Studio forwards the
> field too, but determinism ultimately depends on its underlying runtime
> (llama.cpp-backed vs MLX-backed models don't guarantee identical output
> for the same seed). Treat `fixed` as "best effort reproducibility", not an
> absolute guarantee, especially on LM Studio/MLX.

---

## Backends

| Backend | Native API | Default host | Notes |
|---|---|---|---|
| `ollama` | `/api/tags`, `/api/chat` | `http://localhost:11434` | Model auto-discovery for the combos works out of the box. |
| `lmstudio` | OpenAI-compatible `/v1/models`, `/v1/chat/completions` | `http://localhost:1234` | Start the local server from LM Studio's *Developer* tab first. |
| `llamacpp` | OpenAI-compatible `/v1/models`, `/v1/chat/completions` | `http://localhost:8080` | Run `llama-server -m model.gguf --port 8080`. Older llama.cpp builds may not expose `/v1/models`; update if model discovery fails. |

Switch `backend` and set `host` to match; the node dispatches internally, no
other setting needs to change.

---

## Troubleshooting

**"Nothing shows up on my Show Text node."**
Make sure it's connected to `story` or to a `prompt_N` where `N ≤
n_sequences` — outputs beyond `n_sequences` are intentionally empty
strings. Also confirm the node actually ran without errors (check the
ComfyUI console).

**"I asked for N sequences but got fewer prompts."**
This should no longer happen — the node validates the sequence count
returned by the LLM and retries automatically. If it still fails after the
retry, you'll get an explicit error in the console (rather than a silently
truncated result); this usually means your local model's context/output
budget is too small for `n_sequences` at the given `word_count`. Try
lowering `word_count` or increasing the model's context window.

**"The prompt output doesn't look like the 6-section Ref2VA format."**
Make sure you're on the current version of this node — an earlier version
only exposed the intermediate scene brief (Pass B output) instead of
running it through Pass C. The final `prompt_N` should always start with
`subject_definitions:`.

**"Model dropdowns are empty / show a placeholder."**
The combos are populated by querying Ollama's `/api/tags` when the node
graph is loaded. Make sure Ollama (or your chosen backend) is running
*before* ComfyUI loads this node, then refresh the browser page. You can
also just type a model name directly into the combo field — it's an
editable widget, not a locked dropdown.

**"`LLMError: Impossible de contacter ... `"**
Your `host`/`backend` combination doesn't match a reachable server. Double
check the backend is running and the port matches the default (or your
custom) host for that backend.

---

## File layout

```
comfyui_h3_story2seq/
├── __init__.py            # registers NODE_CLASS_MAPPINGS for ComfyUI
├── nodes.py                # the H3StoryToSequences node (INPUT_TYPES + run())
├── llm_client.py            # local LLM HTTP client (Ollama / LM Studio / llama.cpp)
├── sequence_pipeline.py     # story generation, JSON sequence breakdown, brief builder
├── system_prompts.py        # REF2VA_SYSTEM_PROMPT (Pass C system prompt)
└── README.md                 # this file
```

---

## Credits

This node is a ComfyUI port of the **Story → Sequences** feature of *H3
Prompt Studio*, reusing its `llm_client.py` and `sequence_pipeline.py`
logic and its MiniMax H3 Ref2VA system prompt verbatim. All prompt-writing
rules, JSON schemas, and the reference-labelling convention
(`<Picture N>`, `<Subject N>`, etc.) originate from that project.
