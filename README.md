# LLM Council — Sequential Mixture-of-Agents with Memory Palace

A production-grade, VRAM-efficient multi-model reasoning system.
Models load and unload sequentially — you never need more VRAM than
a single model requires. All shared memory lives in the **Memory Palace**.

```
┌────────────────────────────────────────────────────────────┐
│                      Memory Palace (JSON)                  │
│  original_prompt · research · ideas · discussion · votes   │
└───────────────────────┬────────────────────────────────────┘
                        │  injected into every model's context
        ┌───────────────┼──────────────────────────┐
        ▼               ▼                          ▼
  [Researcher]   [Council × N]            [Synthesizer 27B]
  query SearXNG  brainstorm / critique     unify all ideas
  (unloads)      (unloads after each)      (unloads)
                        │                          │
                        └────────── vote ──────────┘
                              loop until ≥ 0.998
```

---

## Prerequisites

| Service | Purpose | Install |
|---------|---------|---------|
| **Ollama** | Load/unload models | [ollama.com](https://ollama.com) |
| **Search provider** | Web research | SearXNG (self-hosted), Brave API, or DuckDuckGo |
| **Playwright** *(optional)* | JS-page scraping | `pip install playwright && playwright install chromium` |

### Use LM Studio GGUF files in Ollama

This repository assumes runtime through **Ollama model aliases**.
If you use LM Studio GGUF files locally, users of this repo still need Ollama aliases
that map to local models (or pulled Ollama models) to run the council.

```bash
# Create aliases that point at local GGUF files via Modelfiles in this repo.
ollama create gemma4-26b    -f Modelfile_gemma
ollama create qwen3-14b     -f Modelfile_qwen3
ollama create ministral-14b -f Modelfile_ministral
ollama create phi4-mini     -f Modelfile_phi4
ollama create qwen35-9b     -f Modelfile_qwen35

# Mem0 local embedder model (required for fully local persistent memory)
ollama pull nomic-embed-text
```

The default stack is:
- Synthesizer: `gemma4-26b`
- Council: `qwen3-14b`, `ministral-14b`, `phi4-mini`
- Researcher + Compressor: `qwen35-9b`

---

## Setup

```bash
git clone <repo>
cd llm_council
pip install -r requirements.txt
```

### One-command installer / environment check

Use the installer to validate dependencies, set search provider, and optionally create/pull Ollama models:

```bash
python install.py
```

Examples:

```bash
# DuckDuckGo (no key)
python install.py --provider duckduckgo

# Brave API
python install.py --provider brave --brave-api-key <YOUR_KEY>

# Skip model alias creation
python install.py --skip-model-setup
```

Mem0 local memory is included via `mem0ai` and enabled by default in `config.yaml`.
This project configures Mem0 to use Ollama for both extraction LLM and embeddings,
so no OpenAI key is required when local models are available.
Use `memory.user_id` to scope persistent memories per user/persona.

---

## Usage

### GUI (recommended)

```bash
python gui.py
```

Then open `http://127.0.0.1:7860` in your browser.

GUI highlights:
- Live phase trace (Research, Brainstorm, Critique, Synthesis, Vote)
- Consensus timeline chart across synthesizer iterations
- Final answer, snapshot metrics, and full Memory Palace JSON
- Saved-session browser for replay and inspection
- Memory Inspector tab for Mem0 search/add/delete operations
- Search provider selector in runtime controls (`searxng`, `brave`, `duckduckgo`)

Differentiators now built-in:
- Adversarial evidence retrieval: separate supporting vs counter-evidence web passes
- Minority report output: final answer always prints "The case against this decision"
- Stack-grounded debate: council reads local stack constraints from files like `requirements.txt`/`package.json`
- Future You council seat: a permanent long-term-consequences voice in every debate

### Interactive CLI

```bash
python orchestrator.py
# Prompts for input

python orchestrator.py "What is the most robust architecture for a real-time ML pipeline?"
```

### With all options

```bash
python orchestrator.py "your question" \
  --ollama    http://localhost:11434 \
  --searxng   http://localhost:8080  \
  --playwright                        \   # enable JS scraping
  --threshold 0.995                   \   # lower = faster, less strict
  --max-iter  5                       \   # safety cap
  --state-dir ./sessions              \   # state snapshots
  --memory-enabled                    \   # force-enable Mem0
  --memory-user local_user            \   # memory scope
  --memory-top-k 8
```

---

## Pipeline Walkthrough

### Phase 0 — Research
A lightweight model generates optimal SearXNG queries from your prompt.
Results (+ optional Playwright-scraped full text) are stored in `web_research`.

### Phase 1 — Independent Brainstorm
Each council member loads, reads the prompt + research, and proposes a solution.
Every model has a distinct **personality** (pragmatist / challenger / systematist)
to guarantee genuine intellectual diversity. Each model **unloads** after responding.

### Phase 2 — Cross-Examination
Each council member loads, reads all other ideas, and submits:
- A 4-decimal-place score (0.0000–1.0000) per idea
- A structured critique (strengths, weaknesses, blind spots)

When Mem0 is enabled, long-term memory context is injected into critique and vote phases
in addition to brainstorm/synthesis, so persistent preferences influence the full council loop.

The orchestrator averages scores across voters. Models unload after each turn.

### Phase 3 — Synthesis
The large (27B) Synthesizer loads, ingests the **entire Memory Palace**
(prompt + research + ideas + critiques + scores + previous iteration feedback),
and produces a unified proposal that resolves conflicts and merges the best elements.

### Phase 4 — Consensus Vote → Loop
All council members vote on the Synthesizer's proposal.
- **Score ≥ 0.998** → consensus reached, present final answer
- **Score < 0.998** → critiques are fed back to the Synthesizer (Phase 3 repeats)
- **Max iterations hit** → return the highest-scoring proposal seen

### Context Compression
When the discussion log grows beyond 15 entries, a lightweight compressor
summarizes it into a dense paragraph, preventing context blowout in the Synthesizer.

---

## Customising Council Members

Edit `config.yaml` or modify `DEFAULT_COUNCIL` in `orchestrator.py`.

Core model classes now live under `core/` and research utilities under `tools/`, so imports are:

```python
from core.model_interface import ModelConfig
from tools.web_tools import ResearchAgent
```

Example model entry:

```python
ModelConfig(
    model_id     = "phi3",
    ollama_name  = "phi3:14b-medium-4k-instruct-q5_K_M",
    display_name = "Phi-3 (Ethicist)",
    role         = ROLE_COUNCIL,
    context_size = 4096,
    temperature  = 0.72,
    personality  = "ethicist who evaluates every proposal through the lens of fairness and unintended consequences",
)
```

---

## State Files

Every session writes a JSON snapshot of the full Memory Palace to `./council_states/`.
These are human-readable and can be inspected, resumed, or replayed.

## GitHub Upload Notes

- Personal session data in `council_states/*.json` is ignored by default.
- The repository keeps `council_states/.gitkeep` so the folder exists after clone.
- Do not commit API keys in `config.yaml` (for Brave use local/private values only).

## Release Checklist

Before publishing:

1. Validate runtime and imports:

```bash
python -m py_compile gui.py orchestrator.py tools/web_tools.py install.py
python -c "from gui import build_app; build_app(); print('ui_ok')"
```

2. Validate environment setup path:

```bash
python install.py --provider searxng --skip-model-setup --skip-requirements
```

3. Confirm no personal session data is staged:

```bash
git status
```

4. Confirm search provider configuration for public defaults:
- `search.provider: searxng` or `duckduckgo`
- `search.brave_api_key: ""`

5. Commit and push:

```bash
git add .
git commit -m "Prepare repository for GitHub release"
git push
```

---

## Tuning Tips

| Goal | Change |
|------|--------|
| Faster runs | Lower `--threshold` to 0.990, reduce `--max-iter` |
| Richer debate | Add a 4th council member with a unique personality |
| Better synthesis | Upgrade synthesizer to `deepseek-r1:32b` or `command-r:35b` |
| Less VRAM | Swap all models to Q4 quants |
| More web depth | Set `use_playwright: true` and `scrape_top_n: 4` |

---

## Architecture Notes

- **`keep_alive: 0`** in every Ollama call ensures models are unloaded immediately,
  freeing VRAM for the next model in the sequence.
- The **Memory Palace** is a plain Python dataclass serialised to JSON — no vector DB,
  no embedding store. Context is injected via string-building methods.
- **JSON-only model outputs** with a robust `extract_json()` fallback means the
  pipeline survives occasional malformed responses without crashing.
- Runtime settings are loaded from `config.yaml` through `build_orchestrator_from_config()`,
  used by both `orchestrator.py` (CLI) and `gui.py` (web UI).

---

## Acknowledgements

This project builds on the following open-source tools and libraries:

- [Ollama](https://github.com/ollama/ollama) — local model runtime and model management.
- [Gradio](https://github.com/gradio-app/gradio) — interactive web UI for council control and replay.
- [Mem0](https://github.com/mem0ai/mem0) via `mem0ai` — persistent local long-term memory support.
- [Playwright](https://github.com/microsoft/playwright) — optional browser-based page enrichment.
- [SearXNG](https://github.com/searxng/searxng) — self-hosted metasearch provider option.
- [Requests](https://github.com/psf/requests) — HTTP client for provider and service checks.
- [PyYAML](https://github.com/yaml/pyyaml) — configuration parsing for runtime settings.

And thanks to the maintainers and communities behind:

- [DuckDuckGo](https://duckduckgo.com/) — no-key search provider option.
- [Brave Search API](https://brave.com/search/api/) — optional API-based search provider.
