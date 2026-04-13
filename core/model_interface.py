"""
Model Interface — Handles sequential loading/unloading via Ollama.
Each call loads the model, runs inference, then immediately frees VRAM.
"""

import json
import re
import time
import logging
from typing import Optional
from dataclasses import dataclass

import requests

logger = logging.getLogger("council.model")

# ─── Model Roles ─────────────────────────────────────────────────────────────

ROLE_RESEARCHER  = "researcher"    # lightweight, fast — queries + summarizes
ROLE_COUNCIL     = "council"       # brainstorm, critique, vote
ROLE_SYNTHESIZER = "synthesizer"   # large model — unifies proposals
ROLE_COMPRESSOR  = "compressor"    # lightweight — summarizes discussion logs


@dataclass
class ModelConfig:
    model_id: str          # internal name  e.g. "llama3"
    ollama_name: str       # ollama pull name  e.g. "llama3:8b-instruct-q5_K_M"
    display_name: str      # human label  e.g. "Llama-3 8B"
    role: str
    context_size: int = 8192
    temperature: float = 0.7
    personality: str = ""  # injected into system prompt for council diversity


# ─── Ollama Client ────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base = base_url.rstrip("/")

    # ── Low-level call ────────────────────────────────────────────────────────

    def generate(
        self,
        model: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        timeout: int = 300,
    ) -> str:
        """
        Calls Ollama /api/chat.
        Sets keep_alive=0 so the model is unloaded from VRAM after the response.
        """
        payload = {
            "model": model.ollama_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "options": {
                "temperature":  model.temperature,
                "num_predict":  max_tokens,
                "num_ctx":      model.context_size,
            },
            "keep_alive": 0,    # ← KEY: unload immediately after response
            "stream": False,
        }

        logger.info(f"  ↳ Calling [{model.display_name}] ({model.ollama_name})")
        t0 = time.time()

        try:
            resp = requests.post(
                f"{self.base}/api/chat",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["message"]["content"].strip()
            elapsed = time.time() - t0
            logger.info(f"  ↳ [{model.display_name}] finished in {elapsed:.1f}s")
            return text

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama. Is it running? → `ollama serve`"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"[{model.display_name}] timed out after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"[{model.display_name}] error: {e}")

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def is_available(self, model: ModelConfig) -> bool:
        try:
            return model.ollama_name in self.list_models()
        except Exception:
            return False


# ─── JSON extraction helper ───────────────────────────────────────────────────

def extract_json(text: str) -> dict | list:
    """
    Robustly extracts the first JSON object or array from model output.
    Handles markdown fences, stray text, and common formatting issues.
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"Invalid input: expected non-empty string, got {type(text).__name__}")
    
    # Try to strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    
    # Remove any leading/trailing whitespace and common prefixes
    cleaned = re.sub(r"^\s*(?:json\s*)?", "", cleaned, flags=re.IGNORECASE).strip()
    
    # Remove any trailing explanatory text after the JSON block
    # This handles cases where models add text after closing brace/bracket
    json_end_match = re.search(r"(?:\}|\])\s*$", cleaned)
    if json_end_match:
        # Find the last } or ] and truncate there
        last_brace = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if last_brace > 0:
            cleaned = cleaned[:last_brace + 1]

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } or [ ... ] block with balanced braces/brackets
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = cleaned.find(start_char)
        if start_idx == -1:
            continue
        
        # Count braces to find matching end
        depth = 0
        end_idx = -1
        for i in range(start_idx, len(cleaned)):
            if cleaned[i] == start_char:
                depth += 1
            elif cleaned[i] == end_char:
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        
        if end_idx > start_idx:
            candidate = cleaned[start_idx:end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    # Fix missing quotes around keys
                    fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', candidate)
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    continue

    raise ValueError(f"Could not extract JSON from model output:\n{text[:500]}")
