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
    
    Enhanced to handle:
    - Thinking/reasoning text before JSON
    - Explanatory text after JSON
    - Multiple JSON blocks (takes the most complete one)
    - Unquoted keys, trailing commas
    - Markdown code fences with language hints
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"Invalid input: expected non-empty string, got {type(text).__name__}")
    
    original_text = text
    
    # Try to strip markdown fences completely
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    
    # Remove common prefixes that models add
    prefix_patterns = [
        r"^here\s*(is|'s)?\s*(my)?\s*(json|response|answer|vote)[:\s]*",
        r"^json\s*[:\s]*",
        r"^response\s*[:\s]*",
        r"^output\s*[:\s]*",
        r"^thinking\s*[:\s]*",
        r"^reasoning\s*[:\s]*",
    ]
    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    
    # Strategy 1: Try parsing the entire cleaned text directly
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find ALL JSON-like blocks and try each one
    # This handles cases where models output multiple attempts
    json_blocks = []
    
    # Find all {...} blocks with balanced braces
    depth = 0
    start_idx = -1
    for i, char in enumerate(cleaned):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx >= 0:
                json_blocks.append(cleaned[start_idx:i+1])
                start_idx = -1
    
    # Also find all [... ] blocks
    depth = 0
    start_idx = -1
    for i, char in enumerate(cleaned):
        if char == '[':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and start_idx >= 0:
                json_blocks.append(cleaned[start_idx:i+1])
                start_idx = -1
    
    # Try each block, preferring longer/more complete ones
    for block in sorted(json_blocks, key=len, reverse=True):
        # Try direct parse
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
        
        # Fix unquoted keys
        try:
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', block)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix trailing commas
        try:
            fixed = re.sub(r',(\s*[}\]])', r'\1', block)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix both unquoted keys AND trailing commas
        try:
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', block)
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Last resort - try to extract score-like values even from broken JSON
    # This is specifically helpful for vote parsing
    if '"score"' in cleaned.lower():
        # Try to salvage by finding score value even in malformed JSON
        score_match = re.search(r'"score"\s*[:=]\s*([0-9.]+)', cleaned)
        if score_match:
            try:
                score_val = float(score_match.group(1))
                # Return a minimal valid structure
                return {"score": min(1.0, max(0.0, score_val)), "partial": True}
            except (ValueError, TypeError):
                pass
    
    raise ValueError(
        f"Could not extract valid JSON from model output.\n"
        f"First 500 chars: {original_text[:500]}\n"
        f"Found {len(json_blocks)} potential JSON blocks, all invalid"
    )
