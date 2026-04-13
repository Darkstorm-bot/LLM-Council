"""Mem0 OSS memory manager for local persistent long-term memory."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("council.mem0")


class Mem0MemoryManager:
    def __init__(
        self,
        enabled: bool = True,
        user_id: str = "local_user",
        agent_id: str = "llm_council",
        top_k: int = 6,
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "qwen35-9b",
        embedder_model: str = "nomic-embed-text",
    ) -> None:
        self.enabled = enabled
        self.user_id = user_id
        self.agent_id = agent_id
        self.top_k = top_k
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.embedder_model = embedder_model
        self._client = None
        self._init_error: str | None = None

        if self.enabled:
            self._client = self._init_client()

    def _init_client(self):
        try:
            from mem0 import Memory

            config = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": self.llm_model,
                        "temperature": 0.1,
                        "max_tokens": 1200,
                        "ollama_base_url": self.ollama_base_url,
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": self.embedder_model,
                        "ollama_base_url": self.ollama_base_url,
                    },
                },
                "version": "v1.0",
            }

            return Memory.from_config(config)
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("Mem0 unavailable; memory disabled: %s", exc)
            self.enabled = False
            return None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "top_k": self.top_k,
            "ollama_base_url": self.ollama_base_url,
            "llm_model": self.llm_model,
            "embedder_model": self.embedder_model,
            "init_error": self._init_error,
        }

    def search(self, query: str, *, user_id: str | None = None, top_k: int | None = None) -> list[str]:
        records = self.search_records(query, user_id=user_id, top_k=top_k)
        return [r.get("memory", "") for r in records if r.get("memory")]

    def search_records(
        self,
        query: str,
        *,
        user_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled or not self._client:
            return []

        uid = user_id or self.user_id
        limit = top_k or self.top_k

        try:
            raw = self._client.search(query, user_id=uid, agent_id=self.agent_id, limit=limit)
        except TypeError:
            raw = self._client.search(query, user_id=uid)
        except Exception as exc:
            logger.warning("Mem0 search failed: %s", exc)
            return []

        return self._normalize_records(raw)[:limit]

    def add(
        self,
        content: str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if not self.enabled or not self._client:
            return False

        uid = user_id or self.user_id
        meta = metadata or {}

        try:
            self._client.add(content, user_id=uid, agent_id=self.agent_id, metadata=meta)
            return True
        except TypeError:
            try:
                self._client.add(content, user_id=uid)
                return True
            except Exception as exc:
                logger.warning("Mem0 add fallback failed: %s", exc)
                return False
        except Exception as exc:
            logger.warning("Mem0 add failed: %s", exc)
            return False

    def add_session_memory(
        self,
        prompt: str,
        final_answer: str,
        *,
        research_summary: str = "",
        session_id: str = "",
    ) -> bool:
        if not self.enabled:
            return False

        text = (
            f"Session {session_id}\n"
            f"User task: {prompt}\n"
            f"Research summary: {research_summary[:1200]}\n"
            f"Council final answer: {final_answer[:2400]}"
        )
        return self.add(
            text,
            metadata={
                "kind": "council_session",
                "session_id": session_id,
            },
        )

    def delete(self, memory_id: str) -> bool:
        if not self.enabled or not self._client or not memory_id.strip():
            return False
        try:
            self._client.delete(memory_id=memory_id)
            return True
        except TypeError:
            try:
                self._client.delete(memory_id)
                return True
            except Exception as exc:
                logger.warning("Mem0 delete fallback failed: %s", exc)
                return False
        except Exception as exc:
            logger.warning("Mem0 delete failed: %s", exc)
            return False

    @staticmethod
    def _normalize_results(raw: Any) -> list[str]:
        records = Mem0MemoryManager._normalize_records(raw)
        return [r.get("memory", "") for r in records if r.get("memory")]

    @staticmethod
    def _normalize_records(raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []

        if isinstance(raw, dict):
            candidates = raw.get("results") or raw.get("memories") or []
        elif isinstance(raw, list):
            candidates = raw
        else:
            return [str(raw)]

        out: list[dict[str, Any]] = []
        for item in candidates:
            if isinstance(item, str):
                out.append({"id": "", "memory": item, "score": None})
                continue
            if not isinstance(item, dict):
                out.append({"id": "", "memory": str(item), "score": None})
                continue

            text = (
                item.get("memory")
                or item.get("content")
                or item.get("text")
                or item.get("value")
                or ""
            )
            out.append(
                {
                    "id": str(item.get("id", "")),
                    "memory": str(text),
                    "score": item.get("score"),
                }
            )

        return out
