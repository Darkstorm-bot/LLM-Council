"""
Memory Palace — The Central State Brain of the LLM Council.
All models are stateless; this object IS their shared memory.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class CouncilIdea:
    model_id: str
    model_name: str
    idea: str
    reasoning: str
    timestamp: float = field(default_factory=time.time)
    scores_received: dict = field(default_factory=dict)   # {voter_id: score}
    critiques_received: list = field(default_factory=list)
    average_score: float = 0.0


@dataclass
class SynthesizerProposal:
    iteration: int
    proposal: str
    rationale: str
    council_scores: dict = field(default_factory=dict)   # {model_id: score}
    council_critiques: dict = field(default_factory=dict)
    average_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DiscussionEntry:
    round: int
    speaker_id: str
    speaker_name: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryPalace:
    # ── Core prompt ─────────────────────────────────────────────────────────
    original_prompt: str = ""
    session_id: str = field(default_factory=lambda: f"council_{int(time.time())}")

    # ── Research ────────────────────────────────────────────────────────────
    long_term_memories: list = field(default_factory=list)
    web_research: list = field(default_factory=list)
    supporting_research: list = field(default_factory=list)
    counter_research: list = field(default_factory=list)
    research_summary: str = ""
    adversarial_summary: str = ""
    stack_context: str = ""
    research_metadata: dict = field(default_factory=dict)  # Phase 2: Iterative research metadata
    
    # ── Document Ingestion (Phase 1 Feature) ────────────────────────────────
    ingested_documents: list = field(default_factory=list)  # [{doc_id, title, source, chunks, ...}]
    document_context: str = ""

    # ── Council state ────────────────────────────────────────────────────────
    council_ideas: dict = field(default_factory=dict)       # {model_id: CouncilIdea}
    discussion_log: list = field(default_factory=list)
    discussion_summary: str = ""                            # compressed after each iter

    # ── Synthesizer iterations ───────────────────────────────────────────────
    synthesizer_proposals: list = field(default_factory=list)
    current_iteration: int = 0
    current_highest_score: float = 0.0
    best_proposal_index: int = -1

    # ── Control ──────────────────────────────────────────────────────────────
    consensus_reached: bool = False
    consensus_mode: str = "not-run"  # threshold | max-iter-best-effort | not-run
    evidence_quality_score: float = 0.0
    evidence_quality_note: str = ""
    final_answer: str = ""

    # ── Metadata ─────────────────────────────────────────────────────────────
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # ────────────────────────────────────────────────────────────────────────
    #  Serialization
    # ────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "MemoryPalace":
        raw = json.loads(Path(path).read_text())
        mp = cls()
        for k, v in raw.items():
            setattr(mp, k, v)
        return mp

    # ────────────────────────────────────────────────────────────────────────
    #  Prompt-injection helpers (what each model actually reads)
    # ────────────────────────────────────────────────────────────────────────

    def build_research_context(self) -> str:
        parts = []
        if self.stack_context:
            parts.append(self.build_stack_context())

        if self.long_term_memories:
            parts.append(self.build_long_term_memory_context())

        # Add ingested documents context (Phase 1 feature)
        if self.document_context:
            parts.append(self.document_context)

        if not self.web_research and not self.ingested_documents:
            parts.append("No web research or documents available.")
            return "\n\n".join(parts)

        if self.supporting_research:
            parts.append("=== SUPPORTING EVIDENCE ===")
            for r in self.supporting_research[:6]:
                parts.append(f"[{r.get('source', 'unknown')}] {r.get('title', '')}\n{r.get('snippet', '')}")

        if self.counter_research:
            parts.append("=== COUNTER-EVIDENCE & FAILURE CASES ===")
            for r in self.counter_research[:6]:
                parts.append(f"[{r.get('source', 'unknown')}] {r.get('title', '')}\n{r.get('snippet', '')}")

        if not self.supporting_research and not self.counter_research and not self.document_context:
            parts.append("=== WEB RESEARCH ===")
            for r in self.web_research[:6]:   # cap to avoid context blowout
                parts.append(f"[{r.get('source', 'unknown')}] {r.get('title', '')}\n{r.get('snippet', '')}")

        if self.research_summary:
            parts.append(f"\nSUMMARY: {self.research_summary}")
        if self.adversarial_summary:
            parts.append(f"\nADVERSARIAL FINDINGS: {self.adversarial_summary}")
        return "\n\n".join(parts)

    def build_stack_context(self) -> str:
        if not self.stack_context:
            return ""
        return f"=== STACK CONSTRAINTS ===\n{self.stack_context}"

    def build_long_term_memory_context(self) -> str:
        if not self.long_term_memories:
            return ""
        parts = ["=== LONG-TERM MEMORY (MEM0) ==="]
        for idx, mem in enumerate(self.long_term_memories[:8], start=1):
            parts.append(f"{idx}. {mem}")
        return "\n".join(parts)

    def build_ideas_context(self, exclude_model_id: Optional[str] = None) -> str:
        if not self.council_ideas:
            return "No ideas submitted yet."
        parts = ["=== COUNCIL IDEAS ==="]
        for mid, idea in self.council_ideas.items():
            if mid == exclude_model_id:
                continue
            parts.append(
                f"[{idea['model_name']}]\n"
                f"IDEA: {idea['idea']}\n"
                f"REASONING: {idea['reasoning']}"
            )
        return "\n\n".join(parts)

    def build_discussion_context(self, max_entries: int = 20) -> str:
        if not self.discussion_log:
            return "No discussion yet."
        recent = self.discussion_log[-max_entries:]
        parts = ["=== DISCUSSION LOG ==="]
        if self.discussion_summary:
            parts.append(f"[SUMMARY OF EARLIER ROUNDS]\n{self.discussion_summary}")
        for e in recent:
            parts.append(f"[{e['speaker_name']}] {e['content']}")
        return "\n\n".join(parts)

    def build_synthesizer_history(self) -> str:
        if not self.synthesizer_proposals:
            return ""
        parts = ["=== PREVIOUS PROPOSALS ==="]
        for p in self.synthesizer_proposals[-3:]:   # last 3 only
            parts.append(
                f"[Iteration {p['iteration']} — avg score {p['average_score']:.4f}]\n"
                f"PROPOSAL: {p['proposal'][:800]}...\n"
                f"CRITIQUES: {json.dumps(p['council_critiques'], indent=2)[:600]}"
            )
        return "\n\n".join(parts)

    def build_full_context_for_synthesizer(self) -> str:
        sections = [
            f"ORIGINAL PROMPT:\n{self.original_prompt}",
            self.build_stack_context(),
            self.build_long_term_memory_context(),
            self.build_research_context(),
            self.build_ideas_context(),
            self.build_discussion_context(),
            self.build_synthesizer_history(),
        ]
        return "\n\n" + "\n\n".join(s for s in sections if s)

    # ────────────────────────────────────────────────────────────────────────
    #  Mutators
    # ────────────────────────────────────────────────────────────────────────

    def add_research(self, results: list[dict], stance: str = "neutral") -> None:
        tagged = []
        for r in results:
            entry = dict(r)
            entry["stance"] = stance
            tagged.append(entry)

        self.web_research.extend(tagged)
        if stance == "support":
            self.supporting_research.extend(tagged)
        elif stance == "counter":
            self.counter_research.extend(tagged)
        self._touch()
    
    def add_ingested_documents(self, documents: list[dict]) -> None:
        """
        Add ingested documents from DocumentIngestionEngine.
        
        Args:
            documents: List of document dicts with keys: doc_id, title, source, chunks, metadata
        """
        for doc in documents:
            self.ingested_documents.append({
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "source_type": doc.get("source_type", ""),
                "chunk_count": len(doc.get("chunks", [])),
                "metadata": doc.get("metadata", {}),
            })
        
        # Build formatted context string from all chunks
        from tools.document_tools import DocumentIngestionEngine
        engine = DocumentIngestionEngine()
        # Note: In production, we'd pass actual parsed documents here
        # For now, we'll build context from the stored document info
        context_parts = [f"=== INGESTED DOCUMENTS ({len(documents)} docs) ===\n"]
        
        for doc in documents:
            chunks = doc.get("chunks", [])
            if chunks:
                context_parts.append(f"\n--- {doc.get('title', 'Untitled')} [{doc.get('source_type', 'unknown')}] ---")
                for chunk in chunks[:5]:  # Limit chunks per doc to avoid context explosion
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    citation = f"[{doc.get('title', 'Unknown')}]"
                    if hasattr(chunk, 'to_citation'):
                        citation = chunk.to_citation()
                    context_parts.append(f"{citation}\n{content[:1000]}...")
        
        self.document_context = "\n\n".join(context_parts)
        self._touch()

    def set_stack_context(self, context: str) -> None:
        self.stack_context = context.strip()
        self._touch()

    def add_long_term_memories(self, memories: list[str]) -> None:
        self.long_term_memories = memories[:12]
        self._touch()

    def add_idea(self, model_id: str, model_name: str, idea: str, reasoning: str) -> None:
        self.council_ideas[model_id] = {
            "model_id": model_id,
            "model_name": model_name,
            "idea": idea,
            "reasoning": reasoning,
            "scores_received": {},
            "critiques_received": [],
            "average_score": 0.0,
            "timestamp": time.time(),
        }
        self._touch()

    def add_scores(self, voter_id: str, voter_name: str, scores: dict[str, float],
                   critiques: dict[str, str]) -> None:
        """voter submits scores & critiques for all other ideas."""
        for target_id, score in scores.items():
            if target_id in self.council_ideas:
                self.council_ideas[target_id]["scores_received"][voter_id] = score
                critique = critiques.get(target_id, "")
                if critique:
                    self.council_ideas[target_id]["critiques_received"].append(
                        {"from": voter_name, "critique": critique}
                    )
        # recompute averages
        for idea in self.council_ideas.values():
            sc = list(idea["scores_received"].values())
            idea["average_score"] = sum(sc) / len(sc) if sc else 0.0
        self._touch()

    def add_discussion_entry(self, round_: int, speaker_id: str,
                             speaker_name: str, content: str) -> None:
        self.discussion_log.append({
            "round": round_,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "content": content,
            "timestamp": time.time(),
        })
        self._touch()

    def add_synthesizer_proposal(self, proposal: str, rationale: str) -> int:
        self.current_iteration += 1
        entry = {
            "iteration": self.current_iteration,
            "proposal": proposal,
            "rationale": rationale,
            "council_scores": {},
            "council_critiques": {},
            "average_score": 0.0,
            "timestamp": time.time(),
        }
        self.synthesizer_proposals.append(entry)
        self._touch()
        return self.current_iteration - 1   # index

    def add_vote_on_proposal(self, proposal_index: int, voter_id: str,
                             voter_name: str, score: float, critique: str) -> None:
        p = self.synthesizer_proposals[proposal_index]
        p["council_scores"][voter_id] = score
        p["council_critiques"][voter_id] = critique
        # recompute
        scores = list(p["council_scores"].values())
        p["average_score"] = sum(scores) / len(scores) if scores else 0.0
        if p["average_score"] > self.current_highest_score:
            self.current_highest_score = p["average_score"]
            self.best_proposal_index = proposal_index
        self._touch()

    def compress_discussion(self, summary: str) -> None:
        """Replace old log entries with a summary to save context space."""
        self.discussion_summary = summary
        # keep only the last 5 entries verbatim
        self.discussion_log = self.discussion_log[-5:]
        self._touch()

    def _touch(self) -> None:
        self.last_updated = time.time()
