"""
Council Orchestrator — The conductor of the sequential LLM council pipeline.

Pipeline:
  Phase 0 → Research (lightweight model queries SearXNG)
  Phase 1 → Independent brainstorm (each council member, sequential)
  Phase 2 → Cross-examination & scoring
  Phase 3 → Synthesizer produces unified proposal
  Phase 4 → Council votes; loop back to Phase 3 if score < threshold
  Phase 5 → Present final answer
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Callable, Any

from core.memory_palace import MemoryPalace
from core.mem0_manager import Mem0MemoryManager
from core.model_interface import ModelConfig, OllamaClient, extract_json, ROLE_COUNCIL, ROLE_SYNTHESIZER, ROLE_RESEARCHER, ROLE_COMPRESSOR
from core.prompts import (
    RESEARCHER_SYSTEM, RESEARCHER_USER,
    COUNCIL_BRAINSTORM_SYSTEM, COUNCIL_BRAINSTORM_USER,
    COUNCIL_CRITIQUE_SYSTEM, COUNCIL_CRITIQUE_USER,
    SYNTHESIZER_SYSTEM, SYNTHESIZER_USER, SYNTHESIZER_ITERATION_FEEDBACK,
    COUNCIL_VOTE_SYSTEM, COUNCIL_VOTE_USER,
    COMPRESSOR_SYSTEM, COMPRESSOR_USER,
)
from tools.web_tools import ResearchAgent, SearXNG, BraveSearch, DuckDuckGoSearch, PlaywrightScraper, WebSearchProvider, IterativeResearchAgent
from tools.document_tools import DocumentIngestionEngine, ParsedDocument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("council.orchestrator")

CONFIG_PATH = Path(__file__).with_name("config.yaml")


# ════════════════════════════════════════════════════════════════════════════
#  Default council configuration — edit to match your pulled Ollama models
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_COUNCIL: list[ModelConfig] = [
    ModelConfig(
        model_id     = "qwen3",
        ollama_name  = "qwen3-14b",
        display_name = "Qwen-3 14B (Systematist)",
        role         = ROLE_COUNCIL,
        context_size = 8192,
        temperature  = 0.70,
        personality  = "systematic thinker who builds structured frameworks and reasons in explicit layers",
    ),
    ModelConfig(
        model_id     = "ministral",
        ollama_name  = "ministral-14b",
        display_name = "Ministral-3 14B (Challenger)",
        role         = ROLE_COUNCIL,
        context_size = 8192,
        temperature  = 0.80,
        personality  = "contrarian who aggressively challenges assumptions and looks for edge cases",
    ),
    ModelConfig(
        model_id     = "phi4",
        ollama_name  = "phi4-mini",
        display_name = "Phi-4 Mini (Pragmatist)",
        role         = ROLE_COUNCIL,
        context_size = 8192,
        temperature  = 0.62,
        personality  = "pragmatic implementer focused on feasible plans, execution risks, and concrete delivery",
    ),
    ModelConfig(
        model_id     = "future_you",
        ollama_name  = "qwen35-9b",
        display_name = "Future You (8-Month Consequence Seat)",
        role         = ROLE_COUNCIL,
        context_size = 8192,
        temperature  = 0.45,
        personality  = "you are this developer 8 months from now, living with consequences of today's decision; you focus on regret avoidance, maintenance burden, and long-term trade-offs for a developer under time pressure",
    ),
]

DEFAULT_SYNTHESIZER = ModelConfig(
    model_id     = "synthesizer",
    ollama_name  = "gemma4-26b",
    display_name = "Gemma-4 26B-A4B (Synthesizer)",
    role         = ROLE_SYNTHESIZER,
    context_size = 16384,
    temperature  = 0.65,
    personality  = "",
)

DEFAULT_RESEARCHER = ModelConfig(
    model_id     = "researcher",
    ollama_name  = "qwen35-9b",
    display_name = "Researcher",
    role         = ROLE_RESEARCHER,
    context_size = 4096,
    temperature  = 0.3,
    personality  = "",
)

DEFAULT_COMPRESSOR = ModelConfig(
    model_id     = "compressor",
    ollama_name  = "qwen35-9b",
    display_name = "Compressor",
    role         = ROLE_COMPRESSOR,
    context_size = 4096,
    temperature  = 0.2,
    personality  = "",
)


# ════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ════════════════════════════════════════════════════════════════════════════

class CouncilOrchestrator:
    def __init__(
        self,
        council:       list[ModelConfig]   = None,
        synthesizer:   ModelConfig         = None,
        researcher:    ModelConfig         = None,
        compressor:    ModelConfig         = None,
        ollama_url:    str                 = "http://localhost:11434",
        searxng_url:   str                 = "http://localhost:8080",
        search_provider: str              = "searxng",
        brave_api_key: str | None         = None,
        use_playwright: bool               = False,
        consensus_threshold: float        = 0.998,
        max_iterations: int               = 6,
        state_dir:     str                 = "./council_states",
        results_per_query: int            = 4,
        scrape_top_n: int                 = 2,
        project_root: Optional[str]       = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        memory_manager: Optional[Mem0MemoryManager] = None,
        document_ingestion_enabled: bool = False,
    ):
        self.council      = council      or DEFAULT_COUNCIL
        self.synthesizer  = synthesizer  or DEFAULT_SYNTHESIZER
        self.researcher   = researcher   or DEFAULT_RESEARCHER
        self.compressor   = compressor   or DEFAULT_COMPRESSOR

        self.client    = OllamaClient(ollama_url)
        searx = SearXNG(searxng_url)
        provider: WebSearchProvider
        provider_key = (search_provider or "searxng").strip().lower()
        if provider_key == "brave":
            provider = BraveSearch(brave_api_key or "")
        elif provider_key == "duckduckgo":
            provider = DuckDuckGoSearch()
        else:
            provider = searx

        self.research  = ResearchAgent(
            searxng    = searx,
            search_provider=provider,
            playwright = PlaywrightScraper() if use_playwright else None,
            use_playwright_fallback = use_playwright,
        )
        
        # Iterative deep-dive research agent (Phase 2 feature)
        self.iterative_research_agent = IterativeResearchAgent(
            base_research_agent=self.research,
            model_client=self.client,
            researcher_model=self.researcher,
            max_iterations=2,
            gap_threshold=0.6,
        )
        
        # Document ingestion engine (Phase 1 feature)
        self.document_ingestion_enabled = document_ingestion_enabled
        self.doc_engine = DocumentIngestionEngine() if document_ingestion_enabled else None
        
        self.threshold     = consensus_threshold
        self.max_iterations = max_iterations
        self.results_per_query = results_per_query
        self.scrape_top_n = scrape_top_n
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent
        self.progress_callback = progress_callback
        self.memory_manager = memory_manager
        self.state_dir      = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)

    def _emit_progress(self, stage: str, message: str, **extra: Any) -> None:
        if not self.progress_callback:
            return
        payload = {
            "stage": stage,
            "message": message,
            "timestamp": time.time(),
        }
        payload.update(extra)
        try:
            self.progress_callback(payload)
        except Exception:
            logger.debug("Progress callback failed", exc_info=True)

    def _extract_json_object(self, raw: str, phase: str) -> dict[str, Any]:
        """Parse model JSON and normalize object-like payloads.

        Some models occasionally emit a JSON array instead of an object. When that
        happens, accept the first object element to keep the pipeline moving.
        """
        data = extract_json(raw)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            logger.warning(f"   [{phase}] model returned JSON array; using first object item")
            return data[0]
        raise ValueError(f"[{phase}] Expected JSON object but got {type(data).__name__}")

    def _normalize_synthesizer_payload(self, data: dict[str, Any], raw: str) -> tuple[str, str]:
        proposal = data.get("proposal", raw)
        rationale = str(data.get("rationale", ""))

        if isinstance(proposal, str):
            p = proposal.strip()
            if p.startswith("{") and p.endswith("}"):
                try:
                    nested = json.loads(p)
                    if isinstance(nested, dict):
                        proposal = nested.get("proposal", proposal)
                        rationale = str(nested.get("rationale", rationale))
                except Exception:
                    pass

            # Recover proposal text from partial JSON wrappers when model outputs
            # a serialized object in the proposal field.
            p2 = str(proposal).strip()
            if '"proposal"' in p2 and not str(proposal).startswith("As "):
                try:
                    nested = extract_json(p2)
                    if isinstance(nested, dict):
                        proposal = nested.get("proposal", proposal)
                        rationale = str(nested.get("rationale", rationale))
                except Exception:
                    match = re.search(
                        r'"proposal"\s*:\s*"([\s\S]+?)"\s*,\s*"(?:how_it_merges_ideas|conflicts_resolved|key_improvements_over_last|rationale)"',
                        p2,
                    )
                    if match:
                        try:
                            proposal = bytes(match.group(1), "utf-8").decode("unicode_escape")
                        except Exception:
                            proposal = match.group(1)

        return str(proposal), rationale

    def _is_noisy_critique(self, critique: str) -> bool:
        c = critique.strip()
        if not c:
            return True
        if len(c) > 900:
            return True
        noisy_markers = ['"score"', 'Task:', 'Provide feedback', '"proposal"', '{', '}']
        marker_hits = sum(1 for m in noisy_markers if m in c)
        return marker_hits >= 3

    def _extract_score_from_text(self, text: str, default: float = 0.5) -> float:
        if not text:
            return default
        for pattern in [r"score\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", r"\b(0\.\d+|1\.0+|1|0)\b"]:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            try:
                return max(0.0, min(1.0, float(m.group(1))))
            except Exception:
                continue
        return default

    def _clean_model_text(self, text: str, max_len: int = 500) -> str:
        if not text:
            return ""
        s = str(text)
        s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\n{3,}", "\n\n", s)
        if len(s) > max_len:
            s = s[:max_len].rstrip() + "..."
        return s

    def _recover_vote_from_text(self, raw: str) -> tuple[float, str]:
        """
        Ultra-robust fallback for extracting score and critique from malformed vote responses.
        Uses multiple strategies in sequence to maximize extraction success.
        """
        if not raw or not isinstance(raw, str):
            return 0.5, "Empty or invalid response"
        
        text = raw.strip()
        
        # Strategy 1: Look for explicit score patterns with various formats
        score_patterns = [
            r'"score"\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)',  # "score": 0.75
            r'score\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)',     # score: 0.75
            r'\bscore\b\s*[:=]\s*(\d+(?:\.\d+)?)',          # score: 75 (will normalize)
            r'(?:rating|evaluation)\s*[:=]\s*(0?\.\d+|1\.0+|\d+)',  # rating: 0.8
        ]
        
        score = None
        for pattern in score_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    raw_score = float(match.group(1))
                    # Normalize if model gave percentage (e.g., 75 instead of 0.75)
                    if raw_score > 1.0:
                        raw_score = raw_score / 100.0
                    score = max(0.0, min(1.0, raw_score))
                    break
                except (ValueError, TypeError):
                    continue
        
        # Strategy 2: If no explicit score, look for standalone decimal numbers
        if score is None:
            decimal_matches = re.findall(r'\b(0\.\d{1,4}|1\.0{0,4})\b', text)
            if decimal_matches:
                # Take the first reasonable score-like number
                for match in decimal_matches:
                    try:
                        candidate = float(match)
                        if 0.0 <= candidate <= 1.0:
                            score = candidate
                            break
                    except ValueError:
                        continue
        
        # Strategy 3: Last resort - check for common score words
        if score is None:
            if any(word in text.lower() for word in ['perfect', 'excellent', 'flawless']):
                score = 1.0
            elif any(word in text.lower() for word in ['good', 'solid', 'acceptable']):
                score = 0.7
            elif any(word in text.lower() for word in ['poor', 'weak', 'flawed']):
                score = 0.3
            elif any(word in text.lower() for word in ['reject', 'terrible', 'failed']):
                score = 0.0
            else:
                score = 0.5  # Default neutral
        
        # Extract critique: find meaningful text that isn't JSON structure
        critique_lines = []
        in_json = False
        json_depth = 0
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            
            # Track JSON boundaries
            json_depth += line_stripped.count('{') - line_stripped.count('}')
            json_depth += line_stripped.count('[') - line_stripped.count(']')
            
            # Skip pure JSON lines
            if re.match(r'^[\s{},:\[\]"0-9.\-]+$', line_stripped):
                continue
            if re.match(r'^\s*"(score|verdict|remaining_issues|what_was_done_well|critique)"\s*:', line_stripped, flags=re.IGNORECASE):
                continue
            if line_stripped in ['{', '}', '[', ']', '']:
                continue
            
            # Keep substantive lines outside JSON
            if len(line_stripped) > 20 and not line_stripped.startswith('```'):
                critique_lines.append(line_stripped)
        
        # Build critique from extracted lines
        if critique_lines:
            critique = ' '.join(critique_lines[:10])  # Limit to first 10 lines
            if len(critique) > 500:
                critique = critique[:500].rstrip() + "..."
        else:
            # Fallback: extract the "critique" field value if present
            critique_match = re.search(r'"critique"\s*[:=]\s*"([^"]*(?:\\"[^"]*)*)"', text)
            if critique_match:
                critique = critique_match.group(1).replace('\\"', '"')
            else:
                critique = "Fallback: Could not extract structured critique from response."
        
        # Clean up critique
        critique = re.sub(r'\n{3,}', '\n\n', critique).strip()
        if not critique or len(critique) < 10:
            critique = "Model provided minimal feedback in fallback mode."
        
        return score, critique

    def _recover_critique_payload(self, raw: str, mp: MemoryPalace, model_id: str) -> tuple[dict[str, dict[str, Any]], str]:
        score = self._extract_score_from_text(raw, default=0.5)
        summary = self._clean_model_text(raw, max_len=360) or "Unstructured critique; fallback summary applied."
        evaluations: dict[str, dict[str, Any]] = {}
        for target_id in mp.council_ideas.keys():
            if target_id == model_id:
                continue
            evaluations[target_id] = {
                "strengths": [],
                "weaknesses": [],
                "blind_spots": "",
                "score": score,
                "critique_summary": summary,
            }
        return evaluations, "Fallback critique extraction used"

    def _compute_evidence_quality(self, support_results: list[dict], counter_results: list[dict]) -> tuple[float, str]:
        combined = [*support_results, *counter_results]
        if not combined:
            return 0.0, "No web evidence available"

        strong = 0
        for r in combined:
            url = str(r.get("url", ""))
            snippet = str(r.get("snippet", ""))
            source = str(r.get("source", "")).lower()
            host_score = 1 if any(t in url for t in [".edu", ".gov", ".dk", "uvm.dk", "studienet.dk", "astra.dk"]) else 0
            snippet_score = 1 if len(snippet) >= 120 else 0
            source_score = 1 if source in {"brave", "duckduckgo", "google", "startpage", "playwright"} else 0
            if host_score + snippet_score + source_score >= 2:
                strong += 1

        quality = strong / len(combined)
        return quality, f"{strong}/{len(combined)} evidence items passed quality gate"

    def _apply_consensus_penalty(self, scores: dict[str, float], avg_score: float) -> tuple[float, str]:
        if not scores:
            return avg_score, ""
        vals = list(scores.values())
        low_count = sum(1 for s in vals if s <= 0.55)
        high_count = sum(1 for s in vals if s >= 0.75)
        penalty = 0.0
        reason = ""
        if low_count >= 2 and high_count >= 1:
            penalty += 0.06
            reason = "multi-seat disagreement penalty"
        adjusted = max(0.0, min(1.0, avg_score - penalty))
        return adjusted, reason

    def _recompute_best_proposal(self, mp: MemoryPalace) -> None:
        if not mp.synthesizer_proposals:
            return
        best_idx = max(range(len(mp.synthesizer_proposals)), key=lambda i: float(mp.synthesizer_proposals[i].get("average_score", 0.0)))
        mp.best_proposal_index = best_idx
        mp.current_highest_score = float(mp.synthesizer_proposals[best_idx].get("average_score", 0.0))
        mp._touch()

    def _request_missing_critique(self, model: ModelConfig, prompt: str, proposal: str, score: float) -> str:
        sys_p = "You provide concise critical risk statements. Plain text only."
        user_p = (
            "Give 2-3 concrete risks for this proposal. "
            f"Original task: {prompt}\n"
            f"Model score: {score:.4f}\n"
            f"Proposal:\n{proposal[:1800]}"
        )
        try:
            critique = self.client.generate(model, sys_p, user_p, max_tokens=180).strip()
            return critique or "No concrete critique provided."
        except Exception:
            return "No concrete critique provided."

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, prompt: str) -> str:
        mp = MemoryPalace(original_prompt=prompt)
        logger.info(f"╔══ Council session: {mp.session_id} ══╗")
        logger.info(f"   Prompt: {prompt[:80]}...")
        self._emit_progress("session", "Session started", session_id=mp.session_id)

        # ── Mem0 long-term memory retrieval ────────────────────────────────
        if self.memory_manager and self.memory_manager.enabled:
            self._emit_progress("memory", "Retrieving long-term memories")
            memories = self.memory_manager.search(prompt)
            if memories:
                mp.add_long_term_memories(memories)
                self._emit_progress("memory", f"Loaded {len(memories)} memories")
            else:
                self._emit_progress("memory", "No relevant long-term memories found")

        # ── Project stack/context ingestion ─────────────────────────────────
        self._emit_progress("stack", "Inspecting local stack constraints")
        stack_context = self._build_stack_context(prompt)
        if stack_context:
            mp.set_stack_context(stack_context)
            self._emit_progress("stack", "Loaded local stack constraints")
        else:
            self._emit_progress("stack", "No stack files found; using prompt-only context")

        # ── Phase 0: Research ─────────────────────────────────────────────────
        self._phase_research(mp)
        
        # ── Document Ingestion (Phase 1 Feature) ────────────────────────────────
        if self.document_ingestion_enabled and self.doc_engine:
            self._emit_progress("documents", "Processing ingested documents")
            self._phase_document_ingestion(mp)
        
        self._save(mp)

        # ── Phase 1: Independent brainstorm ───────────────────────────────────
        self._phase_brainstorm(mp)
        self._save(mp)

        # ── Phase 2: Cross-examination ────────────────────────────────────────
        self._phase_critique(mp)
        self._save(mp)

        # ── Phase 3+4: Synthesis loop ─────────────────────────────────────────
        final_proposal = self._synthesis_loop(mp)
        self._save(mp)

        # ── Phase 5: Present ──────────────────────────────────────────────────
        result = self._format_final_answer(mp, final_proposal)
        mp.final_answer = result
        self._save(mp)

        # ── Mem0 session persistence ────────────────────────────────────────
        if self.memory_manager and self.memory_manager.enabled:
            self._emit_progress("memory", "Saving session memory")
            self.memory_manager.add_session_memory(
                prompt,
                result,
                research_summary=mp.research_summary,
                session_id=mp.session_id,
            )
            self._emit_progress("memory", "Session memory saved")

        self._emit_progress("session", "Session completed", session_id=mp.session_id)

        return result

    # ── Phase 0: Research ─────────────────────────────────────────────────────

    def _phase_research(self, mp: MemoryPalace) -> None:
        logger.info("\n── Phase 0: Research ──────────────────────────────────────")
        self._emit_progress("research", "Generating web queries")

        # Use researcher model to generate smart queries
        sys_p  = RESEARCHER_SYSTEM
        user_p = RESEARCHER_USER.format(
            prompt=mp.original_prompt,
            stack_context=mp.stack_context or "No detected constraints.",
        )
        raw    = self.client.generate(self.researcher, sys_p, user_p, max_tokens=300)

        try:
            data = self._extract_json_object(raw, "research")
            supporting_queries = data.get("supporting_queries", [])
            counter_queries = data.get("counter_queries", [])
            fallback = data.get("queries", [])
            if not supporting_queries:
                supporting_queries = fallback[:3] if fallback else [mp.original_prompt]
            if not counter_queries:
                counter_queries = [f"{mp.original_prompt} failure cases", f"{mp.original_prompt} risks"]
        except ValueError:
            supporting_queries = [mp.original_prompt]
            counter_queries = [f"{mp.original_prompt} failure cases"]

        logger.info(f"   Supporting queries: {supporting_queries}")
        logger.info(f"   Counter queries: {counter_queries}")
        self._emit_progress(
            "research",
            "Adversarial query sets generated",
            supporting_queries=supporting_queries,
            counter_queries=counter_queries,
        )

        # Use iterative research for deeper investigation
        logger.info("   Starting iterative deep-dive research...")
        self._emit_progress("research", "Iterative deep-dive research initiated")
        
        all_support_results = []
        all_counter_results = []
        support_metadata = {}
        counter_metadata = {}
        
        # Iterative research for supporting queries
        for idx, query in enumerate(supporting_queries):
            results, metadata = self.iterative_research_agent.iterative_research(
                initial_query=query,
                mp=mp,
                base_results_per_query=self.results_per_query,
                followup_results_per_query=3,
            )
            all_support_results.extend(results)
            if idx == 0:
                support_metadata = metadata
        
        # Iterative research for counter queries  
        for idx, query in enumerate(counter_queries):
            results, metadata = self.iterative_research_agent.iterative_research(
                initial_query=query,
                mp=mp,
                base_results_per_query=self.results_per_query,
                followup_results_per_query=3,
            )
            all_counter_results.extend(results)
            if idx == 0:
                counter_metadata = metadata
        
        # Deduplicate results by URL
        seen_urls = set()
        deduped_support = []
        for r in all_support_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped_support.append(r)
        
        seen_urls = set()
        deduped_counter = []
        for r in all_counter_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped_counter.append(r)
        
        mp.add_research(deduped_support, stance="support")
        mp.add_research(deduped_counter, stance="counter")
        quality_score, quality_note = self._compute_evidence_quality(deduped_support, deduped_counter)
        mp.evidence_quality_score = quality_score
        mp.evidence_quality_note = quality_note
        
        # Store iteration metadata
        mp.research_metadata = {
            "support": support_metadata,
            "counter": counter_metadata,
            "total_support_results": len(deduped_support),
            "total_counter_results": len(deduped_counter),
        }
        
        self._emit_progress(
            "research",
            f"Iterative research complete: {len(deduped_support)} supporting and {len(deduped_counter)} counter sources",
            support_count=len(deduped_support),
            counter_count=len(deduped_counter),
        )

        # Summarize supporting + counter evidence with researcher model
        if deduped_support or deduped_counter:
            support_snippets = "\n\n".join(
                f"[{r['title']}]\n{r['snippet']}" for r in deduped_support[:4]
            )
            counter_snippets = "\n\n".join(
                f"[{r['title']}]\n{r['snippet']}" for r in deduped_counter[:4]
            )
            summary_raw = self.client.generate(
                self.researcher,
                "You reconcile evidence for and against a proposal. Output two compact paragraphs: SUPPORTING then COUNTER-EVIDENCE.",
                (
                    f"TASK: {mp.original_prompt}\n\n"
                    f"SUPPORTING EVIDENCE:\n{support_snippets or 'None'}\n\n"
                    f"COUNTER-EVIDENCE:\n{counter_snippets or 'None'}"
                ),
                max_tokens=420,
            )
            support_summary = self.client.generate(
                self.researcher,
                "Summarize supporting evidence only into one dense paragraph. Plain text.",
                f"TASK: {mp.original_prompt}\n\nSUPPORTING EVIDENCE:\n{support_snippets or 'None'}",
                max_tokens=220,
            )
            mp.research_summary = support_summary.strip()
            mp.adversarial_summary = summary_raw.strip()
            logger.info(f"   Adversarial summary: {mp.adversarial_summary[:80]}...")
            self._emit_progress("research", "Adversarial evidence summary ready")
    
    # ── Document Ingestion Phase (Phase 1 Feature) ────────────────────────────
    
    def _phase_document_ingestion(self, mp: MemoryPalace) -> None:
        """
        Process pre-loaded documents from the document ingestion engine.
        Documents should be loaded via orchestrator.doc_engine.ingest() before run().
        """
        logger.info("\n── Document Ingestion ──────────────────────────────────────")
        self._emit_progress("documents", "Formatting ingested documents for context")
        
        if not self.doc_engine or not self.doc_engine.parsed_documents:
            logger.info("   No documents ingested; skipping document phase")
            return
        
        # Format all chunks for context injection
        formatted_context = self.doc_engine.format_chunks_for_context(
            max_chunks=15,  # Limit to avoid context explosion
            max_total_chars=10000
        )
        
        # Convert parsed documents to dict format for Memory Palace
        doc_dicts = []
        for doc in self.doc_engine.parsed_documents.values():
            doc_dicts.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "source_type": doc.source_type,
                "chunks": doc.chunks,
                "metadata": doc.metadata,
            })
        
        # Add to Memory Palace
        mp.add_ingested_documents(doc_dicts)
        
        logger.info(f"   Ingested {len(doc_dicts)} documents with {sum(len(d['chunks']) for d in doc_dicts)} total chunks")
        self._emit_progress(
            "documents",
            f"Loaded {len(doc_dicts)} documents into context",
            doc_count=len(doc_dicts),
            total_chunks=sum(len(d['chunks']) for d in doc_dicts),
        )

    # ── Phase 1: Brainstorm ───────────────────────────────────────────────────

    def _phase_brainstorm(self, mp: MemoryPalace) -> None:
        logger.info("\n── Phase 1: Independent Brainstorm ─────────────────────────")
        self._emit_progress("brainstorm", "Council brainstorming started")
        research_ctx = mp.build_research_context()

        for model in self.council:
            logger.info(f"   [{model.display_name}] generating idea...")
            sys_p = COUNCIL_BRAINSTORM_SYSTEM.format(
                name=model.display_name,
                personality=model.personality,
            )
            user_p = COUNCIL_BRAINSTORM_USER.format(
                prompt=mp.original_prompt,
                stack_context=mp.stack_context or "No detected constraints.",
                research_context=research_ctx,
            )
            raw = self.client.generate(model, sys_p, user_p, max_tokens=1024)
            try:
                data = self._extract_json_object(raw, f"brainstorm/{model.model_id}")
                idea      = data.get("idea", raw)
                reasoning = data.get("reasoning", "")
            except ValueError:
                logger.warning(f"   [{model.display_name}] JSON parse failed — using raw output")
                idea, reasoning = raw, ""

            mp.add_idea(model.model_id, model.display_name, idea, reasoning)
            logger.info(f"   [{model.display_name}] idea captured ({len(idea)} chars)")
            self._emit_progress("brainstorm", f"{model.display_name} submitted idea")

    # ── Phase 2: Cross-examination ────────────────────────────────────────────

    def _phase_critique(self, mp: MemoryPalace) -> None:
        logger.info("\n── Phase 2: Cross-Examination ──────────────────────────────")
        self._emit_progress("critique", "Cross-examination started")
        memory_ctx = mp.build_long_term_memory_context()

        for model in self.council:
            logger.info(f"   [{model.display_name}] critiquing others...")
            own_idea   = mp.council_ideas.get(model.model_id, {}).get("idea", "")
            others_ctx = mp.build_ideas_context(exclude_model_id=model.model_id)

            sys_p = COUNCIL_CRITIQUE_SYSTEM.format(
                name=model.display_name,
                personality=model.personality,
            )
            user_p = COUNCIL_CRITIQUE_USER.format(
                prompt=mp.original_prompt,
                stack_context=mp.stack_context or "No detected constraints.",
                own_idea=own_idea,
                others_ideas=others_ctx,
            )
            if memory_ctx:
                user_p += f"\n\n{memory_ctx}"
            raw = self.client.generate(model, sys_p, user_p, max_tokens=1200)
            try:
                data = self._extract_json_object(raw, f"critique/{model.model_id}")
                evaluations = data.get("evaluations", {})
                comment     = data.get("discussion_comment", "")
            except ValueError:
                logger.warning(f"   [{model.display_name}] critique JSON parse failed; applying fallback extraction")
                evaluations, comment = self._recover_critique_payload(raw, mp, model.model_id)

            if not isinstance(evaluations, dict):
                logger.warning(f"   [{model.display_name}] evaluations not an object; skipping")
                continue

            # Build score & critique dicts
            scores = {}
            critiques = {}
            for mid, ev in evaluations.items():
                if not isinstance(ev, dict):
                    continue
                try:
                    scores[mid] = float(ev.get("score", 0.5))
                except (TypeError, ValueError):
                    scores[mid] = 0.5
                critiques[mid] = str(ev.get("critique_summary", ""))
            mp.add_scores(model.model_id, model.display_name, scores, critiques)

            # Log discussion comment
            if comment:
                mp.add_discussion_entry(0, model.model_id, model.display_name, comment)
            self._emit_progress("critique", f"{model.display_name} submitted critiques")

        # Ensure the system always has usable critique priors even when some
        # models fail JSON formatting. This avoids misleading all-zero averages.
        for _, idea in mp.council_ideas.items():
            if idea.get("scores_received"):
                continue
            idea["scores_received"] = {"orchestrator_baseline": 0.5}
            idea["critiques_received"].append(
                {"from": "Orchestrator Baseline", "critique": "No structured critique returned; assigned neutral baseline."}
            )
            idea["average_score"] = 0.5
            mp.add_discussion_entry(
                0,
                "orchestrator",
                "Orchestrator",
                f"Applied neutral critique baseline for {idea['model_name']} due to missing structured critiques.",
            )
        mp._touch()

        # Log averaged scores
        for mid, idea in mp.council_ideas.items():
            logger.info(f"   Avg score [{idea['model_name']}]: {idea['average_score']:.4f}")
        self._emit_progress("critique", "Cross-examination completed")

    # ── Phase 3+4: Synthesis loop ─────────────────────────────────────────────

    def _synthesis_loop(self, mp: MemoryPalace) -> str:
        logger.info("\n── Phase 3+4: Synthesis Loop ───────────────────────────────")
        self._emit_progress("synthesis", "Synthesis loop started")
        best_proposal = ""
        consensus_hit = False
        consensus_mode = "max-iter-best-effort"

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n   ┌─ Iteration {iteration}/{self.max_iterations} ─────────────────────────")
            self._emit_progress("synthesis", f"Iteration {iteration}/{self.max_iterations} started", iteration=iteration)

            # ── Compress discussion if growing too large ───────────────────────
            if len(mp.discussion_log) > 15:
                self._compress_discussion(mp)

            # ── Synthesizer drafts a proposal ─────────────────────────────────
            prop_idx, proposal = self._run_synthesizer(mp, iteration)
            best_proposal = proposal

            # ── Council votes on the proposal ─────────────────────────────────
            avg_score = self._run_council_vote(mp, prop_idx, proposal, iteration)

            logger.info(f"   └─ Iteration {iteration} score: {avg_score:.4f}  (threshold: {self.threshold})")
            self._emit_progress("vote", f"Iteration {iteration} score: {avg_score:.4f}", iteration=iteration, score=avg_score)

            if avg_score >= self.threshold:
                logger.info(f"\n╔══ CONSENSUS REACHED at iteration {iteration} (score {avg_score:.4f}) ══╗")
                self._emit_progress("vote", f"Consensus reached at iteration {iteration}", iteration=iteration, score=avg_score)
                consensus_hit = True
                consensus_mode = "threshold"
                break

            if iteration == self.max_iterations:
                logger.info(
                    f"\n⚠  Max iterations reached. Returning best proposal "
                    f"(score {mp.current_highest_score:.4f})"
                )
                self._emit_progress("vote", "Max iterations reached; returning best proposal", score=mp.current_highest_score)
                best_proposal = mp.synthesizer_proposals[mp.best_proposal_index]["proposal"]

            mp.consensus_reached = consensus_hit
            mp.consensus_mode = consensus_mode
            mp._touch()

        return best_proposal

    def _run_synthesizer(self, mp: MemoryPalace, iteration: int) -> tuple[int, str]:
        logger.info(f"   [Synthesizer] drafting proposal #{iteration}...")
        self._emit_progress("synthesis", f"Synthesizer drafting proposal {iteration}")

        # Build iteration feedback (from previous round, if any)
        iteration_feedback = ""
        if mp.synthesizer_proposals:
            prev = mp.synthesizer_proposals[-1]
            critiques_text = "\n".join(
                f"  - {name}: {crit}"
                for name, crit in prev["council_critiques"].items()
            )
            iteration_feedback = SYNTHESIZER_ITERATION_FEEDBACK.format(
                prev_iter  = prev["iteration"],
                avg_score  = prev["average_score"],
                critiques  = critiques_text or "No specific critiques recorded.",
            )

        sys_p  = SYNTHESIZER_SYSTEM
        user_p = SYNTHESIZER_USER.format(
            full_context         = mp.build_full_context_for_synthesizer(),
            iteration            = iteration,
            iteration_feedback   = iteration_feedback,
        )

        raw = self.client.generate(
            self.synthesizer, sys_p, user_p, max_tokens=2048
        )
        try:
            data = self._extract_json_object(raw, "synthesizer")
            proposal, rationale = self._normalize_synthesizer_payload(data, raw)
        except ValueError:
            proposal, rationale = raw, ""

        prop_idx = mp.add_synthesizer_proposal(proposal, rationale)
        logger.info(f"   [Synthesizer] proposal captured ({len(proposal)} chars)")
        self._emit_progress("synthesis", f"Synthesizer proposal {iteration} captured")
        return prop_idx, proposal

    def _run_council_vote(
        self, mp: MemoryPalace, prop_idx: int, proposal: str, iteration: int
    ) -> float:
        memory_ctx = mp.build_long_term_memory_context()
        discussion_ctx = mp.build_discussion_context()
        if memory_ctx:
            discussion_ctx = f"{memory_ctx}\n\n{discussion_ctx}"

        for model in self.council:
            logger.info(f"   [{model.display_name}] voting on iteration {iteration}...")
            self._emit_progress("vote", f"{model.display_name} voting on iteration {iteration}")
            sys_p = COUNCIL_VOTE_SYSTEM.format(
                name=model.display_name,
                personality=model.personality,
            )
            user_p = COUNCIL_VOTE_USER.format(
                prompt=mp.original_prompt,
                stack_context=mp.stack_context or "No detected constraints.",
                iteration=iteration,
                proposal=proposal,
                discussion_context=discussion_ctx,
            )
            raw = self.client.generate(model, sys_p, user_p, max_tokens=600)
            try:
                data = self._extract_json_object(raw, f"vote/{model.model_id}")
                score    = float(data.get("score", 0.5))
                critique = str(data.get("critique", "")).strip()
                # Clamp score to valid range
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                logger.warning(f"   [{model.display_name}] vote parse failed; applying fallback extraction")
                score, critique = self._recover_vote_from_text(raw)

            if self._is_noisy_critique(critique):
                critique = self._request_missing_critique(model, mp.original_prompt, proposal, score)

            critique = critique.strip()
            if len(critique) > 500:
                critique = critique[:500].rstrip() + "..."

            mp.add_vote_on_proposal(prop_idx, model.model_id, model.display_name, score, critique)
            mp.add_discussion_entry(
                iteration, model.model_id, model.display_name,
                f"Iteration {iteration} vote: {score:.4f}. {critique}"
            )
            logger.info(f"   [{model.display_name}] score: {score:.4f}")
            self._emit_progress("vote", f"{model.display_name} score: {score:.4f}", model=model.model_id, score=score)

        proposal_entry = mp.synthesizer_proposals[prop_idx]
        raw_avg = float(proposal_entry.get("average_score", 0.0))
        adjusted_avg, penalty_reason = self._apply_consensus_penalty(proposal_entry.get("council_scores", {}), raw_avg)
        proposal_entry["average_score"] = adjusted_avg
        if penalty_reason:
            mp.add_discussion_entry(
                iteration,
                "orchestrator",
                "Orchestrator",
                f"Applied {penalty_reason}: {raw_avg:.4f} -> {adjusted_avg:.4f}",
            )
        self._recompute_best_proposal(mp)
        avg = adjusted_avg
        return avg

    # ── Discussion compressor ─────────────────────────────────────────────────

    def _compress_discussion(self, mp: MemoryPalace) -> None:
        logger.info("   [Compressor] summarizing discussion log...")
        log_text = "\n".join(
            f"[{e['speaker_name']}] {e['content']}" for e in mp.discussion_log
        )
        summary = self.client.generate(
            self.compressor,
            COMPRESSOR_SYSTEM,
            COMPRESSOR_USER.format(discussion_log=log_text),
            max_tokens=400,
        )
        mp.compress_discussion(summary.strip())
        logger.info("   [Compressor] discussion compressed")

    # ── Final formatting ──────────────────────────────────────────────────────

    def _format_final_answer(self, mp: MemoryPalace, proposal: str) -> str:
        best = mp.synthesizer_proposals[mp.best_proposal_index]
        dissent_line = "No dissent available."
        if best.get("council_scores"):
            dissent_id = min(best["council_scores"], key=best["council_scores"].get)
            dissent_score = float(best["council_scores"].get(dissent_id, 0.0))
            dissent_name = mp.council_ideas.get(dissent_id, {}).get("model_name", dissent_id)
            dissent_critique = best.get("council_critiques", {}).get(dissent_id, "No critique was provided.")
            dissent_line = (
                f"{dissent_name} ({dissent_score:.4f})\n"
                f"{dissent_critique}"
            )

        lines = [
            "═" * 70,
            f"  COUNCIL FINAL ANSWER  (session: {mp.session_id})",
            f"  Consensus score: {best['average_score']:.4f}  |  "
            f"Iterations: {mp.current_iteration}",
            f"  Consensus mode: {mp.consensus_mode}  |  "
            f"Evidence quality: {mp.evidence_quality_score:.2f}",
            "═" * 70,
            "",
            "THE CASE AGAINST THIS DECISION (Minority Report):",
            dissent_line,
            "",
            "─" * 70,
            "",
            proposal,
            "",
            "─" * 70,
            "Council member final scores:",
        ]
        for voter, score in best["council_scores"].items():
            lines.append(f"  {voter}: {score:.4f}")
        lines += [
            "─" * 70,
            f"Original prompt: {mp.original_prompt}",
            "═" * 70,
        ]
        return "\n".join(lines)

    def _build_stack_context(self, prompt: str) -> str:
        stack_files = [
            "package.json",
            "requirements.txt",
            "pyproject.toml",
            "Pipfile",
            "poetry.lock",
            "go.mod",
            "Cargo.toml",
            "pom.xml",
            "build.gradle",
            "Dockerfile",
        ]
        sections = [f"Workspace: {self.project_root}"]

        for name in stack_files:
            path = self.project_root / name
            if not path.exists() or not path.is_file():
                continue
            content = self._summarize_stack_file(path)
            if content:
                sections.append(f"[{name}]\n{content}")

        if len(sections) == 1:
            return ""
        return "\n\n".join(sections)

    def _summarize_stack_file(self, path: Path) -> str:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

        name = path.name.lower()
        if name == "requirements.txt":
            packages = []
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                packages.append(line)
                if len(packages) >= 30:
                    break
            return "Pinned dependencies:\n" + "\n".join(f"- {p}" for p in packages)

        if name == "package.json":
            try:
                data = json.loads(raw)
                deps = data.get("dependencies", {})
                dev_deps = data.get("devDependencies", {})
                rows = []
                for dep_name, version in list(deps.items())[:20]:
                    rows.append(f"- {dep_name}: {version}")
                for dep_name, version in list(dev_deps.items())[:20]:
                    rows.append(f"- (dev) {dep_name}: {version}")
                return "Node dependencies:\n" + "\n".join(rows)
            except Exception:
                pass

        lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
        return "\n".join(lines[:60])

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, mp: MemoryPalace) -> None:
        path = self.state_dir / f"{mp.session_id}.json"
        mp.save(path)
        logger.debug(f"   State saved → {path}")


def _load_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; skipping config.yaml load")
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw or {}


def _build_model_config(data: dict, role: str, fallback: ModelConfig) -> ModelConfig:
    return ModelConfig(
        model_id=data.get("model_id", fallback.model_id),
        ollama_name=data.get("ollama_name", fallback.ollama_name),
        display_name=data.get("display_name", fallback.display_name),
        role=role,
        context_size=int(data.get("context_size", fallback.context_size)),
        temperature=float(data.get("temperature", fallback.temperature)),
        personality=data.get("personality", fallback.personality),
    )


def build_orchestrator_from_config(
    config_path: str | Path = CONFIG_PATH,
    overrides: dict | None = None,
) -> "CouncilOrchestrator":
    cfg = _load_yaml_config(Path(config_path))
    overrides = overrides or {}

    council_cfg = cfg.get("council", [])
    if council_cfg:
        council = [
            _build_model_config(item, ROLE_COUNCIL, DEFAULT_COUNCIL[idx % len(DEFAULT_COUNCIL)])
            for idx, item in enumerate(council_cfg)
        ]
    else:
        council = DEFAULT_COUNCIL

    ensure_future_you_seat = bool(cfg.get("council_features", {}).get("enable_future_you_seat", True))
    if ensure_future_you_seat and not any(m.model_id == "future_you" for m in council):
        council = [*council, DEFAULT_COUNCIL[-1]]

    synthesizer = _build_model_config(
        cfg.get("synthesizer", {}),
        ROLE_SYNTHESIZER,
        DEFAULT_SYNTHESIZER,
    )
    researcher = _build_model_config(
        cfg.get("researcher", {}),
        ROLE_RESEARCHER,
        DEFAULT_RESEARCHER,
    )
    compressor = _build_model_config(
        cfg.get("compressor", {}),
        ROLE_COMPRESSOR,
        DEFAULT_COMPRESSOR,
    )

    ollama_url = overrides.get("ollama_url") or cfg.get("ollama", {}).get("base_url", "http://localhost:11434")
    searxng_url = overrides.get("searxng_url") or cfg.get("searxng", {}).get("base_url", "http://localhost:8080")
    search_provider = overrides.get("search_provider") or cfg.get("search", {}).get("provider", "searxng")
    brave_api_key = overrides.get("search_brave_api_key") or cfg.get("search", {}).get("brave_api_key", "")
    use_playwright = bool(
        overrides.get("use_playwright")
        if overrides.get("use_playwright") is not None
        else cfg.get("searxng", {}).get("use_playwright", False)
    )
    threshold = float(overrides.get("consensus_threshold") or cfg.get("consensus", {}).get("threshold", 0.998))
    max_iterations = int(overrides.get("max_iterations") or cfg.get("consensus", {}).get("max_iterations", 6))
    state_dir = overrides.get("state_dir") or cfg.get("state", {}).get("dir", "./council_states")
    results_per_query = int(cfg.get("searxng", {}).get("results_per_query", 4))
    scrape_top_n = int(cfg.get("searxng", {}).get("scrape_top_n", 2))
    progress_callback = overrides.get("progress_callback")

    memory_cfg = cfg.get("memory", {})
    memory_enabled = bool(
        overrides.get("memory_enabled")
        if overrides.get("memory_enabled") is not None
        else memory_cfg.get("enabled", False)
    )
    memory_user_id = overrides.get("memory_user_id") or memory_cfg.get("user_id", "local_user")
    memory_agent_id = overrides.get("memory_agent_id") or memory_cfg.get("agent_id", "llm_council")
    memory_top_k = int(overrides.get("memory_top_k") or memory_cfg.get("top_k", 6))
    memory_ollama_url = overrides.get("memory_ollama_url") or memory_cfg.get("ollama_base_url", ollama_url)
    memory_llm_model = overrides.get("memory_llm_model") or memory_cfg.get("llm_model", researcher.ollama_name)
    memory_embedder_model = overrides.get("memory_embedder_model") or memory_cfg.get("embedder_model", "nomic-embed-text")

    memory_manager = Mem0MemoryManager(
        enabled=memory_enabled,
        user_id=memory_user_id,
        agent_id=memory_agent_id,
        top_k=memory_top_k,
        ollama_base_url=memory_ollama_url,
        llm_model=memory_llm_model,
        embedder_model=memory_embedder_model,
    )
    
    # Document ingestion config (Phase 1 feature)
    doc_ingestion_enabled = bool(
        overrides.get("document_ingestion_enabled")
        if overrides.get("document_ingestion_enabled") is not None
        else cfg.get("features", {}).get("document_ingestion_enabled", False)
    )

    return CouncilOrchestrator(
        council=council,
        synthesizer=synthesizer,
        researcher=researcher,
        compressor=compressor,
        ollama_url=ollama_url,
        searxng_url=searxng_url,
        search_provider=search_provider,
        brave_api_key=brave_api_key,
        use_playwright=use_playwright,
        consensus_threshold=threshold,
        max_iterations=max_iterations,
        state_dir=state_dir,
        results_per_query=results_per_query,
        scrape_top_n=scrape_top_n,
        project_root=str(Path(config_path).resolve().parent),
        progress_callback=progress_callback,
        memory_manager=memory_manager,
        document_ingestion_enabled=doc_ingestion_enabled,
    )


# ════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM Council — sequential MoA pipeline")
    parser.add_argument("prompt", nargs="?", help="The question or task for the council")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument("--ollama",    default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--searxng",   default="http://localhost:8080",   help="SearXNG URL")
    parser.add_argument("--search-provider", choices=["searxng", "brave", "duckduckgo"], default="searxng", help="Web search provider")
    parser.add_argument("--brave-api-key", default="", help="Brave Search API key (required if --search-provider brave)")
    parser.add_argument("--playwright", action="store_true",               help="Enable Playwright scraping")
    parser.add_argument("--threshold", type=float, default=0.998,          help="Consensus threshold (default 0.998)")
    parser.add_argument("--max-iter",  type=int,   default=6,              help="Max synthesis iterations")
    parser.add_argument("--state-dir", default="./council_states",         help="Directory for session state files")
    parser.add_argument("--memory-enabled", action="store_true",           help="Force-enable Mem0 local memory")
    parser.add_argument("--memory-disabled", action="store_true",          help="Disable Mem0 local memory")
    parser.add_argument("--memory-user", default=None,                      help="Mem0 user_id scope")
    parser.add_argument("--memory-agent", default=None,                     help="Mem0 agent_id scope")
    parser.add_argument("--memory-top-k", type=int, default=None,           help="Mem0 retrieval depth")
    parser.add_argument("--memory-llm", default=None,                       help="Mem0 local LLM model (Ollama)")
    parser.add_argument("--memory-embedder", default=None,                  help="Mem0 local embedding model (Ollama)")
    parser.add_argument("--documents", nargs="*", default=[],                help="Document paths/URLs to ingest (PDF, DOCX, TXT, MD, GitHub URLs)")
    parser.add_argument("--document-ingestion", action="store_true",         help="Enable document ingestion feature")
    args = parser.parse_args()

    memory_enabled_override = None
    if args.memory_enabled:
        memory_enabled_override = True
    if args.memory_disabled:
        memory_enabled_override = False

    if not args.prompt:
        args.prompt = input("Enter your question or task:\n> ").strip()

    orchestrator = build_orchestrator_from_config(
        config_path=args.config,
        overrides={
            "ollama_url": args.ollama,
            "searxng_url": args.searxng,
            "search_provider": args.search_provider,
            "search_brave_api_key": args.brave_api_key,
            "use_playwright": args.playwright,
            "consensus_threshold": args.threshold,
            "max_iterations": args.max_iter,
            "state_dir": args.state_dir,
            "memory_enabled": memory_enabled_override,
            "memory_user_id": args.memory_user,
            "memory_agent_id": args.memory_agent,
            "memory_top_k": args.memory_top_k,
            "memory_llm_model": args.memory_llm,
            "memory_embedder_model": args.memory_embedder,
            "document_ingestion_enabled": args.document_ingestion or bool(args.documents),
        },
    )
    
    # Pre-load documents if provided
    if args.documents and orchestrator.doc_engine:
        logger.info(f"Loading {len(args.documents)} documents...")
        for doc_source in args.documents:
            try:
                orchestrator.doc_engine.ingest(doc_source)
                logger.info(f"  ✓ Loaded: {doc_source}")
            except Exception as e:
                logger.error(f"  ✗ Failed to load {doc_source}: {e}")

    print("\n" + "═" * 70)
    print("  LLM COUNCIL — Starting session")
    print("═" * 70 + "\n")

    result = orchestrator.run(args.prompt)
    print("\n" + result)


if __name__ == "__main__":
    main()
