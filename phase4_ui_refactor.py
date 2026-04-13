"""Phase 4: UI Refactors - Enhanced GUI for Advanced Research Features

This file implements the Phase 4 UI enhancements for:
1. Multi-Modal Document Ingestion (file uploads, GitHub URLs)
2. Iterative Deep-Dive Research (depth controls, iteration logs)
3. Cross-Source Fact Verification (verification reports, contradiction displays)

Changes to gui.py:
- New accordion for document uploads
- Research settings with depth slider and fact-check toggle
- Additional tabs for citations, verification, and research trail
- Enhanced progress callbacks for new phases
"""

import gradio as gr
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 ADDITIONS - Insert these into gui.py
# ─────────────────────────────────────────────────────────────────────────────

def get_phase4_ui_additions():
    """
    Returns the code snippets to add to gui.py for Phase 4.
    These should be inserted at the appropriate locations in build_app().
    """
    
    # 1. DOCUMENT UPLOAD SECTION (add after "Runtime Controls" accordion)
    document_upload_section = '''
                with gr.Accordion("📎 Document Ingestion", open=False):
                    file_upload = gr.File(
                        label="Upload Documents (PDF, DOCX, TXT, MD)",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt", ".md"],
                    )
                    github_url_input = gr.Textbox(
                        label="GitHub Repository URL",
                        placeholder="https://github.com/username/repo",
                    )
                    web_url_input = gr.Textbox(
                        label="Web Page URLs (one per line)",
                        placeholder="https://example.com/article\\nhttps://arxiv.org/abs/...",
                        lines=3,
                    )
                    ingest_btn = gr.Button("Ingest Documents", variant="secondary")
                    ingestion_status = gr.Markdown("_No documents ingested yet._")
    '''
    
    # 2. RESEARCH SETTINGS SECTION (add after Runtime Controls or Model Stack)
    research_settings_section = '''
                with gr.Accordion("🔬 Research Settings", open=False):
                    research_depth = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        label="Research Iteration Depth",
                        info="Higher values = more thorough but slower research"
                    )
                    enable_fact_check = gr.Checkbox(
                        label="Enable Cross-Source Fact Verification",
                        value=True,
                        info="Automatically detect contradictions and verify claims"
                    )
                    enable_deep_research = gr.Checkbox(
                        label="Enable Iterative Deep-Dive Research",
                        value=True,
                        info="Automatically identify and fill knowledge gaps"
                    )
    '''
    
    # 3. ENHANCED TABS (replace existing Tabs section)
    enhanced_tabs_section = '''
                with gr.Tabs():

                    with gr.Tab("Verdict"):
                        decision_signals = gr.HTML(
                            _placeholder("Decision signals appear after a completed run."),
                            elem_classes=["scroll-pane"],
                        )
                        final_answer = gr.Markdown(
                            "_No council output yet._",
                            elem_classes=["scroll-pane"],
                        )

                    with gr.Tab("Dissent"):
                        minority_report = gr.Markdown(
                            "No minority report yet.",
                            elem_classes=["scroll-pane"],
                        )

                    with gr.Tab("Evidence"):
                        adversarial_evidence = gr.HTML(
                            _placeholder("No adversarial evidence available yet."),
                            elem_classes=["scroll-pane"],
                        )

                    with gr.Tab("📚 Citations"):
                        citations_display = gr.Markdown(
                            "_No citations available yet._",
                            elem_classes=["scroll-pane"],
                        )

                    with gr.Tab("✅ Fact Verification"):
                        verification_report = gr.Markdown(
                            "_No fact verification report yet._",
                            elem_classes=["scroll-pane"],
                        )
                        contradictions_display = gr.JSON(
                            label="Detected Contradictions",
                            visible=True,
                        )

                    with gr.Tab("🔍 Research Trail"):
                        research_iterations_log = gr.Textbox(
                            label="Research Iteration Log",
                            lines=20,
                            elem_classes=["scroll-pane", "mono-pane"],
                        )

                    with gr.Tab("Stream"):
                        live_trace = gr.Textbox(
                            label="Phase-by-phase progress",
                            lines=20,
                            elem_classes=["scroll-pane", "mono-pane"],
                        )

                    with gr.Tab("Metrics"):
                        snapshot_md = gr.Markdown(
                            "Run a session to populate metrics.",
                            elem_classes=["scroll-pane"],
                        )

                    with gr.Tab("Timeline"):
                        timeline_html = gr.HTML(
                            _placeholder("Timeline will appear after a run."),
                            elem_classes=["scroll-pane"],
                        )

                    with gr.Tab("Memory"):
                        mem_query = gr.Textbox(
                            label="Search memories",
                            placeholder="What does this user prefer?",
                        )
                        with gr.Row():
                            mem_search_btn = gr.Button("Search", elem_classes=[])
                            mem_add_btn = gr.Button("Add Entry", elem_classes=[])
                            mem_delete_btn = gr.Button("Delete by ID", elem_classes=[])
                        mem_add_text = gr.Textbox(
                            label="Memory text to add",
                            lines=3,
                        )
                        mem_delete_id = gr.Textbox(label="Memory ID to delete")
                        mem_output = gr.Textbox(
                            label="Memory operation output",
                            lines=14,
                            elem_classes=["scroll-pane", "mono-pane"],
                        )

                    with gr.Tab("State JSON"):
                        state_json = gr.Code(
                            language="json",
                            label="Session State",
                            elem_classes=["scroll-pane", "mono-pane"],
                        )
    '''
    
    return {
        "document_upload": document_upload_section,
        "research_settings": research_settings_section,
        "enhanced_tabs": enhanced_tabs_section,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED run_council_stream FUNCTION SIGNATURE
# ─────────────────────────────────────────────────────────────────────────────

def get_updated_run_council_stream_signature():
    """
    Updated function signature for run_council_stream to accept new parameters.
    """
    return '''
def run_council_stream(
    prompt: str,
    ollama_url: str,
    searxng_url: str,
    search_provider: str,
    search_brave_api_key: str,
    threshold: float,
    max_iter: int,
    use_playwright: bool,
    state_dir: str,
    memory_enabled: bool,
    memory_user_id: str,
    memory_agent_id: str,
    memory_top_k: int,
    memory_ollama_url: str,
    memory_llm_model: str,
    memory_embedder_model: str,
    # NEW PHASE 4 PARAMETERS:
    research_depth: int,
    enable_fact_check: bool,
    enable_deep_research: bool,
    uploaded_files: list[gr.FileData] | None = None,
    github_urls: str = "",
    web_urls: str = "",
):
'''


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED EVENT WIRING
# ─────────────────────────────────────────────────────────────────────────────

def get_updated_event_wiring():
    """
    Updated .click() event wiring for run_btn with new inputs/outputs.
    """
    return '''
        run_btn.click(
            fn=run_council_stream,
            inputs=[
                prompt,
                ollama_url,
                searxng_url,
                search_provider,
                search_brave_api_key,
                threshold,
                max_iter,
                use_playwright,
                state_dir,
                memory_enabled,
                memory_user_id,
                memory_agent_id,
                memory_top_k,
                memory_ollama_url,
                memory_llm_model,
                memory_embedder_model,
                # NEW PHASE 4 INPUTS:
                research_depth,
                enable_fact_check,
                enable_deep_research,
                file_upload,
                github_url_input,
                web_url_input,
            ],
            outputs=[
                final_answer,
                snapshot_md,
                state_json,
                session_file,
                live_trace,
                timeline_html,
                adversarial_evidence,
                minority_report,
                decision_signals,
                # NEW PHASE 4 OUTPUTS:
                citations_display,
                verification_report,
                contradictions_display,
                research_iterations_log,
                ingestion_status,
            ],
        )
'''


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS CALLBACK EXTENSIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_progress_callback_examples():
    """
    Examples of new progress events for Phase 4 features.
    Add these to orchestrator.py _emit_progress calls.
    """
    return '''
# Document Ingestion Progress
self._emit_progress("document", f"Ingesting PDF: {filename}", pages=num_pages)
self._emit_progress("document", f"Parsed {len(chunks)} chunks from {source}")
self._emit_progress("document", f"GitHub repo cloned: {repo_name} ({file_count} files)")

# Deep Research Progress
self._emit_progress("deep_research", f"Iteration {i}/{max_depth}: Analyzing knowledge gaps")
self._emit_progress("deep_research", f"Identified {len(gaps)} knowledge gaps")
self._emit_progress("deep_research", f"Follow-up query: {query}")
self._emit_progress("deep_research", f"Coverage score: {score:.2f}")

# Fact Verification Progress
self._emit_progress("fact_check", f"Extracted {num_claims} claims for verification")
self._emit_progress("fact_check", f"Verifying claim: '{claim_text[:50]}...'")
self._emit_progress("fact_check", f"Confidence: {confidence:.2f} | Verdict: {verdict}")
self._emit_progress("contradiction", f"Conflict detected: {source_a} vs {source_b}")
'''


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS FOR NEW DISPLAYS
# ─────────────────────────────────────────────────────────────────────────────

def format_citations_display(state_json: dict) -> str:
    """Format citations from state into Markdown."""
    citations = state_json.get("citations", [])
    if not citations:
        return "_No citations available yet._"
    
    md_lines = ["### 📚 Source Citations\\n"]
    for i, cite in enumerate(citations, 1):
        source_type = cite.get("type", "unknown")
        title = cite.get("title", "Untitled")
        url = cite.get("url", "")
        page = cite.get("page", "")
        
        icon = {"pdf": "📄", "web": "🌐", "github": "💻", "docx": "📝"}.get(source_type, "📎")
        md_lines.append(f"{icon} **[{i}]** {title}")
        if url:
            md_lines.append(f"   - Source: [{url}]({url})")
        if page:
            md_lines.append(f"   - Page: {page}")
        md_lines.append("")
    
    return "\\n".join(md_lines)


def format_verification_report(state_json: dict) -> str:
    """Format fact verification results into Markdown."""
    verified_claims = state_json.get("verified_claims", [])
    if not verified_claims:
        return "_No fact verification report yet._"
    
    md_lines = ["### ✅ Fact Verification Report\\n"]
    
    verified_count = sum(1 for c in verified_claims if c.get("verdict") == "verified")
    contradicted_count = sum(1 for c in verified_claims if c.get("verdict") == "contradicted")
    unresolved_count = sum(1 for c in verified_claims if c.get("verdict") == "unresolved")
    
    md_lines.append(f"**Summary:** {verified_count} verified | {contradicted_count} contradicted | {unresolved_count} unresolved\\n")
    
    for claim in verified_claims:
        verdict = claim.get("verdict", "unknown")
        confidence = claim.get("confidence", 0.0)
        text = claim.get("claim", "")
        
        emoji = {"verified": "✅", "contradicted": "❌", "unresolved": "⚠️"}.get(verdict, "❓")
        md_lines.append(f"{emoji} **{text[:100]}...**")
        md_lines.append(f"   - Confidence: {confidence:.2f}")
        md_lines.append(f"   - Verdict: {verdict.upper()}")
        if claim.get("reasoning"):
            md_lines.append(f"   - Reasoning: {claim['reasoning'][:150]}...")
        md_lines.append("")
    
    return "\\n".join(md_lines)


def format_research_trail(state_json: dict) -> str:
    """Format research iteration log."""
    iterations = state_json.get("research_iterations", [])
    if not iterations:
        return "No research iterations logged yet."
    
    lines = ["### 🔍 Research Trail\\n"]
    for i, iter_data in enumerate(iterations, 1):
        lines.append(f"**Iteration {i}:**")
        lines.append(f"- Queries: {len(iter_data.get('queries', []))}")
        lines.append(f"- Results found: {len(iter_data.get('results', []))}")
        lines.append(f"- Knowledge gaps identified: {len(iter_data.get('gaps', []))}")
        lines.append(f"- Coverage score: {iter_data.get('coverage_score', 'N/A')}")
        lines.append("")
    
    return "\\n".join(lines)


if __name__ == "__main__":
    print("Phase 4 UI Refactor Guide")
    print("=" * 60)
    print("\nThis file contains code snippets to enhance gui.py with:")
    print("1. Document upload section (PDF, DOCX, GitHub, Web URLs)")
    print("2. Research settings (depth slider, fact-check toggle)")
    print("3. Enhanced tabs (Citations, Fact Verification, Research Trail)")
    print("4. Updated function signatures and event wiring")
    print("\nSee functions above for specific code to insert.")
