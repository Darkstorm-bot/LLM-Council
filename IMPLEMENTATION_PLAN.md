# Implementation Plan: Advanced Research Features

## Executive Summary
This plan outlines the implementation of three game-changing features to transform the LLM Council into a premier autonomous research system:
1. **Multi-Modal Document Ingestion** - PDF, academic papers, GitHub repos
2. **Iterative Deep-Dive Research Agent** - Recursive knowledge gap filling
3. **Cross-Source Fact Verification** - Contradiction detection & confidence scoring

---

## Phase 1: Multi-Modal Document Ingestion

### 1.1 New Dependencies
Add to `requirements.txt`:
```
pypdf2>=3.0.0
pdfplumber>=0.10.0
python-docx>=0.8.11
beautifulsoup4>=4.12.0
lxml>=5.0.0
requests>=2.31.0
aiohttp>=3.9.0
```

### 1.2 New File: `tools/document_ingestion.py`
**Purpose**: Unified document parser with citation tracking

**Key Classes**:
```python
class DocumentParser:
    - parse_pdf(path) -> ParsedDocument
    - parse_docx(path) -> ParsedDocument
    - parse_github_repo(url) -> ParsedDocument
    - parse_webpage(url) -> ParsedDocument
    
class ParsedDocument:
    - content: str
    - metadata: dict (title, authors, date, source_type)
    - citations: list[Citation]
    - chunks: list[TextChunk]  # for RAG
    
class Citation:
    - text: str
    - page_number: int
    - confidence: float
    - source_url: str
```

**Features**:
- PDF text extraction with layout preservation
- Academic paper metadata extraction (DOI, authors, abstract)
- GitHub repo cloning + file filtering (.py, .md, .txt, .yaml)
- Automatic chunking for long documents (500-800 tokens)
- Citation anchor detection (e.g., "[1]", "(Smith et al., 2023)")

### 1.3 New File: `tools/github_ingestion.py`
**Purpose**: Smart GitHub repository analysis

**Key Functions**:
```python
def clone_repo_safely(url: str, max_depth: int = 2) -> Path
def filter_important_files(repo_path: Path) -> list[Path]
    - Prioritize: README, docs/, src/, *.py, *.md
    - Exclude: node_modules, __pycache__, .git, binaries
def extract_code_context(file_path: Path) -> str
    - Extract docstrings, function signatures, class definitions
```

### 1.4 Memory Palace Updates
Extend `MemoryPalace` class in `core/memory_palace.py`:
```python
@dataclass
class MemoryPalace:
    # Add new fields:
    uploaded_documents: list[ParsedDocument] = field(default_factory=list)
    document_chunks: list[TextChunk] = field(default_factory=list)
    citation_index: dict[str, Citation] = field(default_factory=dict)
    
    def add_document(self, doc: ParsedDocument) -> None
    def build_document_context(self) -> str
    def search_document_chunks(self, query: str, top_k: int = 5) -> list[TextChunk]
```

### 1.5 Orchestrator Integration
Modify `orchestrator.py`:
- Add `_phase_document_ingestion()` before Phase 0
- Accept file uploads via GUI
- Inject document context into research phase

---

## Phase 2: Iterative Deep-Dive Research Agent

### 2.1 New File: `tools/deep_research_agent.py`
**Purpose**: Recursive research loop with knowledge gap detection

**Key Architecture**:
```python
class DeepResearchAgent:
    def __init__(self, research_agent: ResearchAgent, model: ModelConfig):
        self.research_agent = research_agent
        self.model = model
        self.knowledge_graph: KnowledgeGraph = KnowledgeGraph()
        self.iteration_limit: int = 3
        self.confidence_threshold: float = 0.75
    
    def run_iterative_research(self, prompt: str, initial_results: list[dict]) -> ResearchState:
        """
        Think → Search → Think Deeper loop
        """
        state = ResearchState(prompt=prompt, results=initial_results)
        
        for iteration in range(self.iteration_limit):
            # Step 1: Analyze current knowledge
            gaps = self._identify_knowledge_gaps(state)
            
            if not gaps or state.confidence >= self.confidence_threshold:
                break
            
            # Step 2: Generate targeted queries for gaps
            follow_up_queries = self._generate_gap_queries(gaps)
            
            # Step 3: Execute searches
            new_results = self.research_agent.research(follow_up_queries)
            
            # Step 4: Update knowledge graph
            state.add_results(new_results)
            state.confidence = self._calculate_confidence(state)
        
        return state
```

### 2.2 Knowledge Graph Implementation
```python
class KnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: list[Edge] = []
    
    def add_claim(self, claim: str, source: dict, confidence: float) -> None
    def detect_contradictions(self) -> list[Contradiction]
    def get_unresolved_questions(self) -> list[str]
```

### 2.3 Gap Detection Prompts
Add to `core/prompts.py`:
```python
KNOWLEDGE_GAP_ANALYSIS_SYSTEM = """
You are a research analyst identifying knowledge gaps.
Given existing research findings, identify:
1. Unanswered critical questions
2. Conflicting claims needing resolution
3. Missing evidence types (statistics, expert opinions, case studies)
4. Outdated information needing verification

Output JSON:
{
  "gaps": [
    {"question": "...", "priority": "high|medium|low", "reason": "..."}
  ],
  "confidence_score": 0.0-1.0,
  "recommendation": "continue_research|sufficient"
}
"""
```

### 2.4 Orchestrator Integration
Replace `_phase_research()` with enhanced version:
```python
def _phase_research(self, mp: MemoryPalace) -> None:
    # Initial research (existing logic)
    initial_results = self.research.research(...)
    
    # Deep-dive iterations
    deep_agent = DeepResearchAgent(self.research, self.researcher)
    research_state = deep_agent.run_iterative_research(
        mp.original_prompt,
        initial_results
    )
    
    # Update memory palace with enriched results
    mp.add_research(research_state.all_results, stance="support")
    mp.knowledge_gaps_identified = research_state.gaps
    mp.research_iterations = research_state.iterations
```

---

## Phase 3: Cross-Source Fact Verification

### 3.1 New File: `tools/fact_verification.py`
**Purpose**: Automated claim verification across sources

**Key Classes**:
```python
class FactVerifier:
    def __init__(self, model: ModelConfig):
        self.model = model
    
    def extract_claims(self, text: str) -> list[Claim]
    def verify_claim_across_sources(self, claim: Claim, sources: list[dict]) -> VerificationResult
    def detect_contradictions(self, sources: list[dict]) -> list[Contradiction]
    def resolve_conflicts(self, contradictions: list[Contradiction]) -> ConflictResolution

class Claim:
    text: str
    source: dict
    confidence: float
    verifiable: bool

class VerificationResult:
    claim: Claim
    supporting_sources: list[dict]
    contradicting_sources: list[dict]
    final_confidence: float
    verdict: "verified|contradicted|unresolved"
```

### 3.2 Verification Algorithm
```python
def verify_claim_across_sources(self, claim, sources):
    # Step 1: Semantic similarity search
    relevant = self._find_relevant_snippets(claim, sources)
    
    # Step 2: LLM-based entailment check
    entailment = self._check_entailment(claim, relevant)
    
    # Step 3: Source credibility weighting
    weighted_score = self._weight_by_credibility(entailment, sources)
    
    # Step 4: Contradiction detection
    contradictions = self._detect_contradictions(claim, relevant)
    
    return VerificationResult(
        claim=claim,
        final_confidence=weighted_score,
        verdict=self._determine_verdict(weighted_score, contradictions)
    )
```

### 3.3 Source Credibility Scoring
```python
def _calculate_source_credibility(source: dict) -> float:
    score = 0.5  # baseline
    
    # Domain bonuses
    if ".edu" in source["url"]: score += 0.2
    if ".gov" in source["url"]: score += 0.2
    if any(x in source["url"] for x in ["arxiv", "nature", "science"]): score += 0.15
    
    # Recency bonus
    if self._is_recent(source): score += 0.1
    
    # Length/depth bonus
    if len(source.get("snippet", "")) > 200: score += 0.05
    
    return min(1.0, score)
```

### 3.4 Memory Palace Updates
```python
@dataclass
class MemoryPalace:
    # Add:
    verified_claims: list[VerificationResult] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    conflict_resolutions: list[ConflictResolution] = field(default_factory=list)
    fact_verification_summary: str = ""
    
    def add_verified_claim(self, result: VerificationResult) -> None
    def build_verification_context(self) -> str
```

### 3.5 Prompts for Verification
Add to `core/prompts.py`:
```python
FACT_VERIFICATION_SYSTEM = """
You are a fact-checking AI. Given a claim and multiple source snippets:
1. Determine if sources support, contradict, or are neutral to the claim
2. Weight evidence by source credibility
3. Output JSON:
{
  "verdict": "verified|contradicted|unresolved",
  "confidence": 0.0-1.0,
  "supporting_evidence": ["...", ...],
  "contradicting_evidence": ["...", ...],
  "reasoning": "..."
}
"""
```

---

## Phase 4: UI Refactors

### 4.1 GUI Enhancements (`gui.py`)

#### New Upload Section
```python
with gr.Accordion("📎 Upload Documents (PDF, DOCX, GitHub URLs)", open=False):
    file_upload = gr.File(
        label="Upload Files",
        file_count="multiple",
        file_types=[".pdf", ".docx", ".txt", ".md"]
    )
    github_url = gr.Textbox(
        label="GitHub Repository URL",
        placeholder="https://github.com/username/repo"
    )
    ingest_btn = gr.Button("Ingest Documents")
```

#### Research Depth Control
```python
with gr.Accordion("🔬 Research Settings", open=False):
    research_depth = gr.Slider(
        minimum=1,
        maximum=5,
        value=2,
        step=1,
        label="Research Iteration Depth"
    )
    enable_fact_check = gr.Checkbox(
        label="Enable Cross-Source Fact Verification",
        value=True
    )
```

#### Results Display Enhancement
```python
# New output tabs
with gr.Tabs():
    with gr.Tab("Final Answer"):
        final_output
    with gr.Tab("📚 Sources & Citations"):
        citations_display
    with gr.Tab("✅ Fact Verification"):
        verification_report
    with gr.Tab("🔍 Research Trail"):
        research_iterations_log
    with gr.Tab("⚠️ Contradictions Found"):
        contradictions_display
```

### 4.2 Progress Callback Extensions
Update progress events:
```python
self._emit_progress("document", "Ingesting PDF: research_paper.pdf", pages=15)
self._emit_progress("deep_research", f"Iteration 2/3: Found {len(gaps)} knowledge gaps")
self._emit_progress("fact_check", f"Verifying claim: 'Quantum computing will...'", confidence=0.87)
self._emit_progress("contradiction", "Detected conflict between source A and B")
```

### 4.3 Visualization Components
```python
# Knowledge graph visualization (optional, using plotly/networkx)
knowledge_graph_viz = gr.Plot(label="Knowledge Graph")

# Confidence score gauge
confidence_gauge = gr.JSON(label="Claim Confidence Scores")

# Contradiction matrix
contradiction_matrix = gr.Dataframe(label="Source Contradiction Matrix")
```

---

## Phase 5: Configuration Updates

### 5.1 Extended `config.yaml`
```yaml
document_ingestion:
  enabled: true
  supported_formats: [pdf, docx, md, txt, github]
  chunk_size: 500
  chunk_overlap: 50
  max_documents: 10

deep_research:
  enabled: true
  max_iterations: 3
  confidence_threshold: 0.75
  gap_detection_model: qwen35-9b

fact_verification:
  enabled: true
  model: qwen35-9b
  credibility_weights:
    edu_domain: 0.2
    gov_domain: 0.2
    peer_reviewed: 0.15
    recency_bonus: 0.1
  contradiction_detection: true
```

---

## Implementation Timeline

### Week 1-2: Document Ingestion
- [ ] Create `tools/document_ingestion.py`
- [ ] Create `tools/github_ingestion.py`
- [ ] Update `MemoryPalace` with document fields
- [ ] Test PDF/DOCX parsing
- [ ] GUI file upload integration

### Week 3-4: Deep Research Agent
- [ ] Create `tools/deep_research_agent.py`
- [ ] Implement knowledge gap detection prompts
- [ ] Build iterative research loop
- [ ] Integrate with orchestrator
- [ ] Add progress tracking

### Week 5-6: Fact Verification
- [ ] Create `tools/fact_verification.py`
- [ ] Implement claim extraction
- [ ] Build contradiction detection
- [ ] Create source credibility scoring
- [ ] Add verification report generation

### Week 7: UI Polish & Testing
- [ ] Complete GUI refactors
- [ ] Add visualization components
- [ ] End-to-end testing
- [ ] Documentation updates
- [ ] Performance optimization

---

## Unique Selling Points

After implementation, this system will be the ONLY open-source LLM council with:

1. **Autonomous Deep Research** - Not just "search once", but iterative gap-filling
2. **Document + Web Hybrid Analysis** - Combine uploaded papers with live web research
3. **Built-in Fact-Checking** - Automatic contradiction detection before deliberation
4. **Citation Tracking** - Every claim traced back to source documents
5. **Research Transparency** - Full audit trail of research iterations
6. **GitHub Repo Intelligence** - Analyze codebases as primary sources

---

## Risk Mitigation

### Technical Risks
- **PDF parsing quality**: Use multiple libraries (pypdf2 + pdfplumber) with fallback
- **Context overflow**: Aggressive chunking + summarization for long documents
- **API rate limits**: Implement exponential backoff in search/research loops
- **Model hallucination**: Fact verification layer catches unsupported claims

### User Experience Risks
- **Long wait times**: Show progressive results, allow early termination
- **Information overload**: Summarize findings, use collapsible sections
- **Complexity**: Default settings work well; advanced options in accordions

---

## Success Metrics

1. **Research Quality**: 40% increase in source diversity (measured by unique domains)
2. **Fact Accuracy**: 90%+ of verified claims match ground truth in test cases
3. **User Satisfaction**: 4.5/5 rating on research comprehensiveness
4. **Contradiction Detection**: Catch 80%+ of intentional conflicts in test scenarios
5. **Document Support**: Successfully parse 95%+ of uploaded PDFs/DOCX files

---

## Next Steps

1. Review and approve this plan
2. Start with Phase 1 (Document Ingestion) - lowest risk, highest immediate value
3. Iterate based on user feedback after each phase
4. Consider community contributions for specialized parsers (academic papers, legal docs)
