"""
Academic Rehabilitation Tools — Tutor Mode Components for HF Math A Preparation

This module provides:
- PrerequisiteEngine: Maps topic dependencies and diagnoses knowledge gaps
- ComprehensionChecker: Verifies student understanding through Q&A loops
- HFTerminologyManager: Enforces Danish academic terminology
- HFProgressTracker: Tracks curriculum state and mastery levels
- HFExamSimulator: Simulates Danish HF exam formats (mundtlig/skriftlig)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("academic_rehab")


# ════════════════════════════════════════════════════════════════════════════
#  Enums and Data Classes
# ════════════════════════════════════════════════════════════════════════════

class MasteryLevel(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    NEEDS_REVIEW = "needs_review"
    MASTERED = "mastered"
    BLOCKED = "blocked"  # Prerequisites not met


@dataclass
class TopicNode:
    """Represents a single topic in the HF Math A curriculum"""
    concept: str
    level: str  # "grundforløb", "kernestof", "supplerende"
    mastery: MasteryLevel = MasteryLevel.NOT_STARTED
    prerequisites: list[str] = field(default_factory=list)
    test_question: str = ""
    last_reviewed: Optional[datetime] = None
    confidence_score: float = 0.0
    notes: str = ""


@dataclass
class TerminologyEntry:
    """Danish academic term with translations and explanations"""
    danish_term: str
    pronunciation: str
    english_equivalent: str
    nepali_concept: str
    mathematical_meaning: str
    example_usage: str
    context: str = ""  # e.g., "calculus", "statistics"


# ════════════════════════════════════════════════════════════════════════════
#  HF Math A Curriculum Dependency Graph
# ════════════════════════════════════════════════════════════════════════════

HF_MATH_A_CURRICULUM = {
    "functions": TopicNode(
        concept="functions",
        level="grundforløb",
        test_question="What is the definition of a function? Explain domain and range.",
        prerequisites=[]
    ),
    "linear_functions": TopicNode(
        concept="linear_functions",
        level="grundforløb",
        test_question="Given two points, find the equation of the line passing through them.",
        prerequisites=["functions"]
    ),
    "quadratic_functions": TopicNode(
        concept="quadratic_functions",
        level="kernestof",
        test_question="Find the vertex and roots of a quadratic function.",
        prerequisites=["functions", "linear_functions"]
    ),
    "exponential_functions": TopicNode(
        concept="exponential_functions",
        level="kernestof",
        test_question="Solve an exponential growth problem using logarithms.",
        prerequisites=["functions", "linear_functions"]
    ),
    "power_functions": TopicNode(
        concept="power_functions",
        level="kernestof",
        test_question="Differentiate a power function and explain the rule.",
        prerequisites=["functions", "exponential_functions"]
    ),
    "limits": TopicNode(
        concept="limits",
        level="kernestof",
        test_question="Evaluate lim(x→2) of (x²-4)/(x-2) and explain what it means.",
        prerequisites=["functions", "quadratic_functions"]
    ),
    "derivatives": TopicNode(
        concept="derivatives",
        level="kernestof",
        test_question="Find the derivative of f(x) = 3x² + 2x - 1 and interpret geometrically.",
        prerequisites=["functions", "limits"]
    ),
    "derivative_rules": TopicNode(
        concept="derivative_rules",
        level="kernestof",
        test_question="Apply chain rule, product rule, and quotient rule to composite functions.",
        prerequisites=["derivatives"]
    ),
    "curve_sketching": TopicNode(
        concept="curve_sketching",
        level="kernestof",
        test_question="Sketch f(x) showing critical points, inflection points, and asymptotes.",
        prerequisites=["derivatives", "derivative_rules"]
    ),
    "optimization": TopicNode(
        concept="optimization",
        level="kernestof",
        test_question="Find the maximum volume of a box with given constraints.",
        prerequisites=["derivatives", "curve_sketching"]
    ),
    "integrals": TopicNode(
        concept="integrals",
        level="kernestof",
        test_question="Calculate the area under f(x) = x² from x=0 to x=3.",
        prerequisites=["derivatives"]
    ),
    "probability": TopicNode(
        concept="probability",
        level="grundforløb",
        test_question="Calculate P(A∪B) given P(A), P(B), and P(A∩B).",
        prerequisites=[]
    ),
    "combinatorics": TopicNode(
        concept="combinatorics",
        level="kernestof",
        test_question="How many ways can you choose 3 items from 10? Explain permutations vs combinations.",
        prerequisites=["probability"]
    ),
    "binomial_distribution": TopicNode(
        concept="binomial_distribution",
        level="kernestof",
        test_question="A coin is flipped 10 times. What is P(exactly 7 heads)?",
        prerequisites=["probability", "combinatorics"]
    ),
    "normal_distribution": TopicNode(
        concept="normal_distribution",
        level="kernestof",
        test_question="Given μ=100 and σ=15, find P(X > 120).",
        prerequisites=["probability"]
    ),
    "hypothesis_testing": TopicNode(
        concept="hypothesis_testing",
        level="supplerende",
        test_question="Perform a hypothesis test at α=0.05 and interpret the p-value.",
        prerequisites=["probability", "normal_distribution"]
    ),
}


# ════════════════════════════════════════════════════════════════════════════
#  Danish Academic Terminology Database (HF Math A)
# ════════════════════════════════════════════════════════════════════════════

DANISH_MATH_TERMINOLOGY = {
    "derivative": TerminologyEntry(
        danish_term="afledt funktion",
        pronunciation="AF-ledt funk-SHON",
        english_equivalent="derivative",
        nepali_concept="परिवर्तनको दर (parivartanako dar)",
        mathematical_meaning="Instantaneous rate of change; slope of tangent line",
        example_usage="f'(x) er den afledte funktion af f(x)"
    ),
    "integral": TerminologyEntry(
        danish_term="integral",
        pronunciation="IN-te-gral",
        english_equivalent="integral",
        nepali_concept="क्षेत्रफल (kshetraphal)",
        mathematical_meaning="Area under curve; antiderivative",
        example_usage="∫f(x)dx fra a til b giver arealet under grafen"
    ),
    "function": TerminologyEntry(
        danish_term="funktion",
        pronunciation="funk-SHON",
        english_equivalent="function",
        nepali_concept="फलन (phalan)",
        mathematical_meaning="Mapping from domain to codomain",
        example_usage="f: A → B er en funktion"
    ),
    "domain": TerminologyEntry(
        danish_term="definitionsmængde",
        pronunciation="de-fi-ni-SHONS-meng-de",
        english_equivalent="domain",
        nepali_concept="परिभाषा क्षेत्र (paribhasha kshetra)",
        mathematical_meaning="Set of all valid input values",
        example_usage="Definitionsmængden for f(x) = 1/x er alle reelle tal undtagen 0"
    ),
    "range": TerminologyEntry(
        danish_term="værdimængde",
        pronunciation="VER-di-meng-de",
        english_equivalent="range",
        nepali_concept="मान क्षेत्र (man kshetra)",
        mathematical_meaning="Set of all output values",
        example_usage="Værdimængden for f(x) = x² er [0; ∞["
    ),
    "limit": TerminologyEntry(
        danish_term="grænseværdi",
        pronunciation="GREN-se-ver-di",
        english_equivalent="limit",
        nepali_concept="सीमा (seema)",
        mathematical_meaning="Value that function approaches as input approaches some value",
        example_usage="lim(x→a) f(x) = L betyder grænseværdien er L"
    ),
    "continuous": TerminologyEntry(
        danish_term="kontinuert",
        pronunciation="kon-tin-OO-ert",
        english_equivalent="continuous",
        nepali_concept="निरन्तर (nirantar)",
        mathematical_meaning="No breaks or holes in the graph",
        example_usage="Funktionen er kontinuert i hele definitionsmængden"
    ),
    "monotonic": TerminologyEntry(
        danish_term="monoton",
        pronunciation="mo-no-TONE",
        english_equivalent="monotonic",
        nepali_concept="एकदिशात्मक (ekadishatmak)",
        mathematical_meaning="Always increasing or always decreasing",
        example_usage="f er voksende monoton hvis f'(x) ≥ 0"
    ),
    "extremum": TerminologyEntry(
        danish_term="ekstremalpunkt",
        pronunciation="eks-tre-MAL-punkt",
        english_equivalent="extremum (maximum/minimum)",
        nepali_concept="चरम बिन्दु (charam bindu)",
        mathematical_meaning="Point where function reaches local or global max/min",
        example_usage="Ekstremalpunkter findes hvor f'(x) = 0"
    ),
    "inflection_point": TerminologyEntry(
        danish_term="vendepunkt",
        pronunciation="VEN-de-punkt",
        english_equivalent="inflection point",
        nepali_concept="वक्रता परिवर्तन बिन्दु (vakrata parivartan bindu)",
        mathematical_meaning="Point where concavity changes",
        example_usage="Vendepunkter findes hvor f''(x) = 0 og skifter fortegn"
    ),
    "asymptote": TerminologyEntry(
        danish_term="asymptote",
        pronunciation="as-ymp-TOTE",
        english_equivalent="asymptote",
        nepali_concept="अनंतस्पर्शी रेखा (anantasparshi rekha)",
        mathematical_meaning="Line that curve approaches but never touches",
        example_usage="x = 0 er en lodret asymptote for f(x) = 1/x"
    ),
    "probability": TerminologyEntry(
        danish_term="sandsynlighed",
        pronunciation="SANN-syn-lig-hed",
        english_equivalent="probability",
        nepali_concept="सम्भावना (sambhawana)",
        mathematical_meaning="Measure of likelihood (0 to 1)",
        example_usage="Sandsynligheden for at slå en 6'er er 1/6"
    ),
    "hypothesis_test": TerminologyEntry(
        danish_term="hypotesetest",
        pronunciation="hy-po-te-se TEST",
        english_equivalent="hypothesis test",
        nepali_concept="परिकल्पना परीक्षण (parikalpana parikshan)",
        mathematical_meaning="Statistical test to reject or fail to reject null hypothesis",
        example_usage="Vi forkaster nulhypotesen hvis p-værdi < α"
    ),
    "confidence_interval": TerminologyEntry(
        danish_term="konfidensinterval",
        pronunciation="kon-fi-dens-IN-ter-val",
        english_equivalent="confidence interval",
        nepali_concept="विश्वास अन्तराल (vishwas antaral)",
        mathematical_meaning="Range of values likely to contain population parameter",
        example_usage="95% konfidensinterval for middelværdien er [48; 52]"
    ),
}


# ════════════════════════════════════════════════════════════════════════════
#  Prerequisite Engine
# ════════════════════════════════════════════════════════════════════════════

class PrerequisiteEngine:
    """
    Diagnoses knowledge gaps and enforces prerequisite chains before allowing
    advancement to new topics.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.curriculum = HF_MATH_A_CURRICULUM.copy()
        self.custom_topics: dict[str, TopicNode] = {}
        
    def diagnose_gaps(self, topic: str) -> dict[str, Any]:
        """
        Returns ordered list of prerequisites with confidence scores.
        
        Args:
            topic: The target topic the student wants to learn
            
        Returns:
            Dictionary with learning path, missing prerequisites, and diagnostic questions
        """
        topic_lower = topic.lower().strip()
        
        # Find matching topic
        matched_topic = self._find_topic(topic_lower)
        if not matched_topic:
            return {
                "error": f"Topic '{topic}' not found in HF Math A curriculum",
                "available_topics": list(self.curriculum.keys())
            }
        
        # Build prerequisite chain
        prereq_chain = self._build_prerequisite_chain(matched_topic)
        
        return {
            "target_topic": matched_topic,
            "learning_path": prereq_chain,
            "total_steps": len(prereq_chain),
            "ready_to_advance": self._check_readiness(prereq_chain)
        }
    
    def _find_topic(self, topic_key: str) -> Optional[TopicNode]:
        """Find topic by key or partial match"""
        if topic_key in self.curriculum:
            return self.curriculum[topic_key]
        
        # Partial match
        for key, node in self.curriculum.items():
            if topic_key in key or key in topic_key:
                return node
        
        return None
    
    def _build_prerequisite_chain(self, topic: TopicNode, visited: set = None) -> list[dict]:
        """Recursively build ordered prerequisite chain"""
        if visited is None:
            visited = set()
        
        if topic.concept in visited:
            return []
        
        visited.add(topic.concept)
        chain = []
        
        # First add all prerequisites
        for prereq_name in topic.prerequisites:
            if prereq_name in self.curriculum:
                prereq_node = self.curriculum[prereq_name]
                chain.extend(self._build_prerequisite_chain(prereq_node, visited))
        
        # Then add current topic
        chain.append({
            "concept": topic.concept,
            "level": topic.level,
            "test_question": topic.test_question,
            "mastery_required": topic.mastery != MasteryLevel.NOT_STARTED
        })
        
        return chain
    
    def _check_readiness(self, chain: list[dict]) -> bool:
        """Check if all prerequisites in chain are mastered"""
        # This would integrate with Mem0 to check actual mastery
        # For now, return True if chain is empty or has only one item
        return len(chain) <= 1
    
    def verify_mastery(self, concept: str, student_response: str, model_config=None) -> dict:
        """
        Council debates whether student actually understands the concept.
        
        Args:
            concept: The concept being tested
            student_response: Student's answer to diagnostic question
            model_config: Optional model config for evaluation
            
        Returns:
            Dictionary with mastery verdict, confidence, and feedback
        """
        if not self.orchestrator:
            # Fallback: simple keyword matching
            return self._simple_mastery_check(concept, student_response)
        
        # Use council to evaluate response
        prompt = f"""
        You are evaluating a student's understanding of "{concept}".
        
        Student's response:
        {student_response}
        
        Evaluate based on:
        1. Conceptual accuracy
        2. Use of correct terminology
        3. Logical reasoning
        
        Respond in JSON format:
        {{
            "mastery_achieved": true/false,
            "confidence_score": 0.0-1.0,
            "feedback": "Constructive feedback in Danish/Nepali bilingual format",
            "next_step": "advance" or "review" or "reteach"
        }}
        """
        
        try:
            result = self.orchestrator.query_model(model_config or "phi4-mini", prompt)
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            logger.warning(f"Mastery verification failed: {e}")
            return self._simple_mastery_check(concept, student_response)
    
    def _simple_mastery_check(self, concept: str, response: str) -> dict:
        """Fallback mastery check without council"""
        response_lower = response.lower()
        
        # Simple heuristics based on concept
        key_indicators = {
            "functions": ["mapping", "domain", "range", "input", "output"],
            "derivatives": ["slope", "rate", "change", "tangent", "f'"],
            "integrals": ["area", "under", "antiderivative", "∫"],
            "limits": ["approaches", "limit", "lim", "infinity"],
        }
        
        indicators = key_indicators.get(concept.lower(), [])
        match_count = sum(1 for ind in indicators if ind in response_lower)
        
        confidence = min(1.0, match_count / max(1, len(indicators)))
        
        return {
            "mastery_achieved": confidence >= 0.6,
            "confidence_score": confidence,
            "feedback": f"Response shows {'good' if confidence >= 0.6 else 'limited'} understanding of {concept}",
            "next_step": "advance" if confidence >= 0.6 else "review"
        }


# ════════════════════════════════════════════════════════════════════════════
#  Comprehension Checker
# ════════════════════════════════════════════════════════════════════════════

class ComprehensionChecker:
    """
    Implements the teaching loop: Explain → Question → Verify → Re-explain if needed
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.question_history: list[dict] = []
        
    def generate_comprehension_questions(self, explanation: str, topic: str) -> list[dict]:
        """
        Generate 2-3 questions from the synthesis to verify understanding.
        
        Returns:
            List of questions with type, difficulty, and expected concepts
        """
        if not self.orchestrator:
            return self._generate_fallback_questions(topic)
        
        prompt = f"""
        Based on this explanation about {topic}:
        {explanation[:2000]}
        
        Generate 3 comprehension questions:
        1. One conceptual question (Why does X work?)
        2. One procedural question (Calculate Y given Z)
        3. One Danish terminology question (Explain X using correct Danish terms)
        
        Format as JSON array:
        [
            {{
                "type": "conceptual/procedural/terminology",
                "question": "...",
                "expected_concepts": ["...", "..."],
                "difficulty": "easy/medium/hard"
            }}
        ]
        """
        
        try:
            result = self.orchestrator.query_model("phi4-mini", prompt)
            questions = json.loads(result) if isinstance(result, str) else result
            self.question_history.extend([{"topic": topic, "q": q} for q in questions])
            return questions
        except Exception as e:
            logger.warning(f"Question generation failed: {e}")
            return self._generate_fallback_questions(topic)
    
    def _generate_fallback_questions(self, topic: str) -> list[dict]:
        """Generate standard questions when model unavailable"""
        return [
            {
                "type": "conceptual",
                "question": f"Explain the key idea behind {topic} in your own words.",
                "expected_concepts": ["main concept", "purpose"],
                "difficulty": "easy"
            },
            {
                "type": "procedural",
                "question": f"Show how you would solve a typical {topic} problem step-by-step.",
                "expected_concepts": ["method", "steps"],
                "difficulty": "medium"
            },
            {
                "type": "terminology",
                "question": f"What is the Danish term for {topic}? Use it in a sentence.",
                "expected_concepts": ["danish term", "usage"],
                "difficulty": "easy"
            }
        ]
    
    def verify_learning(self, questions: list[dict], student_answers: list[str]) -> dict:
        """
        Council debates whether student actually understood.
        
        Returns:
            Dictionary with pass/fail verdict, areas of weakness, and recommendation
        """
        if not self.orchestrator:
            return self._simple_verification(questions, student_answers)
        
        # Prepare evaluation prompt
        qa_pairs = "\n\n".join([
            f"Q{idx+1}: {q['question']}\nA{idx+1}: {ans}"
            for idx, (q, ans) in enumerate(zip(questions, student_answers))
        ])
        
        prompt = f"""
        Evaluate student responses:
        
        {qa_pairs}
        
        Debate as a council:
        - Does the student truly understand or just memorize?
        - Are there gaps in reasoning?
        - Should we advance or re-explain with different approach?
        
        JSON response:
        {{
            "passed": true/false,
            "areas_of_strength": ["...", "..."],
            "areas_needing_work": ["...", "..."],
            "recommendation": "advance/review/reteach",
            "suggested_approach": "Which council member should lead next explanation"
        }}
        """
        
        try:
            result = self.orchestrator.query_model("gemma4-9b", prompt)
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            logger.warning(f"Learning verification failed: {e}")
            return self._simple_verification(questions, student_answers)
    
    def _simple_verification(self, questions: list[dict], answers: list[str]) -> dict:
        """Simple heuristic-based verification"""
        total_length = sum(len(ans) for ans in answers)
        avg_length = total_length / max(1, len(answers))
        
        # Heuristic: substantive answers suggest engagement
        passed = avg_length > 50  # At least 50 chars per answer on average
        
        return {
            "passed": passed,
            "areas_of_strength": ["engagement"] if passed else [],
            "areas_needing_work": ["depth" if not passed else ""],
            "recommendation": "advance" if passed else "review",
            "suggested_approach": "explainer" if not passed else "examiner"
        }


# ════════════════════════════════════════════════════════════════════════════
#  Danish Terminology Manager
# ════════════════════════════════════════════════════════════════════════════

class HFTerminologyManager:
    """
    Manages Danish academic terminology enforcement and bilingual bridging.
    """
    
    def __init__(self, native_language: str = "Nepali"):
        self.native_language = native_language
        self.terminology_db = DANISH_MATH_TERMINOLOGY.copy()
        self.user_custom_terms: dict[str, TerminologyEntry] = {}
        self.usage_log: list[dict] = []
        
    def get_term(self, english_term: str) -> Optional[TerminologyEntry]:
        """Retrieve Danish term with full context"""
        term_lower = english_term.lower().strip()
        
        if term_lower in self.terminology_db:
            return self.terminology_db[term_lower]
        
        # Try fuzzy match
        for key, entry in self.terminology_db.items():
            if term_lower in key or key in term_lower:
                return entry
        
        return None
    
    def format_bilingual_explanation(self, danish_term: str, concept_explanation: str) -> str:
        """
        Format explanation with Danish term and native language concept.
        
        Example: "Den afledte funktion (परिवर्तनको दर) measures..."
        """
        entry = self.get_term(danish_term)
        if not entry:
            return concept_explanation
        
        if self.native_language == "Nepali":
            native_hint = entry.nepali_concept
        else:
            native_hint = entry.english_equivalent
        
        return f"{entry.danish_term} ({native_hint}) - {concept_explanation}"
    
    def generate_terminology_quiz(self, topic: str, num_questions: int = 5) -> list[dict]:
        """Generate quiz on Danish mathematical terminology"""
        relevant_terms = [
            (key, entry) for key, entry in self.terminology_db.items()
            if topic.lower() in entry.context or entry.context == ""
        ]
        
        import random
        selected = random.sample(relevant_terms, min(num_questions, len(relevant_terms)))
        
        quiz = []
        for key, entry in selected:
            quiz.append({
                "question": f"What is the Danish term for '{entry.english_equivalent}'?",
                "answer": entry.danish_term,
                "pronunciation": entry.pronunciation,
                "hint": entry.nepali_concept if self.native_language == "Nepali" else entry.english_equivalent,
                "example": entry.example_usage
            })
        
        return quiz
    
    def log_usage(self, term: str, used_correctly: bool):
        """Track terminology usage for spaced repetition"""
        self.usage_log.append({
            "timestamp": datetime.now().isoformat(),
            "term": term,
            "correct": used_correctly
        })
    
    def get_weak_terms(self) -> list[str]:
        """Identify terms the student struggles with"""
        from collections import Counter
        
        incorrect = [
            entry["term"] for entry in self.usage_log
            if not entry["correct"]
        ]
        
        return [term for term, count in Counter(incorrect).most_common(5)]


# ════════════════════════════════════════════════════════════════════════════
#  HF Progress Tracker
# ════════════════════════════════════════════════════════════════════════════

class HFProgressTracker:
    """
    Tracks curriculum state and mastery levels across sessions.
    Integrates with Mem0 for persistence.
    """
    
    def __init__(self, user_id: str = "hf_student", mem0_manager=None):
        self.user_id = user_id
        self.mem0_manager = mem0_manager
        self.subjects: dict[str, dict[str, TopicNode]] = {
            "math_a": HF_MATH_A_CURRICULUM.copy()
        }
        self.session_history: list[dict] = []
        
    def record_mastery(self, concept: str, confidence: float, source_session: str):
        """Mark concept as learned with evidence"""
        timestamp = datetime.now().isoformat()
        
        # Update in-memory state
        for subject, topics in self.subjects.items():
            if concept in topics:
                topics[concept].mastery = (
                    MasteryLevel.MASTERED if confidence >= 0.8
                    else MasteryLevel.IN_PROGRESS if confidence >= 0.5
                    else MasteryLevel.NEEDS_REVIEW
                )
                topics[concept].confidence_score = confidence
                topics[concept].last_reviewed = datetime.fromisoformat(timestamp)
        
        # Persist to Mem0
        if self.mem0_manager:
            memory_text = (
                f"Student demonstrated {confidence*100:.0f}% mastery of {concept} "
                f"in session {source_session}"
            )
            self.mem0_manager.add_memory(
                text=memory_text,
                metadata={
                    "type": "mastery",
                    "concept": concept,
                    "confidence": confidence,
                    "subject": "math_a",
                    "session": source_session,
                    "timestamp": timestamp
                }
            )
    
    def get_weak_areas(self) -> list[str]:
        """Query for concepts with confidence < 0.7"""
        weak = []
        for subject, topics in self.subjects.items():
            for concept, node in topics.items():
                if node.confidence_score < 0.7 and node.mastery != MasteryLevel.NOT_STARTED:
                    weak.append(concept)
        
        # Also query Mem0 if available
        if self.mem0_manager:
            try:
                memories = self.mem0_manager.search("mastery confidence low")
                for mem in memories:
                    if mem.get("confidence", 1.0) < 0.7:
                        concept = mem.get("concept")
                        if concept and concept not in weak:
                            weak.append(concept)
            except Exception as e:
                logger.warning(f"Failed to query Mem0 for weak areas: {e}")
        
        return weak
    
    def unblock_next_topic(self, completed_concept: str) -> Optional[str]:
        """
        When a concept is mastered, unlock dependent topics.
        
        Returns:
            Name of newly unblocked topic, or None if none available
        """
        # Update completed concept
        self.record_mastery(completed_concept, 0.85, "manual_unlock")
        
        # Find topics that had this as prerequisite
        for subject, topics in self.subjects.items():
            for concept, node in topics.items():
                if completed_concept in node.prerequisites:
                    # Check if ALL prerequisites are now mastered
                    all_prereqs_met = all(
                        self.subjects.get("math_a", {}).get(prereq, TopicNode(prereq, "")).mastery 
                        in [MasteryLevel.MASTERED, MasteryLevel.IN_PROGRESS]
                        for prereq in node.prerequisites
                    )
                    
                    if all_prereqs_met and node.mastery == MasteryLevel.BLOCKED:
                        node.mastery = MasteryLevel.NOT_STARTED
                        return concept
        
        return None
    
    def suggest_next_topic(self) -> dict:
        """
        Uses HF syllabus order + student gaps to suggest what to study next.
        
        Returns:
            Dictionary with suggested topic, reason, and prerequisites status
        """
        # Priority order:
        # 1. Topics in progress (not yet mastered)
        # 2. Unblocked topics not started
        # 3. Review of weak areas
        
        math_topics = self.subjects.get("math_a", {})
        
        # In-progress topics
        in_progress = [
            (name, node) for name, node in math_topics.items()
            if node.mastery == MasteryLevel.IN_PROGRESS
        ]
        
        if in_progress:
            # Pick the one with highest confidence (closest to mastery)
            best = max(in_progress, key=lambda x: x[1].confidence_score)
            return {
                "suggested_topic": best[0],
                "reason": "Continue building on current progress",
                "current_confidence": best[1].confidence_score,
                "prerequisites_status": "met"
            }
        
        # Not-started but unblocked topics
        ready_topics = [
            (name, node) for name, node in math_topics.items()
            if node.mastery == MasteryLevel.NOT_STARTED and self._are_prerequisites_met(node)
        ]
        
        if ready_topics:
            # Follow curriculum order (grundforløb → kernestof → supplerende)
            level_order = {"grundforløb": 0, "kernestof": 1, "supplerende": 2}
            ready_topics.sort(key=lambda x: level_order.get(x[1].level, 99))
            
            return {
                "suggested_topic": ready_topics[0][0],
                "reason": f"New topic ready to learn ({ready_topics[0][1].level})",
                "prerequisites_status": "met"
            }
        
        # Weak areas needing review
        weak = self.get_weak_areas()
        if weak:
            return {
                "suggested_topic": weak[0],
                "reason": "Review needed - confidence below threshold",
                "prerequisites_status": "previously_attempted"
            }
        
        return {
            "suggested_topic": None,
            "reason": "All topics either mastered or blocked",
            "prerequisites_status": "unknown"
        }
    
    def _are_prerequisites_met(self, topic: TopicNode) -> bool:
        """Check if all prerequisites are mastered or in progress"""
        math_topics = self.subjects.get("math_a", {})
        
        for prereq in topic.prerequisites:
            prereq_node = math_topics.get(prereq)
            if not prereq_node or prereq_node.mastery == MasteryLevel.NOT_STARTED:
                return False
            if prereq_node.mastery == MasteryLevel.BLOCKED:
                return False
        
        return True
    
    def get_curriculum_status(self) -> dict:
        """Get overview of entire curriculum progress"""
        math_topics = self.subjects.get("math_a", {})
        
        status = {
            "mastered": [],
            "in_progress": [],
            "needs_review": [],
            "not_started": [],
            "blocked": []
        }
        
        for name, node in math_topics.items():
            status[node.mastery.value].append({
                "concept": name,
                "level": node.level,
                "confidence": node.confidence_score
            })
        
        return status


# ════════════════════════════════════════════════════════════════════════════
#  HF Exam Simulator
# ════════════════════════════════════════════════════════════════════════════

class ExamFormat(Enum):
    MUNDTLIG = "mundtlig"  # Oral exam
    SKRIFTLIG = "skriftlig"  # Written exam
    DRILL = "drill"  # Practice problems


class HFExamSimulator:
    """
    Simulates Danish HF exam formats for preparation.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.exam_templates = self._load_exam_templates()
        
    def _load_exam_templates(self) -> dict:
        """Load HF exam format templates"""
        return {
            ExamFormat.MUNDTLIG: {
                "duration_minutes": 30,
                "prep_time_minutes": 30,
                "structure": [
                    "Student presents solution (10 min)",
                    "Examiner asks clarifying questions (10 min)",
                    "Discussion of method choice and alternatives (10 min)"
                ],
                "evaluation_criteria": [
                    "Mathematical reasoning",
                    "Method selection justification",
                    "Communication clarity",
                    "Use of correct terminology"
                ]
            },
            ExamFormat.SKRIFTLIG: {
                "duration_minutes": 240,  # 4 hours
                "allowed_aids": "All aids allowed (typisk)",
                "structure": [
                    "Problem 1: Basic skills (mandatory)",
                    "Problem 2-4: Applied problems",
                    "Problem 5: Advanced/challenge problem"
                ],
                "notation_requirements": [
                    "Show all steps clearly",
                    "Use proper mathematical notation",
                    "Include units where applicable",
                    "State assumptions explicitly"
                ]
            },
            ExamFormat.DRILL: {
                "duration_minutes": 15,
                "problems_per_session": 5,
                "focus": "Speed and accuracy on specific topic"
            }
        }
    
    def generate_exam_problem(self, topic: str, exam_format: ExamFormat, difficulty: str = "medium") -> dict:
        """Generate an exam-style problem"""
        if not self.orchestrator:
            return self._generate_fallback_problem(topic, exam_format)
        
        template = self.exam_templates[exam_format]
        
        prompt = f"""
        Generate a Danish HF Math A {exam_format.value} exam problem about {topic}.
        Difficulty: {difficulty}
        
        Requirements:
        - Use authentic Danish exam style and phrasing
        - Include any necessary diagrams or data (describe them)
        - Specify point allocation
        - For mundtlig: Include follow-up questions examiner might ask
        - For skriftlig: Ensure it fits time constraints
        
        Format as JSON:
        {{
            "problem_statement": "...",
            "points": number,
            "estimated_time_minutes": number,
            "required_concepts": ["...", "..."],
            "follow_up_questions": ["...", "..."],  // for mundtlig
            "solution_outline": "...",
            "common_mistakes": ["...", "..."]
        }}
        """
        
        try:
            result = self.orchestrator.query_model("gemma4-9b", prompt)
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            logger.warning(f"Exam problem generation failed: {e}")
            return self._generate_fallback_problem(topic, exam_format)
    
    def _generate_fallback_problem(self, topic: str, exam_format: ExamFormat) -> dict:
        """Generate basic problem when model unavailable"""
        return {
            "problem_statement": f"Solve a {topic} problem: [Specific problem would be generated by model]",
            "points": 10,
            "estimated_time_minutes": 15,
            "required_concepts": [topic],
            "follow_up_questions": ["Explain your method choice", "Are there alternative approaches?"],
            "solution_outline": "[Solution would be generated by model]",
            "common_mistakes": ["Sign errors", "Misapplying formulas"]
        }
    
    def evaluate_exam_response(self, problem: dict, student_response: str, exam_format: ExamFormat) -> dict:
        """Evaluate student response using HF rubrics"""
        if not self.orchestrator:
            return self._simple_evaluation(problem, student_response, exam_format)
        
        template = self.exam_templates[exam_format]
        
        prompt = f"""
        Evaluate this HF Math A {exam_format.value} exam response.
        
        Problem:
        {json.dumps(problem, indent=2)}
        
        Student Response:
        {student_response}
        
        Use Danish HF censor (eksaminator) criteria:
        {template['evaluation_criteria']}
        
        Provide scoring and feedback in JSON:
        {{
            "overall_score": 0.0-1.0,
            "breakdown": {{
                "mathematical_reasoning": 0.0-1.0,
                "method_selection": 0.0-1.0,
                "communication": 0.0-1.0,
                "terminology": 0.0-1.0
            }},
            "points_awarded": number,
            "max_points": number,
            "feedback_da": "Feedback in Danish",
            "feedback_en": "Feedback in English",
            "areas_for_improvement": ["...", "..."],
            "grade_estimate": "-3/00/02/4/7/10/12"  // Danish grading scale
        }}
        """
        
        try:
            result = self.orchestrator.query_model("gemma4-9b", prompt)
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            logger.warning(f"Exam evaluation failed: {e}")
            return self._simple_evaluation(problem, student_response, exam_format)
    
    def _simple_evaluation(self, problem: dict, response: str, exam_format: ExamFormat) -> dict:
        """Simple fallback evaluation"""
        response_length = len(response)
        has_math_notation = any(c in response for c in ["=", "+", "-", "*", "/", "^", "√"])
        
        base_score = min(1.0, response_length / 500)
        notation_bonus = 0.2 if has_math_notation else 0
        
        overall = min(1.0, base_score + notation_bonus)
        
        return {
            "overall_score": overall,
            "breakdown": {
                "mathematical_reasoning": overall,
                "method_selection": overall,
                "communication": overall,
                "terminology": overall
            },
            "points_awarded": round(overall * problem.get("points", 10)),
            "max_points": problem.get("points", 10),
            "feedback_da": f"Dit svar viser {'god' if overall > 0.6 else 'begrænset'} forståelse.",
            "feedback_en": f"Your response shows {'good' if overall > 0.6 else 'limited'} understanding.",
            "areas_for_improvement": ["Show more steps", "Use proper notation"],
            "grade_estimate": self._score_to_grade(overall)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to Danish 7-trins skala"""
        if score >= 0.95:
            return "12"
        elif score >= 0.85:
            return "10"
        elif score >= 0.75:
            return "7"
        elif score >= 0.60:
            return "4"
        elif score >= 0.50:
            return "02"
        elif score >= 0.35:
            return "00"
        else:
            return "-3"


# ════════════════════════════════════════════════════════════════════════════
#  Tutor Mode Configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TutorModeConfig:
    """Configuration for tutor mode operation"""
    enabled: bool = False
    study_mode: str = "tutorial"  # tutorial, drill, exam_prep
    subject: str = "math_a"
    topic: Optional[str] = None
    enforce_danish_terminology: bool = True
    prerequisite_blocking: bool = True
    comprehension_checks: bool = True
    spaced_repetition: bool = True
    native_language: str = "Nepali"
    exam_format: ExamFormat = ExamFormat.DRILL


# Specialized council for tutor mode
TUTOR_MODE_COUNCIL = [
    {
        "model_id": "foundations",
        "ollama_name": "qwen3-8b",
        "display_name": "Prerequisites Guardian",
        "personality": "You are a strict diagnostic tutor. Before answering ANY question, you MUST verify the student knows the prerequisites. If they don't, stop and teach those first. Never skip steps."
    },
    {
        "model_id": "examiner",
        "ollama_name": "gemma4-9b",
        "display_name": "HF Examiner",
        "personality": "You are a Danish HF censor (eksaminator). You evaluate answers based on HF rubrics: mathematical reasoning, method selection, and communication. Be critical but constructive."
    },
    {
        "model_id": "explainer",
        "ollama_name": "phi4-mini",
        "display_name": "Simple Explainer",
        "personality": "You explain complex concepts using everyday analogies. When possible, relate to Nepali contexts or universal experiences. Focus on intuition over formalism."
    }
]
