"""LLM Council GUI – redesigned dark deliberation chamber."""

from __future__ import annotations

import json
import queue
import re
import threading
import warnings
from pathlib import Path

import gradio as gr

try:
    from pandas.errors import Pandas4Warning
except Exception:
    Pandas4Warning = None

# Gradio currently triggers Pandas 3.x/4.x transition warnings internally.
# These are non-actionable in this app and can be safely silenced.
warnings.filterwarnings(
    "ignore",
    message=r".*future\.no_silent_downcasting.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*copy keyword is deprecated.*",
)
if Pandas4Warning is not None:
    warnings.filterwarnings("ignore", category=Pandas4Warning)

from core.mem0_manager import Mem0MemoryManager
from orchestrator import CONFIG_PATH, build_orchestrator_from_config

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_DIR = ROOT_DIR / "council_states"

# ─────────────────────────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def _latest_session_file(state_dir: Path) -> Path | None:
    if not state_dir.exists():
        return None
    files = sorted(state_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _list_session_files(state_dir: Path) -> list[str]:
    if not state_dir.exists():
        return []
    files = sorted(state_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.name for f in files]


def _clip(text: str, max_chars: int = 240) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _build_snapshot_markdown(state_json: dict) -> str:
    prompt = state_json.get("original_prompt", "")
    summary = state_json.get("research_summary", "")
    ideas = state_json.get("council_ideas", {})
    proposals = state_json.get("synthesizer_proposals", [])
    supporting = state_json.get("supporting_research", [])
    counter = state_json.get("counter_research", [])
    consensus_mode = state_json.get("consensus_mode", "not-run")
    evidence_quality_score = float(state_json.get("evidence_quality_score", 0.0) or 0.0)
    evidence_quality_note = state_json.get("evidence_quality_note", "")
    iteration = state_json.get("current_iteration", 0)
    best_index = state_json.get("best_proposal_index", -1)

    best_score = 0.0
    risk_label = "n/a"
    density_text = "n/a"
    delta_text = "n/a"
    if isinstance(best_index, int) and best_index >= 0 and best_index < len(proposals):
        best_score = float(proposals[best_index].get("average_score", 0.0))
        best = proposals[best_index]
        scores = best.get("council_scores", {})
        if scores:
            min_score = float(min(scores.values()))
            delta = max(0.0, best_score - min_score)
            delta_text = f"{delta:.4f}"
            total = len(supporting) + len(counter)
            density = (len(counter) / total) if total > 0 else 0.0
            density_text = f"{density * 100:.1f}% ({len(counter)}/{total})"
            if delta < 0.02 and density < 0.35:
                risk_label = "Low"
            elif delta < 0.06 and density < 0.5:
                risk_label = "Medium"
            else:
                risk_label = "High"

    return (
        "### Session Snapshot\n"
        f"- Prompt length: {len(prompt)} chars\n"
        f"- Research summary length: {len(summary)} chars\n"
        f"- Council ideas captured: {len(ideas)}\n"
        f"- Supporting sources: {len(supporting)}\n"
        f"- Counter-evidence sources: {len(counter)}\n"
        f"- Synthesizer iterations: {iteration}\n"
        f"- Best consensus score: {best_score:.4f}\n"
        f"- Accepted risk level: {risk_label}\n"
        f"- Counter-evidence density: {density_text}\n"
        f"- Majority-dissent delta: {delta_text}\n"
        f"- Consensus mode: {consensus_mode}\n"
        f"- Evidence quality: {evidence_quality_score:.2f}\n"
        f"- Evidence note: {evidence_quality_note}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Minority report
# ─────────────────────────────────────────────────────────────────────────────

def _build_minority_report_markdown(state_json: dict) -> str:
    proposals = state_json.get("synthesizer_proposals", [])
    best_index = state_json.get("best_proposal_index", -1)
    ideas = state_json.get("council_ideas", {})

    if not proposals or not isinstance(best_index, int) or best_index < 0 or best_index >= len(proposals):
        return "No minority report available yet. Run or load a completed session."

    best = proposals[best_index]
    scores = best.get("council_scores", {})
    critiques = best.get("council_critiques", {})
    if not scores:
        return "No minority report available for this proposal."

    dissent_id = min(scores, key=scores.get)
    dissent_score = float(scores.get(dissent_id, 0.0))
    model_name = ideas.get(dissent_id, {}).get("model_name", dissent_id)
    critique = critiques.get(dissent_id, "No dissent critique was recorded.")

    return (
        "### The Case Against This Decision\n"
        f"- Dissenting seat: {model_name}\n"
        f"- Score: {dissent_score:.4f}\n\n"
        f"{critique}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decision signals HTML
# ─────────────────────────────────────────────────────────────────────────────

def _build_decision_signals_html(state_json: dict) -> str:
    proposals = state_json.get("synthesizer_proposals", [])
    best_index = state_json.get("best_proposal_index", -1)
    supporting = state_json.get("supporting_research", [])
    counter = state_json.get("counter_research", [])
    consensus_mode = str(state_json.get("consensus_mode", "not-run"))
    evidence_quality_score = float(state_json.get("evidence_quality_score", 0.0) or 0.0)

    if not proposals or not isinstance(best_index, int) or best_index < 0 or best_index >= len(proposals):
        return "<div style='padding:16px;color:#5e6358;font-family:\"Martian Mono\",monospace;font-size:12px;'>Decision signals appear after a completed run.</div>"

    best = proposals[best_index]
    scores = best.get("council_scores", {})
    if not scores:
        return "<div style='padding:16px;color:#5e6358;font-family:\"Martian Mono\",monospace;font-size:12px;'>No vote data available.</div>"

    avg_score = float(best.get("average_score", 0.0))
    min_score = float(min(scores.values()))
    delta = max(0.0, avg_score - min_score)

    counter_count = len(counter)
    support_count = len(supporting)
    total = support_count + counter_count
    density = (counter_count / total) if total > 0 else 0.0

    if delta < 0.02 and density < 0.35:
        risk_label = "LOW"
        risk_color = "#1a9e84"
        risk_bg = "rgba(26,158,132,0.06)"
        risk_border = "rgba(26,158,132,0.25)"
    elif delta < 0.06 and density < 0.5:
        risk_label = "MEDIUM"
        risk_color = "#c49215"
        risk_bg = "rgba(196,146,21,0.06)"
        risk_border = "rgba(196,146,21,0.25)"
    else:
        risk_label = "HIGH"
        risk_color = "#c4442c"
        risk_bg = "rgba(196,68,44,0.06)"
        risk_border = "rgba(196,68,44,0.25)"

    mono = '"Martian Mono",monospace'
    serif = '"Cormorant Garamond","Georgia",serif'
    bg1 = "#141614"
    border = "#2c2f28"
    text = "#e0dbd0"
    dim = "#5e6358"

    def card(eyebrow, value, sub, val_color, bg=bg1, bdr=border):
        return f"""
        <div style='padding:18px 20px;background:{bg};border:1px solid {bdr};border-top:2px solid {val_color};'>
            <div style='font-family:{mono};font-size:9.5px;letter-spacing:0.16em;text-transform:uppercase;color:{dim};margin-bottom:8px;'>{eyebrow}</div>
            <div style='font-family:{serif};font-size:2.2rem;font-weight:600;color:{val_color};line-height:1;margin-bottom:8px;'>{value}</div>
            <div style='font-family:{mono};font-size:11px;color:{dim};'>{sub}</div>
        </div>"""

    cards_html = "".join([
        card("Accepted Risk Level", risk_label, "based on dissent severity", risk_color, risk_bg, risk_border),
        card("Counter-Evidence Density", f"{density * 100:.1f}%", f"{counter_count} of {total} sources", "#c4442c"),
        card("Majority-Dissent Delta", f"{delta:.4f}", f"avg {avg_score:.4f} vs dissent {min_score:.4f}", "#3a78c4"),
        card("Consensus Score", f"{avg_score:.4f}", f"across {len(scores)} council seats", "#1a9e84"),
        card("Consensus Validity", "YES" if consensus_mode == "threshold" else "BEST-EFFORT", consensus_mode, "#1a9e84" if consensus_mode == "threshold" else "#c49215"),
        card("Evidence Quality", f"{evidence_quality_score:.2f}", "filtered source quality score", "#3a78c4"),
    ])

    return f"""
<div style='font-family:{mono};padding:4px;'>
    <div style='font-family:{serif};font-size:1.05rem;font-weight:600;color:{text};
                letter-spacing:0.01em;padding:12px 4px 10px;border-bottom:1px solid {border};margin-bottom:14px;'>
        Decision Signals
    </div>
    <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;'>
        {cards_html}
    </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial evidence HTML
# ─────────────────────────────────────────────────────────────────────────────

def _build_adversarial_evidence_html(state_json: dict) -> str:
    supporting = state_json.get("supporting_research", [])
    counter = state_json.get("counter_research", [])

    if not supporting and not counter:
        mixed = state_json.get("web_research", [])
        supporting = [r for r in mixed if r.get("stance") == "support"]
        counter = [r for r in mixed if r.get("stance") == "counter"]

    mono = '"Martian Mono",monospace'
    serif = '"Cormorant Garamond","Georgia",serif'
    bg1 = "#141614"
    bg2 = "#1b1d18"
    border = "#2c2f28"
    text = "#e0dbd0"
    dim = "#5e6358"

    def render_items(rows: list[dict], accent: str, accent_bg: str, accent_border: str, title: str) -> str:
        if not rows:
            return (
                f"<div style='padding:14px;border:1px dashed {border};color:{dim};"
                f"font-family:{mono};font-size:11px;'>No {title.lower()} in this session.</div>"
            )
        cards = []
        for r in rows[:5]:
            url = (r.get("url") or "").strip()
            link = (
                f"<a href='{url}' target='_blank' style='color:{accent};text-decoration:none;"
                f"font-size:10px;letter-spacing:0.08em;'>→ source</a>"
            ) if url else ""
            cards.append(
                f"<div style='padding:14px;background:{bg1};border:1px solid {border};"
                f"border-left:2px solid {accent};margin-bottom:8px;'>"
                f"<div style='font-family:{serif};font-size:0.95rem;font-weight:600;"
                f"color:{text};margin-bottom:6px;line-height:1.3;'>{_clip(r.get('title', ''), 100)}</div>"
                f"<div style='font-family:{mono};font-size:11px;color:{dim};line-height:1.55;'>{_clip(r.get('snippet', ''), 260)}</div>"
                f"<div style='margin-top:10px;display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='font-family:{mono};font-size:10px;color:{dim};'>{r.get('source', 'unknown')}</span>"
                f"{link}</div>"
                f"</div>"
            )
        return "".join(cards)

    left = render_items(supporting, "#1a9e84", "rgba(26,158,132,0.04)", "rgba(26,158,132,0.2)", "Supporting Evidence")
    right = render_items(counter, "#c4442c", "rgba(196,68,44,0.04)", "rgba(196,68,44,0.2)", "Counter-Evidence")

    def col_header(label, count, accent):
        return (
            f"<div style='font-family:{mono};font-size:9.5px;letter-spacing:0.16em;"
            f"text-transform:uppercase;color:{accent};margin-bottom:12px;"
            f"padding-bottom:8px;border-bottom:1px solid {border};'>"
            f"{label} <span style='color:{dim};'>({count})</span></div>"
        )

    return f"""
<div style='font-family:{mono};padding:4px;'>
    <div style='font-family:{serif};font-size:1.05rem;font-weight:600;color:{text};
                letter-spacing:0.01em;padding:12px 4px 10px;border-bottom:1px solid {border};margin-bottom:14px;'>
        Adversarial Evidence Split
    </div>
    <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;'>
        <div style='padding:14px;background:{bg2};border:1px solid {border};'>
            {col_header("Supporting Evidence", len(supporting), "#1a9e84")}
            {left}
        </div>
        <div style='padding:14px;background:{bg2};border:1px solid {border};'>
            {col_header("Counter-Evidence / Failure Cases", len(counter), "#c4442c")}
            {right}
        </div>
    </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Timeline HTML
# ─────────────────────────────────────────────────────────────────────────────

def _build_timeline_html(state_json: dict, threshold: float = 0.998) -> str:
    proposals = state_json.get("synthesizer_proposals", [])
    if not proposals:
        return "<div style='padding:16px;color:#5e6358;font-family:\"Martian Mono\",monospace;font-size:12px;'>No iterations available yet.</div>"

    points = []
    for p in proposals:
        iteration = int(p.get("iteration", 0))
        score = float(p.get("average_score", 0.0))
        points.append((iteration, max(0.0, min(1.0, score))))

    points.sort(key=lambda x: x[0])
    max_iter = max(i for i, _ in points) if len(points) > 1 else 1

    width, height = 720, 240
    left, right, top, bottom = 52, 24, 24, 40
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(i: int) -> float:
        if max_iter <= 1:
            return left + (plot_w / 2)
        return left + ((i - 1) / (max_iter - 1)) * plot_w

    def sy(s: float) -> float:
        return top + (1.0 - s) * plot_h

    polyline = " ".join(f"{sx(i):.1f},{sy(s):.1f}" for i, s in points)

    # Grid lines
    grid = ""
    for v in [0.25, 0.5, 0.75, 1.0]:
        gy = sy(v)
        grid += f"<line x1='{left}' y1='{gy:.1f}' x2='{width-right}' y2='{gy:.1f}' stroke='#222520' stroke-width='1'/>"

    x_ticks = "".join(
        f"<text x='{sx(i):.1f}' y='{height - 10}' text-anchor='middle' font-size='10' fill='#3a3d34' font-family='Martian Mono,monospace'>{i}</text>"
        for i, _ in points
    )
    y_ticks = "".join(
        f"<text x='{left - 8}' y='{sy(v) + 4:.1f}' text-anchor='end' font-size='10' fill='#3a3d34' font-family='Martian Mono,monospace'>{v:.2f}</text>"
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    thr = max(0.0, min(1.0, float(threshold)))
    thr_y = sy(thr)

    circles = "".join(
        f"<circle cx='{sx(i):.1f}' cy='{sy(s):.1f}' r='4' fill='{'#1a9e84' if s >= thr else '#c4442c'}' stroke='#0d0f0c' stroke-width='1.5'/>"
        for i, s in points
    )

    # Area fill under line
    if len(points) >= 2:
        area_pts = " ".join(f"{sx(i):.1f},{sy(s):.1f}" for i, s in points)
        first_x, last_x = sx(points[0][0]), sx(points[-1][0])
        baseline_y = sy(0.0)
        area_path = f"M{first_x:.1f},{baseline_y} " + " ".join(f"L{sx(i):.1f},{sy(s):.1f}" for i, s in points) + f" L{last_x:.1f},{baseline_y} Z"
    else:
        area_path = ""

    mono = "Martian Mono,monospace"
    serif = "Cormorant Garamond,Georgia,serif"
    text = "#e0dbd0"
    dim = "#5e6358"
    bg1 = "#141614"
    border = "#2c2f28"

    return f"""
<div style='padding:4px;font-family:{mono};'>
    <div style='font-family:{serif};font-size:1.05rem;font-weight:600;color:{text};
                letter-spacing:0.01em;padding:12px 4px 10px;border-bottom:1px solid {border};margin-bottom:14px;'>
        Consensus Timeline
    </div>
    <svg viewBox='0 0 {width} {height}' width='100%'
         style='background:{bg1};border:1px solid {border};display:block;'>

        <defs>
            <linearGradient id='areaGrad' x1='0' y1='0' x2='0' y2='1'>
                <stop offset='0%' stop-color='#c49215' stop-opacity='0.12'/>
                <stop offset='100%' stop-color='#c49215' stop-opacity='0'/>
            </linearGradient>
        </defs>

        {grid}

        <line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' stroke='#2c2f28' stroke-width='1'/>
        <line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#2c2f28' stroke-width='1'/>

        <!-- Threshold line -->
        <line x1='{left}' y1='{thr_y:.1f}' x2='{width-right}' y2='{thr_y:.1f}'
              stroke='#1a9e84' stroke-width='1.5' stroke-dasharray='5,4'/>
        <text x='{width-right-4}' y='{thr_y - 6:.1f}' text-anchor='end'
              font-size='9' fill='#1a9e84' font-family='{mono}'>τ {thr:.3f}</text>

        <!-- Area fill -->
        {f"<path d='{area_path}' fill='url(#areaGrad)'/>" if area_path else ""}

        <!-- Main line -->
        <polyline points='{polyline}' fill='none' stroke='#c49215' stroke-width='2.5'
                  stroke-linecap='round' stroke-linejoin='round'/>

        {circles}
        {x_ticks}
        {y_ticks}

        <!-- Legend -->
        <circle cx='{left+10}' cy='{top+9}' r='3.5' fill='#1a9e84'/>
        <text x='{left+18}' y='{top+13}' font-size='9' fill='#5e6358' font-family='{mono}'>At/Above Threshold</text>
        <circle cx='{left+148}' cy='{top+9}' r='3.5' fill='#c4442c'/>
        <text x='{left+156}' y='{top+13}' font-size='9' fill='#5e6358' font-family='{mono}'>Below Threshold</text>

        <!-- Axis labels -->
        <text x='{left + plot_w/2:.1f}' y='{height-3}' text-anchor='middle'
              font-size='9' fill='#3a3d34' font-family='{mono}'>Iteration</text>
        <text x='10' y='{top+14}' font-size='9' fill='#3a3d34' font-family='{mono}'>Score</text>
    </svg>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Live trace
# ─────────────────────────────────────────────────────────────────────────────

def _build_live_trace(lines: list[str]) -> str:
    if not lines:
        return "Waiting for events…"
    return "\n".join(lines[-300:])


def _build_trace_from_state(state_json: dict) -> str:
    lines: list[str] = []

    sid = state_json.get("session_id", "unknown")
    lines.append(f"[SESSION] Loaded saved session {sid}")

    support = len(state_json.get("supporting_research", []) or [])
    counter = len(state_json.get("counter_research", []) or [])
    lines.append(f"[RESEARCH] Supporting sources: {support} | Counter sources: {counter}")

    ideas = state_json.get("council_ideas", {}) or {}
    if ideas:
        lines.append(f"[BRAINSTORM] Captured ideas: {len(ideas)}")
        for idea in ideas.values():
            name = idea.get("model_name", "Unknown")
            chars = len(str(idea.get("idea", "") or ""))
            lines.append(f"[BRAINSTORM] {name} idea chars: {chars}")

    discussion = state_json.get("discussion_log", []) or []
    for entry in discussion[-80:]:
        speaker = entry.get("speaker_name", "Unknown")
        content = _clip(str(entry.get("content", "") or ""), 220)
        lines.append(f"[DISCUSSION] {speaker}: {content}")

    proposals = state_json.get("synthesizer_proposals", []) or []
    for p in proposals:
        it = p.get("iteration", "?")
        avg = float(p.get("average_score", 0.0) or 0.0)
        lines.append(f"[VOTE] Iteration {it} average score: {avg:.4f}")

    mode = state_json.get("consensus_mode", "not-run")
    reached = bool(state_json.get("consensus_reached", False))
    lines.append(f"[SESSION] consensus_mode={mode} consensus_reached={reached}")

    return _build_live_trace(lines)


def _format_council_output(text: str) -> str:
    if not text:
        return "_No council output yet._"

    s = str(text)

    # Normalize common escaped/newline variants from model output.
    s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("/n", "\n")

    # Preserve simple emphasis tags as markdown and convert line breaks.
    s = re.sub(r"<\\s*br\\s*/?\\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<\\s*(i|em)\\s*>", "*", s, flags=re.IGNORECASE)
    s = re.sub(r"<\\s*/\\s*(i|em)\\s*>", "*", s, flags=re.IGNORECASE)
    s = re.sub(r"<\\s*(b|strong)\\s*>", "**", s, flags=re.IGNORECASE)
    s = re.sub(r"<\\s*/\\s*(b|strong)\\s*>", "**", s, flags=re.IGNORECASE)

    # Drop all other HTML/CSS tags so raw markup is not shown.
    s = re.sub(r"<[^>]+>", "", s)

    def _extract_proposal_text(block: str) -> str:
        raw_block = (block or "").strip()
        if not raw_block:
            return ""
        if not (raw_block.startswith("{") and '"proposal"' in raw_block):
            return raw_block
        try:
            obj = json.loads(raw_block)
            if isinstance(obj, dict) and isinstance(obj.get("proposal"), str):
                return str(obj["proposal"]).strip()
        except Exception:
            pass
        m = re.search(
            r'"proposal"\s*:\s*"([\s\S]+?)"\s*,\s*"(?:how_it_merges_ideas|conflicts_resolved|key_improvements_over_last|rationale)"',
            raw_block,
        )
        if m:
            try:
                return bytes(m.group(1), "utf-8").decode("unicode_escape").strip()
            except Exception:
                return m.group(1).strip()
        return raw_block

    # Convert long unicode separators to markdown separators for readability.
    out_lines = []
    for line in s.splitlines():
        stripped = line.strip()
        if stripped and len(set(stripped)) == 1 and stripped[0] in {"═", "─", "-"} and len(stripped) >= 8:
            out_lines.append("---")
        else:
            out_lines.append(line.rstrip())

    formatted = "\n".join(out_lines)
    formatted = re.sub(r"\n{3,}", "\n\n", formatted).strip()

    if "COUNCIL FINAL ANSWER" in formatted:
        header_match = re.search(
            r"COUNCIL FINAL ANSWER\s*\(session:\s*([^\)]+)\)\s*Consensus score:\s*([0-9.]+)\s*\|\s*Iterations:\s*([0-9]+)\s*Consensus mode:\s*([^|\n]+)\|\s*Evidence quality:\s*([0-9.]+)",
            formatted,
            flags=re.IGNORECASE,
        )
        minority_match = re.search(
            r"THE CASE AGAINST THIS DECISION \(Minority Report\):\s*([\s\S]+?)\n---",
            formatted,
            flags=re.IGNORECASE,
        )
        proposal_match = re.search(
            r"THE CASE AGAINST THIS DECISION \(Minority Report\):[\s\S]+?\n---\n\n([\s\S]+?)\n\n---\nCouncil member final scores:",
            formatted,
            flags=re.IGNORECASE,
        )
        scores_match = re.search(
            r"Council member final scores:\n([\s\S]+?)\n---\nOriginal prompt:",
            formatted,
            flags=re.IGNORECASE,
        )

        session_line = ""
        if header_match:
            session_id, score, iters, mode, evid = header_match.groups()
            session_line = (
                f"- Session: {session_id.strip()}\n"
                f"- Consensus score: {score}\n"
                f"- Iterations: {iters}\n"
                f"- Consensus mode: {mode.strip()}\n"
                f"- Evidence quality: {evid}"
            )

        minority_text = (minority_match.group(1).strip() if minority_match else "No minority report text.")
        proposal_text = _extract_proposal_text(proposal_match.group(1).strip() if proposal_match else "")
        proposal_text = proposal_text.replace("\\n", "\n").strip()
        scores_text = (scores_match.group(1).strip() if scores_match else "")

        sections = [
            "## Council Final Answer",
            "",
            session_line or "- Session metadata unavailable",
            "",
            "## The Case Against This Decision",
            "",
            minority_text,
            "",
            "## Final Proposal",
            "",
            proposal_text or "No proposal text available.",
        ]
        if scores_text:
            sections.extend(["", "## Council Scores", "", scores_text])
        formatted = "\n".join(sections).strip()

    return formatted or "_No council output yet._"


def _derive_session_output(state_json: dict) -> str:
    """Build a useful output view even for partial/unfinished sessions."""
    final_answer = str(state_json.get("final_answer", "") or "").strip()
    if final_answer:
        return _format_council_output(final_answer)

    proposals = state_json.get("synthesizer_proposals", []) or []
    best_index = state_json.get("best_proposal_index", -1)
    if proposals:
        if isinstance(best_index, int) and 0 <= best_index < len(proposals):
            proposal = str(proposals[best_index].get("proposal", "")).strip()
            if proposal:
                header = "## Loaded Session Output (Best Available Proposal)\n\n"
                return _format_council_output(header + proposal)

        # Fallback to latest proposal if best index was never assigned.
        latest_proposal = str(proposals[-1].get("proposal", "")).strip()
        if latest_proposal:
            header = "## Loaded Session Output (Latest Proposal)\n\n"
            return _format_council_output(header + latest_proposal)

    ideas = state_json.get("council_ideas", {}) or {}
    if ideas:
        lines = [
            "## Loaded Session Output (Partial Run)",
            "",
            "This session ended before synthesis/finalization.",
            "Showing captured council ideas:",
            "",
        ]
        for entry in ideas.values():
            model_name = str(entry.get("model_name", "Unknown Model"))
            idea_text = _clip(str(entry.get("idea", "") or "").strip(), 320)
            if not idea_text:
                idea_text = "(No idea captured for this seat.)"
            lines.append(f"- **{model_name}**: {idea_text}")
        return "\n".join(lines)

    return "_No council output available in this saved session._"


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder HTML snippets
# ─────────────────────────────────────────────────────────────────────────────

def _placeholder(msg: str) -> str:
    return (
        f"<div style='padding:16px;color:#3a3d34;font-family:\"Martian Mono\",monospace;"
        f"font-size:11px;border-left:2px solid #2c2f28;'>{msg}</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run council (streaming)
# ─────────────────────────────────────────────────────────────────────────────

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
    document_ingestion: bool = False,
    deep_dive: bool = False,
    fact_check: bool = False,
    tutor_mode: bool = False,
    study_mode: str = "Research",
    subject_topic: str = "Math A - Functions",
    danish_terminology: bool = True,
    document_files=None,
    document_urls: str = "",
):
    if not prompt.strip():
        raise gr.Error("Please provide a prompt before convening the council.")

    target_state_dir = Path(state_dir).expanduser()
    target_state_dir.mkdir(parents=True, exist_ok=True)

    events: "queue.Queue[dict]" = queue.Queue()
    trace_lines: list[str] = []
    run_result: dict = {"final": None, "error": None}

    def on_progress(payload: dict) -> None:
        events.put(payload)

    # Build document list from files and URLs
    documents_to_process = []
    if document_files:
        if isinstance(document_files, list):
            documents_to_process.extend([f.name for f in document_files])
        else:
            documents_to_process.append(document_files.name)
    
    if document_urls and document_urls.strip():
        urls = [u.strip() for u in document_urls.split('\n') if u.strip()]
        documents_to_process.extend(urls)

    orchestrator = build_orchestrator_from_config(
        config_path=CONFIG_PATH,
        overrides={
            "ollama_url": ollama_url,
            "searxng_url": searxng_url,
            "search_provider": search_provider,
            "search_brave_api_key": search_brave_api_key,
            "consensus_threshold": threshold,
            "max_iterations": max_iter,
            "use_playwright": use_playwright,
            "state_dir": str(target_state_dir),
            "memory_enabled": memory_enabled,
            "memory_user_id": memory_user_id,
            "memory_agent_id": memory_agent_id,
            "memory_top_k": memory_top_k,
            "memory_ollama_url": memory_ollama_url,
            "memory_llm_model": memory_llm_model,
            "memory_embedder_model": memory_embedder_model,
            "progress_callback": on_progress,
            "document_ingestion_enabled": document_ingestion,
            "deep_dive_enabled": deep_dive,
            "fact_check_enabled": fact_check,
        },
    )
    
    # Ingest documents if provided and enabled
    if documents_to_process and document_ingestion:
        try:
            orchestrator.doc_engine.ingest_multiple(documents_to_process)
        except Exception as e:
            trace_lines.append(f"[WARNING] Document ingestion issue: {str(e)}")

    def _worker() -> None:
        try:
            run_result["final"] = orchestrator.run(prompt)
        except Exception as exc:
            run_result["error"] = str(exc)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    sessions_now = _list_session_files(target_state_dir)
    
    # Tutor mode initialization
    tutor_dashboard_html = ""
    prereq_status_md = "**Prerequisites:** Not checked yet"
    comprehension_md = "**Comprehension:** No checks performed yet"
    terminology_quiz_html = ""
    next_topic_md = "_Suggested next topic will appear after enabling tutor mode._"
    
    if tutor_mode:
        try:
            from tools.academic_tools import (
                PrerequisiteEngine, HFProgressTracker, 
                HFTerminologyManager, ComprehensionChecker,
                HF_MATH_A_CURRICULUM
            )
            
            # Initialize tutor components
            prereq_engine = PrerequisiteEngine()
            progress_tracker = HFProgressTracker(user_id=memory_user_id)
            term_manager = HFTerminologyManager(native_language="Nepali")
            
            # Map subject_topic to curriculum key
            topic_key = subject_topic.lower().replace("math a - ", "").replace(" ", "_")
            
            # Diagnose prerequisites
            diag = prereq_engine.diagnose_gaps(topic_key)
            if "error" not in diag:
                learning_path = diag.get("learning_path", [])
                prereq_status_md = f"**Learning Path for {topic_key}:**\n\n"
                for i, step in enumerate(learning_path):
                    status = "✓" if step.get("mastery_required") else "○"
                    prereq_status_md += f"{status} **{step['concept']}** ({step['level']})\n"
                
                # Generate terminology quiz
                quiz = term_manager.generate_terminology_quiz(topic_key, num_questions=3)
                terminology_quiz_html = "<h4>Danish Terminology Quiz</h4>"
                for q in quiz:
                    terminology_quiz_html += f"""
                    <div style='padding:12px;margin:8px 0;background:#1b1d18;border:1px solid #2c2f28;'>
                        <p style='color:#e0dbd0;font-family:"Martian Mono",monospace;font-size:12px;'>
                            <strong>Q:</strong> {q['question']}
                        </p>
                        <p style='color:#5e6358;font-size:11px;margin-top:6px;'>
                            <em>Hint: {q['hint']}</em>
                        </p>
                        <p style='color:#1a9e84;font-size:11px;margin-top:4px;'>
                            <strong>A:</strong> {q['answer']} ({q['pronunciation']})
                        </p>
                    </div>
                    """
            
            # Get next topic suggestion
            suggestion = progress_tracker.suggest_next_topic()
            if suggestion.get("suggested_topic"):
                next_topic_md = f"""
                **Suggested Next Topic:** {suggestion['suggested_topic']}
                
                **Reason:** {suggestion['reason']}
                
                **Prerequisites:** {suggestion['prerequisites_status']}
                """
            
            # Build tutor dashboard HTML
            tutor_dashboard_html = f"""
            <div style='font-family:"Martian Mono",monospace;padding:16px;'>
                <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px;'>
                    <div style='padding:16px;background:#1b1d18;border:1px solid #2c2f28;border-left:3px solid #1a9e84;'>
                        <div style='font-size:10px;color:#5e6358;text-transform:uppercase;'>Study Mode</div>
                        <div style='font-size:18px;color:#e0dbd0;margin-top:4px;'>{study_mode}</div>
                    </div>
                    <div style='padding:16px;background:#1b1d18;border:1px solid #2c2f28;border-left:3px solid #c49215;'>
                        <div style='font-size:10px;color:#5e6358;text-transform:uppercase;'>Topic</div>
                        <div style='font-size:18px;color:#e0dbd0;margin-top:4px;'>{topic_key}</div>
                    </div>
                    <div style='padding:16px;background:#1b1d18;border:1px solid #2c2f28;border-left:3px solid #3a78c4;'>
                        <div style='font-size:10px;color:#5e6358;text-transform:uppercase;'>Danish Terms</div>
                        <div style='font-size:18px;color:#e0dbd0;margin-top:4px;'>{"Enforced" if danish_terminology else "Disabled"}</div>
                    </div>
                </div>
            </div>
            """
            
        except Exception as e:
            prereq_status_md = f"**Error initializing tutor mode:** {str(e)}"
    
    yield (
        "Convening council session…",
        "### Session Snapshot\n- In progress",
        "{}",
        gr.Dropdown(choices=sessions_now, value=sessions_now[0] if sessions_now else None),
        _build_live_trace(trace_lines),
        _placeholder("Timeline will appear after the first synthesizer vote."),
        _placeholder("Adversarial evidence split will appear after research completes."),
        "No minority report yet.",
        _placeholder("Decision signals will appear after voting completes."),
        tutor_dashboard_html if tutor_mode else _placeholder("Enable Tutor Mode to see progress tracking."),
        prereq_status_md,
        comprehension_md,
        terminology_quiz_html if tutor_mode else _placeholder("Danish terminology quiz appears here when enabled."),
        next_topic_md,
    )

    while worker.is_alive() or not events.empty():
        try:
            event = events.get(timeout=0.25)
            stamp = event.get("timestamp", 0.0)
            stage = str(event.get("stage", "event")).upper()
            message = str(event.get("message", ""))
            trace_lines.append(f"[{stage}] {message}  ({stamp:.2f})")
            yield (
                "Convening council session…",
                "### Session Snapshot\n- In progress",
                "{}",
                gr.Dropdown(choices=sessions_now, value=sessions_now[0] if sessions_now else None),
                _build_live_trace(trace_lines),
                _placeholder("Timeline updating after each completed iteration."),
                _placeholder("Adversarial evidence split will appear after run completion."),
                "No minority report yet.",
                _placeholder("Decision signals will appear after voting completes."),
                tutor_dashboard_html if tutor_mode else _placeholder("Enable Tutor Mode to see progress tracking."),
                prereq_status_md,
                comprehension_md,
                terminology_quiz_html if tutor_mode else _placeholder("Danish terminology quiz appears here when enabled."),
                next_topic_md,
            )
        except queue.Empty:
            continue

    if run_result["error"]:
        raise gr.Error(run_result["error"])

    final_text = str(run_result["final"])

    latest = _latest_session_file(target_state_dir)
    if not latest:
        yield (
            final_text,
            "No state file found.",
            "{}",
            gr.Dropdown(choices=[]),
            _build_live_trace(trace_lines),
            _placeholder("No timeline available."),
            _placeholder("No adversarial evidence available."),
            "No minority report available.",
            _placeholder("No decision signals available."),
        )
        return

    raw = latest.read_text(encoding="utf-8")
    state_json = json.loads(raw)
    snapshot = _build_snapshot_markdown(state_json)
    sessions = _list_session_files(target_state_dir)

    yield (
        _format_council_output(final_text),
        snapshot,
        json.dumps(state_json, indent=2),
        gr.Dropdown(choices=sessions, value=latest.name),
        _build_live_trace(trace_lines),
        _build_timeline_html(state_json, threshold=threshold),
        _build_adversarial_evidence_html(state_json),
        _build_minority_report_markdown(state_json),
        _build_decision_signals_html(state_json),
        tutor_dashboard_html if tutor_mode else _placeholder("Enable Tutor Mode to see progress tracking."),
        prereq_status_md,
        comprehension_md,
        terminology_quiz_html if tutor_mode else _placeholder("Danish terminology quiz appears here when enabled."),
        next_topic_md,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load / refresh sessions
# ─────────────────────────────────────────────────────────────────────────────

def load_session(state_dir: str, session_name: str, threshold: float):
    if not session_name:
        return (
            "_No council output yet._",
            "Select a session file.",
            "{}",
            "No saved stream available.",
            _placeholder("No timeline available."),
            _placeholder("No adversarial evidence available."),
            "No minority report available.",
            _placeholder("No decision signals available."),
        )

    state_file = Path(state_dir).expanduser() / session_name
    if not state_file.exists():
        return (
            "_No council output yet._",
            "Session file not found.",
            "{}",
            "No saved stream available.",
            _placeholder("No timeline available."),
            _placeholder("No adversarial evidence available."),
            "No minority report available.",
            _placeholder("No decision signals available."),
        )

    state_json = json.loads(state_file.read_text(encoding="utf-8"))
    snapshot = _build_snapshot_markdown(state_json)
    final_answer = _derive_session_output(state_json)
    trace_text = _build_trace_from_state(state_json)
    return (
        final_answer,
        snapshot,
        json.dumps(state_json, indent=2),
        trace_text,
        _build_timeline_html(state_json, threshold=threshold),
        _build_adversarial_evidence_html(state_json),
        _build_minority_report_markdown(state_json),
        _build_decision_signals_html(state_json),
    )


def refresh_sessions(state_dir: str):
    files = _list_session_files(Path(state_dir).expanduser())
    value = files[0] if files else None
    return gr.Dropdown(choices=files, value=value)


# ─────────────────────────────────────────────────────────────────────────────
# Mem0 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_mem0_manager_from_inputs(
    memory_enabled: bool,
    memory_user_id: str,
    memory_agent_id: str,
    memory_top_k: int,
    memory_ollama_url: str,
    memory_llm_model: str,
    memory_embedder_model: str,
) -> Mem0MemoryManager:
    return Mem0MemoryManager(
        enabled=memory_enabled,
        user_id=memory_user_id.strip() or "local_user",
        agent_id=memory_agent_id.strip() or "llm_council",
        top_k=int(memory_top_k),
        ollama_base_url=memory_ollama_url.strip() or "http://localhost:11434",
        llm_model=memory_llm_model.strip() or "qwen35-9b",
        embedder_model=memory_embedder_model.strip() or "nomic-embed-text",
    )


def mem0_search(
    query: str,
    memory_enabled: bool,
    memory_user_id: str,
    memory_agent_id: str,
    memory_top_k: int,
    memory_ollama_url: str,
    memory_llm_model: str,
    memory_embedder_model: str,
):
    manager = _build_mem0_manager_from_inputs(
        memory_enabled, memory_user_id, memory_agent_id, memory_top_k,
        memory_ollama_url, memory_llm_model, memory_embedder_model,
    )
    status = manager.status()
    if not status.get("enabled"):
        return f"Mem0 disabled: {status.get('init_error') or 'disabled in settings'}"
    q = query.strip() or "user preferences and prior session outcomes"
    records = manager.search_records(q, user_id=memory_user_id, top_k=int(memory_top_k))
    if not records:
        return "No memories found for this query/user scope."
    lines = []
    for idx, r in enumerate(records, start=1):
        rid = r.get("id", "")
        score = r.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "n/a"
        lines.append(f"{idx}. id={rid or 'n/a'} score={score_text}\n{r.get('memory', '')}\n")
    return "\n".join(lines)


def mem0_add(
    content: str,
    memory_enabled: bool,
    memory_user_id: str,
    memory_agent_id: str,
    memory_top_k: int,
    memory_ollama_url: str,
    memory_llm_model: str,
    memory_embedder_model: str,
):
    if not content.strip():
        return "Provide memory text to add."
    manager = _build_mem0_manager_from_inputs(
        memory_enabled, memory_user_id, memory_agent_id, memory_top_k,
        memory_ollama_url, memory_llm_model, memory_embedder_model,
    )
    status = manager.status()
    if not status.get("enabled"):
        return f"Mem0 disabled: {status.get('init_error') or 'disabled in settings'}"
    ok = manager.add(content.strip(), user_id=memory_user_id, metadata={"kind": "manual_memory"})
    return "Memory added." if ok else "Failed to add memory."


def mem0_delete(
    memory_id: str,
    memory_enabled: bool,
    memory_user_id: str,
    memory_agent_id: str,
    memory_top_k: int,
    memory_ollama_url: str,
    memory_llm_model: str,
    memory_embedder_model: str,
):
    if not memory_id.strip():
        return "Provide a memory id to delete."
    manager = _build_mem0_manager_from_inputs(
        memory_enabled, memory_user_id, memory_agent_id, memory_top_k,
        memory_ollama_url, memory_llm_model, memory_embedder_model,
    )
    status = manager.status()
    if not status.get("enabled"):
        return f"Mem0 disabled: {status.get('init_error') or 'disabled in settings'}"
    ok = manager.delete(memory_id.strip())
    return "Memory deleted." if ok else "Delete failed. Check memory id and provider support."


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=Martian+Mono:wght@300;400;500;600&display=swap');

/* ── Tokens ──────────────────────────────────────────────────── */
:root {
    --bg:       #0d0f0c;
    --bg1:      #141614;
    --bg2:      #1b1d18;
    --bg3:      #222520;
    --border:   #2c2f28;
    --border2:  #3a3e34;
    --text:     #e0dbd0;
    --dim:      #b4bbac;
    --ghost:    #7a8174;
    --amber:    #c49215;
    --amber-d:  #7a5c0a;
    --teal:     #1a9e84;
    --red:      #c4442c;
    --blue:     #3a78c4;
    --serif:    'Cormorant Garamond', Georgia, serif;
    --mono:     'Martian Mono', 'Courier New', monospace;
}

/* ── Reset ───────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body {
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Container ───────────────────────────────────────────────── */
.gradio-container {
    font-family: var(--mono) !important;
    font-size: clamp(12px, 0.72vw, 14px) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    width: 100% !important;
    max-width: 100vw !important;
    margin: 0 auto !important;
    padding: clamp(10px, 1.1vw, 20px) !important;

    /* Gradio CSS variables */
    --body-text-color: var(--text) !important;
    --body-text-color-subdued: var(--dim) !important;
    --block-title-text-color: var(--text) !important;
    --block-label-text-color: var(--dim) !important;
    --block-info-text-color: var(--dim) !important;
    --input-placeholder-color: var(--ghost) !important;
    --panel-border-color: var(--border) !important;
    --body-background-fill: var(--bg) !important;
    --block-background-fill: var(--bg1) !important;
    --input-background-fill: var(--bg2) !important;
    --border-color-primary: var(--border) !important;
    --border-color-accent: var(--amber) !important;
}

.gradio-container > .main,
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain {
    width: 100% !important;
    max-width: 100% !important;
}

.gradio-container, .gradio-container * {
    color-scheme: dark !important;
}

/* Hide Gradio theme toggle buttons */
.gradio-container button[aria-label*="Dark"],
.gradio-container button[aria-label*="Theme"],
.gradio-container button[title*="Dark"],
.gradio-container button[title*="Theme"] {
    display: none !important;
}

/* ── Hero ────────────────────────────────────────────────────── */
#hero {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border);
    border-top: 3px solid var(--amber);
    background: var(--bg1);
    padding: 22px 28px 20px;
    margin-bottom: 14px;
}

#hero::after {
    content: "";
    position: absolute;
    right: 0; top: 0;
    width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(196,146,21,0.025));
    pointer-events: none;
}

#hero .eyebrow {
    font-family: var(--mono);
    font-size: 9.5px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 10px;
}

#hero h1 {
    margin: 0 0 8px;
    font-family: var(--serif);
    font-size: clamp(1.7rem, 2.8vw, 2.8rem);
    font-weight: 700;
    font-style: italic;
    color: var(--text);
    line-height: 1.05;
    letter-spacing: -0.01em;
}

#hero p {
    margin: 0;
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--dim);
    max-width: 840px;
    line-height: 1.65;
}

#hero .chips {
    margin-top: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

#hero .chip {
    font-family: var(--mono);
    font-size: 9.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 10px;
    border: 1px solid var(--border2);
    background: var(--bg2);
    color: var(--dim);
}

#hero .chip.lit {
    border-color: var(--amber-d);
    background: rgba(196,146,21,0.07);
    color: var(--amber);
}

/* ── Layout ──────────────────────────────────────────────────── */
.main-grid {
    display: grid !important;
    grid-template-columns: minmax(320px, 0.95fr) minmax(0, 2.05fr);
    gap: clamp(10px, 0.9vw, 16px) !important;
    align-items: stretch !important;
    width: 100% !important;
}

.main-grid > * {
    min-width: 0 !important;
}

.panel {
    background: var(--bg1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: clamp(12px, 0.9vw, 18px) !important;
}

.panel-left {
    min-height: clamp(560px, calc(100dvh - 190px), 1100px);
    width: 100% !important;
    max-width: none !important;
}

.panel-right {
    min-height: clamp(560px, calc(100dvh - 190px), 1100px);
    width: 100% !important;
    max-width: none !important;
    min-width: 0;
}

/* ── All labels ──────────────────────────────────────────────── */
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .label-wrap *,
.gradio-container .block-title,
.gradio-container .block-label {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #c0c7b8 !important;
    opacity: 1 !important;
}

/* ── Inputs ──────────────────────────────────────────────────── */
.tight textarea,
.tight input[type="text"],
.tight input[type="number"],
.tight input[type="url"] {
    font-family: var(--mono) !important;
    font-size: 12.5px !important;
    background: var(--bg2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 10px 12px !important;
    transition: border-color 0.12s ease;
}

.tight textarea:focus,
.tight input:focus {
    border-color: var(--amber-d) !important;
    outline: none !important;
    box-shadow: none !important;
    background: var(--bg3) !important;
}

.tight ::placeholder { color: var(--ghost) !important; opacity: 1 !important; }

.tight .block, .tight .wrap, .tight .form,
.tight .gr-box, .tight .gr-panel, .tight .gradio-html {
    background: var(--bg1) !important;
    border-color: var(--border) !important;
    border-radius: 0 !important;
}

/* ── Accordions ──────────────────────────────────────────────── */
.gradio-accordion {
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    background: var(--bg1) !important;
    margin-top: 8px !important;
}

.gradio-container [class*="accordion"] summary {
    background: var(--bg2) !important;
    border-radius: 0 !important;
    padding: 9px 14px !important;
    font-family: var(--mono) !important;
    font-size: 9.5px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    color: var(--dim) !important;
}

.gradio-container [class*="accordion"] summary:hover {
    background: var(--bg3) !important;
    color: var(--text) !important;
}

.gradio-container [class*="accordion"] summary * {
    color: inherit !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
button.primary {
    background: var(--amber) !important;
    color: #0d0f0c !important;
    border: none !important;
    border-radius: 0 !important;
    min-height: 44px !important;
    font-family: var(--mono) !important;
    font-size: 10.5px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    transition: background 0.12s ease, opacity 0.12s ease !important;
}

button.primary:hover { background: #d4a018 !important; }
button.primary:active { opacity: 0.85 !important; }

button.secondary {
    background: var(--bg3) !important;
    color: var(--dim) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    min-height: 44px !important;
    font-family: var(--mono) !important;
    font-size: 10.5px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    transition: all 0.12s ease !important;
}

button.secondary:hover {
    color: var(--text) !important;
    border-color: var(--amber-d) !important;
    background: var(--bg2) !important;
}

.panel .gr-button:not(.primary):not(.secondary) {
    background: var(--bg2) !important;
    color: var(--dim) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    transition: all 0.12s ease !important;
}

.panel .gr-button:not(.primary):not(.secondary):hover {
    background: var(--bg3) !important;
    border-color: var(--border2) !important;
    color: var(--text) !important;
}

/* ── Tabs ────────────────────────────────────────────────────── */
.gradio-container button[role="tab"],
.gradio-container .tab-nav button,
.gradio-container .gradio-tabs button {
    font-family: var(--mono) !important;
    font-size: 9.5px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    background: var(--bg1) !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    border-radius: 0 !important;
    color: #d6d9cf !important;
    padding: 9px 16px !important;
    transition: all 0.1s ease !important;
}

.gradio-container button[role="tab"] *,
.gradio-container .tab-nav button * {
    color: inherit !important;
}

.gradio-container button[role="tab"]:hover {
    background: var(--bg2) !important;
    color: var(--text) !important;
}

.gradio-container button[role="tab"][aria-selected="true"],
.gradio-container .tab-nav button[aria-selected="true"] {
    background: var(--bg2) !important;
    color: #ffd27a !important;
    border-top: 2px solid var(--amber) !important;
    border-bottom: none !important;
}

/* ── Scroll pane ─────────────────────────────────────────────── */
.scroll-pane {
    max-height: clamp(420px, 64dvh, 980px);
    overflow: auto;
    scrollbar-gutter: stable;
}

.scroll-pane textarea { min-height: clamp(360px, 56dvh, 900px) !important; }

.scroll-pane::-webkit-scrollbar { width: 5px; height: 5px; }
.scroll-pane::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 0; }
.scroll-pane::-webkit-scrollbar-track { background: var(--bg); }

/* ── Mono pane ───────────────────────────────────────────────── */
.mono-pane textarea {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* ── Sliders ─────────────────────────────────────────────────── */
.gradio-container [class*="slider"] span,
.gradio-container [class*="slider"] label,
.gradio-container [class*="slider"] [class*="value"],
.gradio-container [class*="slider"] [class*="tick"],
.gradio-container [class*="slider"] [class*="min"],
.gradio-container [class*="slider"] [class*="max"] {
    color: var(--dim) !important;
    font-family: var(--mono) !important;
    font-size: 10.5px !important;
    opacity: 1 !important;
}

.gradio-container [class*="slider"] input[type="range"] {
    accent-color: var(--amber) !important;
}

/* ── Checkboxes ──────────────────────────────────────────────── */
.gradio-container [class*="checkbox"] label,
.gradio-container [class*="checkbox"] span {
    color: var(--dim) !important;
    font-family: var(--mono) !important;
    font-size: 11.5px !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    opacity: 1 !important;
}

.gradio-container input[type="checkbox"] {
    accent-color: var(--amber) !important;
}

/* ── Dropdown ────────────────────────────────────────────────── */
.gradio-container select,
.gradio-container [class*="dropdown"] select {
    background: var(--bg2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}

/* ── Code ────────────────────────────────────────────────────── */
.gradio-container code,
.gradio-container pre,
.gradio-container .cm-editor,
.gradio-container .cm-content {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    border-color: var(--border) !important;
}

/* ── Markdown ────────────────────────────────────────────────── */
.gradio-container .prose,
.gradio-container .markdown-body {
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 12.5px !important;
    line-height: 1.7 !important;
    background: transparent !important;
}

.gradio-container .prose h3,
.gradio-container .markdown-body h3 {
    color: var(--text) !important;
    font-family: var(--serif) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 6px !important;
    margin: 12px 0 10px !important;
}

.gradio-container .prose li,
.gradio-container .markdown-body li {
    color: #c3cabb !important;
    margin-bottom: 3px !important;
}

.gradio-container .prose code,
.gradio-container .markdown-body code {
    background: var(--bg2) !important;
    color: var(--amber) !important;
    border-radius: 0 !important;
    padding: 1px 5px !important;
    font-size: 0.9em !important;
}

/* ── Session row ─────────────────────────────────────────────── */
.session-row {
    border-top: 1px solid var(--border);
    padding-top: 10px;
    margin-top: 10px;
}

/* ── Responsive ──────────────────────────────────────────────── */
@media (min-width: 1700px) {
    .main-grid {
        grid-template-columns: minmax(360px, 0.9fr) minmax(0, 2.1fr);
    }

    .scroll-pane {
        max-height: clamp(520px, 68dvh, 1100px);
    }
}

@media (max-width: 1400px) {
    .main-grid {
        grid-template-columns: minmax(300px, 1fr) minmax(0, 1.8fr);
    }
}

@media (max-width: 1100px) {
    .main-grid {
        grid-template-columns: 1fr !important;
    }

    .panel-left,
    .panel-right {
        min-height: auto;
        max-width: 100% !important;
        width: 100% !important;
    }

    .scroll-pane { max-height: 52dvh; }
    .scroll-pane textarea { min-height: 42dvh !important; }
    .gradio-container { padding: 10px !important; }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Build app
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="LLM Council", fill_height=True) as app:

        gr.HTML("""
        <div id="hero">
            <div class="eyebrow">Multi-Model Deliberation System</div>
            <h1>The Council</h1>
            <p>
                Adversarial reasoning across a curated stack of local models — synthesized, scored,
                and stress-tested before you receive a verdict.
            </p>
            <div class="chips">
                <span class="chip lit">Adversarial Retrieval</span>
                <span class="chip lit">Minority Dissent Report</span>
                <span class="chip lit">Consensus Scoring</span>
                <span class="chip">Mem0 Local Memory</span>
                <span class="chip">Future-You Seat</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False, elem_classes=["main-grid"]):

            # ── Left: controls ─────────────────────────────────────────────
            with gr.Column(scale=3, elem_classes=["panel", "panel-left", "tight"]):

                prompt = gr.Textbox(
                    label="Task Prompt",
                    placeholder="Put a question to the council…",
                    lines=9,
                )

                with gr.Row():
                    run_btn = gr.Button(
                        "▶  Convene Council",
                        variant="primary",
                        elem_classes=["primary"],
                    )
                    clear_btn = gr.Button(
                        "✕  Clear",
                        variant="secondary",
                        elem_classes=["secondary"],
                    )

                with gr.Accordion("Runtime Controls", open=True):
                    ollama_url = gr.Textbox(
                        label="Ollama URL",
                        value="http://localhost:11434",
                    )
                    searxng_url = gr.Textbox(
                        label="SearXNG URL",
                        value="http://localhost:8080",
                    )
                    search_provider = gr.Dropdown(
                        label="Web Search Provider",
                        choices=["searxng", "brave", "duckduckgo"],
                        value="searxng",
                    )
                    search_brave_api_key = gr.Textbox(
                        label="Brave API Key (if provider=brave)",
                        value="",
                        type="password",
                    )
                    use_playwright = gr.Checkbox(
                        label="Enable Playwright enrichment",
                        value=False,
                    )
                    
                    # Phase 1-3 Feature Toggles
                    with gr.Row():
                        document_ingestion = gr.Checkbox(
                            label="📄 Document Ingestion",
                            value=False,
                            info="Analyze PDFs, DOCX, GitHub repos"
                        )
                        deep_dive = gr.Checkbox(
                            label="🔍 Deep-Dive Research",
                            value=False,
                            info="Iterative gap-filling searches"
                        )
                        fact_check = gr.Checkbox(
                            label="✅ Fact Verification",
                            value=False,
                            info="Detect contradictions & verify claims"
                        )
                    
                    # Academic Rehabilitation / Tutor Mode Toggle
                    with gr.Row():
                        tutor_mode_enabled = gr.Checkbox(
                            label="🎓 Tutor Mode",
                            value=False,
                            info="Enable teaching loop with comprehension checks"
                        )
                        study_mode = gr.Radio(
                            choices=["Research", "Tutorial (Learn)", "Drill (Practice)", "Exam Prep (HF format)"],
                            value="Research",
                            label="Study Mode",
                            interactive=True
                        )
                    
                    with gr.Row():
                        subject_selector = gr.Dropdown(
                            choices=[
                                "Math A - Functions",
                                "Math A - Derivatives",
                                "Math A - Integrals", 
                                "Math A - Probability",
                                "Math A - Statistics",
                                "Physics B - Mechanics",
                                "Physics B - Thermodynamics"
                            ],
                            label="HF Topic",
                            value="Math A - Functions"
                        )
                        danish_terminology = gr.Checkbox(
                            label="🇩🇰 Danish Terminology",
                            value=True,
                            info="Enforce Danish academic terms"
                        )
                    
                    with gr.Row():
                        threshold = gr.Slider(
                            label="Consensus Threshold",
                            minimum=0.9,
                            maximum=1.0,
                            step=0.001,
                            value=0.998,
                        )
                        max_iter = gr.Slider(
                            label="Max Iterations",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=6,
                        )
                    state_dir = gr.Textbox(
                        label="State Directory",
                        value=str(DEFAULT_STATE_DIR),
                    )

                with gr.Accordion("Memory  ·  Mem0 OSS Local", open=False):
                    memory_enabled = gr.Checkbox(
                        label="Enable persistent memory",
                        value=True,
                    )
                    with gr.Row():
                        memory_user_id = gr.Textbox(label="User ID", value="local_user")
                        memory_agent_id = gr.Textbox(label="Agent ID", value="llm_council")
                    memory_top_k = gr.Slider(
                        label="Retrieve top-k memories",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=6,
                    )
                    memory_ollama_url = gr.Textbox(
                        label="Memory Ollama URL",
                        value="http://localhost:11434",
                    )
                    with gr.Row():
                        memory_llm_model = gr.Textbox(
                            label="Memory LLM model",
                            value="qwen35-9b",
                        )
                        memory_embedder_model = gr.Textbox(
                            label="Embedder model",
                            value="nomic-embed-text",
                        )

                with gr.Accordion("Model Stack", open=False):
                    gr.Markdown(
                        """
- **Synthesizer** — gemma4-26b
- **Council #1** — qwen3-14b *(Systematist)*
- **Council #2** — ministral-14b *(Challenger)*
- **Council #3** — phi4-mini *(Pragmatist)*
- **Researcher / Compressor** — qwen35-9b
                        """
                    )

            # ── Right: outputs ─────────────────────────────────────────────
            with gr.Column(scale=7, elem_classes=["panel", "panel-right", "tight"]):

                # Document upload section (Phase 1)
                with gr.Accordion("📁 Document Upload (Optional)", open=False):
                    gr.Markdown("Upload PDFs, DOCX, TXT files or paste GitHub repo URLs for analysis")
                    document_files = gr.File(
                        label="Upload Documents",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".rst", ".tex"],
                        file_count="multiple"
                    )
                    document_urls = gr.Textbox(
                        label="Or paste URLs (GitHub repos, web pages)",
                        placeholder="https://github.com/user/repo\nhttps://example.com/article",
                        lines=3
                    )

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

                    with gr.Tab("📚 Citations & Sources"):
                        citations_display = gr.HTML(
                            _placeholder("Citations appear after document ingestion or research."),
                            elem_classes=["scroll-pane"],
                        )

                    with gr.Tab("🔍 Research Trail"):
                        research_trail = gr.HTML(
                            _placeholder("Deep-dive research iterations appear here."),
                            elem_classes=["scroll-pane"],
                        )

                    with gr.Tab("✅ Verification Report"):
                        verification_report = gr.HTML(
                            _placeholder("Fact verification results appear after analysis."),
                            elem_classes=["scroll-pane"],
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

                    # Tutor Mode Tab - Academic Rehabilitation
                    with gr.Tab("🎓 Tutor Dashboard"):
                        gr.Markdown("### HF Math A Progress Tracker")
                        tutor_progress = gr.HTML(
                            _placeholder("Tutor mode progress appears here. Enable Tutor Mode to start tracking."),
                            elem_classes=["scroll-pane"],
                        )
                        with gr.Row():
                            prerequisite_status = gr.Markdown(
                                "**Prerequisites:** Not checked yet",
                                elem_classes=["scroll-pane"],
                            )
                            comprehension_status = gr.Markdown(
                                "**Comprehension:** No checks performed yet",
                                elem_classes=["scroll-pane"],
                            )
                        terminology_quiz = gr.HTML(
                            _placeholder("Danish terminology quiz appears here when enabled."),
                            elem_classes=["scroll-pane"],
                        )
                        next_topic_suggestion = gr.Markdown(
                            "_Suggested next topic will appear after enabling tutor mode._",
                            elem_classes=["scroll-pane"],
                        )

                # Session loader row
                with gr.Row(elem_classes=["session-row"]):
                    session_file = gr.Dropdown(
                        label="Saved Sessions",
                        choices=[],
                        scale=4,
                    )
                    refresh_btn = gr.Button("↺  Refresh", scale=1)
                    load_btn = gr.Button("↓  Load Session", scale=1, elem_classes=["primary"])

        # ── Event wiring ───────────────────────────────────────────────────
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
                document_ingestion,
                deep_dive,
                fact_check,
                tutor_mode_enabled,
                study_mode,
                subject_selector,
                danish_terminology,
                document_files,
                document_urls,
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
                tutor_progress,
                prerequisite_status,
                comprehension_status,
                terminology_quiz,
                next_topic_suggestion,
            ],
        )

        clear_btn.click(fn=lambda: "", inputs=[], outputs=[prompt])

        refresh_btn.click(
            fn=refresh_sessions,
            inputs=[state_dir],
            outputs=[session_file],
        )

        load_btn.click(
            fn=load_session,
            inputs=[state_dir, session_file, threshold],
            outputs=[
                final_answer,
                snapshot_md,
                state_json,
                live_trace,
                timeline_html,
                adversarial_evidence,
                minority_report,
                decision_signals,
            ],
        )

        mem_search_btn.click(
            fn=mem0_search,
            inputs=[
                mem_query,
                memory_enabled,
                memory_user_id,
                memory_agent_id,
                memory_top_k,
                memory_ollama_url,
                memory_llm_model,
                memory_embedder_model,
            ],
            outputs=[mem_output],
        )

        mem_add_btn.click(
            fn=mem0_add,
            inputs=[
                mem_add_text,
                memory_enabled,
                memory_user_id,
                memory_agent_id,
                memory_top_k,
                memory_ollama_url,
                memory_llm_model,
                memory_embedder_model,
            ],
            outputs=[mem_output],
        )

        mem_delete_btn.click(
            fn=mem0_delete,
            inputs=[
                mem_delete_id,
                memory_enabled,
                memory_user_id,
                memory_agent_id,
                memory_top_k,
                memory_ollama_url,
                memory_llm_model,
                memory_embedder_model,
            ],
            outputs=[mem_output],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
