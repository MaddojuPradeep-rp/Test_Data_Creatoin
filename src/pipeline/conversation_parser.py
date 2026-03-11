"""Conversation log parser — converts raw chat/email logs into structured turns.

Supports two input formats:
  1. Plain text (.txt) — raw conversation logs with USER/ASSISTANT markers.
  2. HTML (.html) — evaluation report files from the LangSmith pipeline.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)

# Role prefixes recognized in conversation logs
# Format A: "User: message" or "Assistant: message" (colon on same line)
_ROLE_COLON_RE = re.compile(
    r"^\s*(User|Agent|Bot|Assistant|Support)\s*:\s*",
    re.IGNORECASE,
)
# Format B: standalone "USER" or "ASSISTANT" on its own line (text follows on next lines)
_ROLE_STANDALONE_RE = re.compile(
    r"^\s*(USER|ASSISTANT|AGENT|BOT|SUPPORT)\s*$",
)

# Session boundary markers
_SESSION_RE = re.compile(r"^\s*Session\s+(\S+)", re.IGNORECASE)

# Metadata / noise lines to strip from raw production chat logs
_METADATA_RE = re.compile(
    r"^\s*(?:"
    r"\U0001f517|\U0001f527|\u26a0\ufe0f|\u2699\ufe0f"  # 🔗 🔧 ⚠️ ⚙️
    r"|Run\s*Id"
    r"|\u23f1\ufe0f?\s*[\d.]+s"                          # ⏱️ 7.36s
    r"|High\s+Latency"
    r"|\d+\s+tools?"
    r")\s*$",
    re.IGNORECASE,
)
# Standalone number on its own line (turn/session markers like "2", "3")
_BARE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")

# HTML / CSS junk patterns (for email cleaning)
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_CSS_RE = re.compile(
    r"@font-face\s*\{[^}]*\}"
    r"|mso-[a-z-]+:\s*[^;]+;"
    r"|/\*.*?\*/"
    r"|<!--.*?-->",
    re.DOTALL | re.IGNORECASE,
)
_EMAIL_BOILERPLATE_RE = re.compile(
    r"(?:^|\n)\s*(?:From:|To:|Cc:|Bcc:|Date:|Sent:|Subject:)\s*.*",
    re.IGNORECASE,
)
_UNSUBSCRIBE_RE = re.compile(
    r"(?:unsubscribe|manage\s+preferences|opt[\s-]?out|click\s+here\s+to)",
    re.IGNORECASE,
)
_URL_ONLY_RE = re.compile(r"^\s*https?://\S+\s*$")

# Trivial lines to skip
_TRIVIAL_LINE_RE = re.compile(
    r"^(?:ok(?:ay)?|sure|thanks?|thank\s*you|hi|hello|hey|bye"
    r"|got\s*it|alright|yep|nope|yes|no|k|yea[h]?)[.!?]*$",
    re.IGNORECASE,
)


def parse_conversations(path: Path) -> List[Dict[str, Any]]:
    """Parse a conversation log file into structured turn dicts.

    Automatically detects the file format:
      - .html → extracts conversations from evaluation report HTML
      - .txt  → parses raw text with USER/ASSISTANT markers

    Args:
        path: Path to a conversation log file (.txt or .html).

    Returns:
        List of turn dicts, each with keys:
        - session_id
        - turn_index
        - user_text
        - agent_text
        - prior_turns (list of preceding turns in the same session)
    """
    if not path.exists():
        log.warning("Conversation file not found: %s", path)
        return []

    if path.suffix.lower() in (".html", ".htm"):
        return _parse_html_report(path)

    return _parse_text_log(path)


# -----------------------------------------------------------------------
# HTML evaluation report parser
# -----------------------------------------------------------------------

# Regex to extract thread blocks from the HTML report
_THREAD_RE = re.compile(
    r'data-thread-id="([^"]+)"',
)
_TURN_SECTION_RE = re.compile(
    r'<div class="turn-text">(.*?)</div>',
    re.DOTALL,
)
_TURN_LABEL_RE = re.compile(
    r'<div class="turn-label">(.*?)</div>',
    re.DOTALL,
)
_TURN_BLOCK_RE = re.compile(
    r'<div class="turn\s+turn-(user|assistant)">\s*'
    r'<div class="turn-label">[^<]*</div>\s*'
    r'<div class="turn-text">(.*?)</div>',
    re.DOTALL,
)


def _parse_html_report(path: Path) -> List[Dict[str, Any]]:
    """Extract conversations from a LangSmith evaluation report HTML file."""
    html = path.read_text(encoding="utf-8", errors="replace")

    # Find all thread IDs
    thread_ids = []
    seen = set()
    for m in _THREAD_RE.finditer(html):
        tid = m.group(1)
        if tid not in seen:
            thread_ids.append(tid)
            seen.add(tid)

    all_turns: List[Dict[str, Any]] = []

    for tid in thread_ids:
        # Find the conversation block for this thread
        conv_id = f'id="conversation-{tid}"'
        conv_idx = html.find(conv_id)
        if conv_idx < 0:
            continue

        # Skip past the current element's opening tag to avoid matching it
        content_start = html.find(">", conv_idx + len(conv_id))
        if content_start < 0:
            continue
        content_start += 1  # move past the '>'

        # End at the next conversation block or next tab-content div
        end_idx = html.find('id="conversation-', content_start)
        if end_idx < 0:
            end_idx = min(content_start + 100_000, len(html))
        conv_chunk = html[content_start:end_idx]

        # Extract all turn blocks (user/assistant pairs)
        turn_blocks = list(_TURN_BLOCK_RE.finditer(conv_chunk))
        if not turn_blocks:
            continue

        # Group into user-assistant pairs
        prior_turns: List[Dict[str, Any]] = []
        turn_index = 0

        i = 0
        while i < len(turn_blocks):
            role = turn_blocks[i].group(1).lower()
            text = _strip_html(turn_blocks[i].group(2))

            if role == "user":
                user_text = text
                agent_text = ""
                # Look ahead for assistant response
                if i + 1 < len(turn_blocks) and turn_blocks[i + 1].group(1).lower() == "assistant":
                    agent_text = _strip_html(turn_blocks[i + 1].group(2))
                    i += 2
                else:
                    i += 1

                if not user_text:
                    continue

                turn = {
                    "session_id": tid,
                    "turn_index": turn_index,
                    "user_text": user_text,
                    "agent_text": agent_text,
                    "prior_turns": list(prior_turns),
                }
                all_turns.append(turn)
                prior_turns.append({"user_text": user_text, "agent_text": agent_text})
                turn_index += 1
            else:
                i += 1

    log.info("Parsed %d turns from %d threads in HTML report %s",
             len(all_turns), len(thread_ids), path.name)
    return all_turns


def _strip_html(text: str) -> str:
    """Strip residual HTML tags and clean whitespace from extracted text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------------------
# Plain text log parser
# -----------------------------------------------------------------------

def _parse_text_log(path: Path) -> List[Dict[str, Any]]:
    """Parse a plain-text conversation log into structured turns."""
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Clean HTML/CSS artifacts (especially in email logs)
    raw = _CSS_RE.sub("", raw)
    raw = _HTML_TAG_RE.sub(" ", raw)

    # Split into sessions
    sessions = _split_sessions(raw)

    all_turns: List[Dict[str, Any]] = []
    today = date.today().isoformat()

    for session_idx, (session_id, session_text) in enumerate(sessions):
        if not session_id:
            session_id = f"AUTO-{today}-{session_idx + 1:04d}"

        turns = _extract_turns(session_text, session_id)
        all_turns.extend(turns)

    log.info("Parsed %d turns from %d sessions in %s", len(all_turns), len(sessions), path)
    return all_turns


def _split_sessions(raw: str) -> List[tuple]:
    """Split raw text into (session_id, session_text) pairs."""
    sessions = []
    lines = raw.split("\n")
    current_id = None
    current_lines: List[str] = []

    for line in lines:
        m = _SESSION_RE.match(line)
        if m:
            # Save previous session
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sessions.append((current_id, text))
            current_id = m.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    # Save last session
    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sessions.append((current_id, text))

    # If no session markers found, try splitting on double blank lines
    if not sessions:
        blocks = re.split(r"\n\s*\n\s*\n", raw)
        for i, block in enumerate(blocks):
            block = block.strip()
            if block:
                sessions.append((None, block))

    return sessions


def _extract_turns(session_text: str, session_id: str) -> List[Dict[str, Any]]:
    """Extract user/agent turn pairs from a session."""
    lines = session_text.split("\n")
    messages: List[tuple] = []  # (role, text)

    current_role = None
    current_text_lines: List[str] = []

    for line in lines:
        # Skip metadata / noise lines
        if _METADATA_RE.match(line) or _BARE_NUMBER_RE.match(line):
            continue

        # Check Format A: "User: message" (colon style)
        m = _ROLE_COLON_RE.match(line)
        if m:
            if current_role and current_text_lines:
                text = _clean_text("\n".join(current_text_lines))
                if text:
                    messages.append((current_role.lower(), text))

            current_role = m.group(1)
            remainder = _ROLE_COLON_RE.sub("", line).strip()
            current_text_lines = [remainder] if remainder else []
            continue

        # Check Format B: standalone "USER" or "ASSISTANT" (no colon)
        m = _ROLE_STANDALONE_RE.match(line)
        if m:
            if current_role and current_text_lines:
                text = _clean_text("\n".join(current_text_lines))
                if text:
                    messages.append((current_role.lower(), text))

            current_role = m.group(1)
            current_text_lines = []
            continue

        # Regular content line
        if current_role:
            current_text_lines.append(line)

    # Save last message
    if current_role and current_text_lines:
        text = _clean_text("\n".join(current_text_lines))
        if text:
            messages.append((current_role.lower(), text))

    # Pair user + agent turns
    turns: List[Dict[str, Any]] = []
    prior_turns: List[Dict[str, Any]] = []
    turn_index = 0
    i = 0

    while i < len(messages):
        role, text = messages[i]

        if role == "user":
            user_text = text
            agent_text = ""
            # Look ahead for agent response
            if i + 1 < len(messages) and messages[i + 1][0] in ("agent", "bot", "assistant", "support"):
                agent_text = messages[i + 1][1]
                i += 2
            else:
                i += 1

            turn = {
                "session_id": session_id,
                "turn_index": turn_index,
                "user_text": user_text,
                "agent_text": agent_text,
                "prior_turns": list(prior_turns),
            }
            turns.append(turn)
            prior_turns.append({"user_text": user_text, "agent_text": agent_text})
            turn_index += 1
        else:
            # Standalone agent message without a preceding user message — skip
            i += 1

    return turns


def _clean_text(text: str) -> str:
    """Clean a text segment — strip boilerplate, URLs, whitespace."""
    # Remove email boilerplate headers
    text = _EMAIL_BOILERPLATE_RE.sub("", text)

    # Remove unsubscribe lines
    lines = text.split("\n")
    lines = [l for l in lines if not _UNSUBSCRIBE_RE.search(l)]
    lines = [l for l in lines if not _URL_ONLY_RE.match(l)]

    text = "\n".join(lines)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
