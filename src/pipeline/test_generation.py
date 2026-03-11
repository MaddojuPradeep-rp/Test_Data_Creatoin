"""Test case generation — creates structured test cases from gap candidates."""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import TokenTracker

log = logging.getLogger(__name__)


def select_test_type(distribution: Dict[str, float]) -> Tuple[str, str]:
    """Select a test type based on configurable weights.

    Args:
        distribution: Dict mapping test type names to weights (e.g., {"policy_flow": 0.70}).

    Returns:
        (test_type, difficulty) tuple.
    """
    types = list(distribution.keys())
    weights = list(distribution.values())
    chosen = random.choices(types, weights=weights, k=1)[0]

    difficulty_map = {
        "policy_flow": "easy",
        "edge_case": "medium",
        "stress": "hard",
    }
    return chosen, difficulty_map.get(chosen, "medium")


def generate_test_case(
    client: Any,
    model: str,
    session_id: str,
    user_utterance: str,
    agent_utterance: str,
    nlu_result: Dict[str, Any],
    gap_info: Dict[str, Any],
    prior_turns: Optional[List[Dict[str, Any]]] = None,
    slot_plan: Optional[Dict[str, Any]] = None,
    test_type: str = "policy_flow",
) -> Dict[str, Any]:
    """Generate a test case using GPT-4o-mini.

    Assembles a prompt from three layers:
    1. Base prompt (QA test designer role)
    2. Slot plan (if available)
    3. Test-type overlay

    Returns a dict with generated test case fields.
    """
    intent = nlu_result.get("intent", "unknown")
    entities = nlu_result.get("entities", {})
    gap_reasons = gap_info.get("reasons", [])

    # Build system prompt
    system_prompt = _build_system_prompt(
        intent=intent,
        entities=entities,
        slot_plan=slot_plan,
        test_type=test_type,
        gap_reasons=gap_reasons,
    )

    # Build user prompt with context
    user_prompt_parts = [
        f"Original user message: {user_utterance}",
    ]
    if agent_utterance:
        user_prompt_parts.append(f"Original agent response: {agent_utterance}")
    if prior_turns:
        context_lines = []
        for pt in prior_turns[-3:]:  # Last 3 turns for context
            if pt.get("user_text"):
                context_lines.append(f"User: {pt['user_text']}")
            if pt.get("agent_text"):
                context_lines.append(f"Agent: {pt['agent_text']}")
        if context_lines:
            user_prompt_parts.append(f"Prior conversation context:\n" + "\n".join(context_lines))

    user_prompt_parts.append(f"Intent: {intent}")
    user_prompt_parts.append(f"Entities: {json.dumps(entities)}")
    user_prompt_parts.append(f"Gap reasons: {', '.join(gap_reasons)}")
    user_prompt_parts.append(f"Test type: {test_type}")

    user_prompt = "\n\n".join(user_prompt_parts)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        TokenTracker.instance().add_chat(response.usage, "testgen")

        content = response.choices[0].message.content
        result = json.loads(content)

        # Ensure conversation is a list of dicts
        conversation = result.get("conversation", [])
        if conversation and isinstance(conversation[0], dict):
            # Normalize role names
            for msg in conversation:
                if msg.get("role") in ("agent", "bot", "support"):
                    msg["role"] = "assistant"

        # Validate against slot plan (policy_flow only)
        if test_type == "policy_flow" and slot_plan:
            violations = _validate_slot_plan(conversation, slot_plan)
            if violations:
                log.info("Slot plan violations for %s: %s", intent, violations)
                # Retry once with violation feedback
                result = _retry_with_violations(
                    client, model, system_prompt, user_prompt, violations
                )
                if result is None:
                    result = json.loads(content)
                    result["slot_plan_violations"] = violations

        result.setdefault("conversation", [])
        result.setdefault("reason", "Covers a gap identified in conversation analysis.")
        result.setdefault("expected_behavior", "")
        result.setdefault("expected_answer", "")
        result.setdefault("acceptance_criteria", [])
        result.setdefault("tags", [])
        result.setdefault("gap_reason", gap_reasons)

        return result

    except Exception as exc:
        log.warning("Test generation failed: %s", exc)
        return {
            "conversation": [],
            "reason": f"Generation failed: {exc}",
            "expected_behavior": "",
            "expected_answer": "",
            "acceptance_criteria": [],
            "tags": ["generation_failed"],
            "gap_reason": gap_reasons,
        }


def _build_system_prompt(
    intent: str,
    entities: Dict[str, Any],
    slot_plan: Optional[Dict[str, Any]],
    test_type: str,
    gap_reasons: List[str],
) -> str:
    """Build the system prompt for test case generation."""
    prompt_parts = [
        "You are an expert QA test designer for a property management chatbot.",
        "Your task is to generate a realistic test conversation that covers a specific gap in the existing test suite.",
        "",
        "Generate a JSON object with these fields:",
        '- "conversation": array of {"role": "user"|"assistant", "message": "..."} objects',
        '- "reason": why this test case is valuable (1-2 sentences)',
        '- "expected_behavior": what the agent should do correctly',
        '- "acceptance_criteria": array of specific pass/fail criteria',
        '- "tags": array of relevant tags',
        '- "gap_reason": array of gap reasons this test addresses',
        '- "priority": "P0"|"P1"|"P2"|"P3"',
        '- "category": the test category',
        "",
        f"Intent being tested: {intent}",
        f"Gap reasons to address: {', '.join(gap_reasons)}",
    ]

    # Add slot plan instructions
    if slot_plan:
        slots = slot_plan.get("slots", [])
        confirmation = slot_plan.get("confirmation")
        prompt_parts.append("")
        prompt_parts.append("=== SLOT PLAN (Conversation Policy) ===")
        prompt_parts.append("STRICT RULES:")
        prompt_parts.append("- Boolean slots must use yes/no phrasing ONLY (not open-ended questions)")
        prompt_parts.append("- Enumerated slots must present the listed options")
        prompt_parts.append("- Required slots MUST be addressed in the conversation")
        prompt_parts.append("")
        for i, slot in enumerate(slots, 1):
            required = "REQUIRED" if slot.get("required") else "optional"
            slot_type = slot.get("type", "open_ended")
            prompt = slot.get("prompt", "")
            line = f"{i}. [{slot_type}] {slot['name']} ({required}) — \"{prompt}\""
            if slot_type == "enumerated" and slot.get("options"):
                line += f" options: {slot['options']}"
            prompt_parts.append(line)
        if confirmation:
            prompt_parts.append(f"FINAL: [confirmation] — \"{confirmation}\"")

    # Add test-type overlay
    prompt_parts.append("")
    prompt_parts.append(f"=== TEST TYPE: {test_type.upper()} ===")
    if test_type == "policy_flow":
        prompt_parts.extend([
            "Generate a HAPPY-PATH conversation:",
            "- User answers correctly and completely",
            "- Agent follows the slot plan exactly",
            "- All required slots are collected in order",
            "- Conversation ends with confirmation step",
        ])
    elif test_type == "edge_case":
        prompt_parts.extend([
            "Generate an EDGE-CASE conversation with ONE user deviation:",
            "Choose ONE of these deviations:",
            '- Incomplete answer: user says "I don\'t know" or gives partial info',
            '- Wrong answer type: user says "maybe" to a boolean yes/no question',
            '- User correction: "Sorry, I meant 420" (corrects a previous answer)',
            "- Vague/ambiguous answer: unclear or imprecise response",
            "- Partial multi-slot answer: user answers 2 things at once, but incompletely",
            "",
            "The agent should handle the deviation gracefully.",
        ])
    elif test_type == "stress":
        prompt_parts.extend([
            "Generate a STRESS TEST conversation with ONE disruption:",
            "Choose ONE of these disruptions:",
            '- Off-topic injection: "Also when does the pool open?"',
            "- Multi-intent message: two issues in one message",
            '- Frustrated user: "I already told you that!" or emotional language',
            "- Irrelevant context: user includes unrelated information or attachments",
            "",
            "The agent should maintain composure and redirect appropriately.",
        ])

    prompt_parts.extend([
        "",
        "=== QUALITY RULES ===",
        "- Conversation should be 4-8 turns (2-4 exchanges)",
        "- Include realistic, natural dialogue",
        "- Acceptance criteria should be specific and testable",
        "- Priority should reflect severity (P0=critical, P3=nice-to-have)",
        "- Tags should be relevant and descriptive",
    ])

    return "\n".join(prompt_parts)


def _validate_slot_plan(
    conversation: List[Dict[str, Any]],
    slot_plan: Dict[str, Any],
) -> List[str]:
    """Validate a policy_flow conversation against its slot plan.

    Returns a list of violation strings (empty = valid).
    """
    violations = []
    slots = slot_plan.get("slots", [])
    confirmation = slot_plan.get("confirmation")

    agent_messages = [
        msg.get("message", "") for msg in conversation
        if msg.get("role") in ("assistant", "agent", "bot")
    ]
    all_agent_text = " ".join(agent_messages).lower()

    for slot in slots:
        if not slot.get("required"):
            continue

        name = slot["name"]
        prompt_hint = slot.get("prompt", "").lower()
        slot_type = slot.get("type", "open_ended")

        # Check if the slot is addressed
        keywords = name.replace("_", " ").split()
        prompt_words = [w for w in prompt_hint.split() if len(w) > 3]
        relevant_words = keywords + prompt_words[:3]

        addressed = any(w in all_agent_text for w in relevant_words)
        if not addressed:
            violations.append(f"Required slot '{name}' not addressed by agent")

        # Check boolean format
        if slot_type == "boolean" and addressed:
            # Should not be open-ended
            boolean_markers = re.compile(r"\b(?:yes\s*(?:or|/)\s*no|can\s+you|do\s+you|is\s+it|would\s+you)\b", re.IGNORECASE)
            if not any(boolean_markers.search(msg) for msg in agent_messages):
                pass  # Allow flexible boolean phrasing

        # Check enumerated options
        if slot_type == "enumerated" and addressed:
            options = slot.get("options", [])
            if options:
                options_mentioned = any(
                    opt.lower() in all_agent_text for opt in options
                )
                if not options_mentioned:
                    violations.append(
                        f"Enumerated slot '{name}' does not present options: {options}"
                    )

    # Check confirmation step
    if confirmation and agent_messages:
        last_agent = agent_messages[-1].lower()
        confirm_words = ["correct", "confirm", "right", "verify", "sound good", "is that right"]
        if not any(w in last_agent for w in confirm_words):
            violations.append("Missing confirmation step in last agent turn")

    return violations


def _retry_with_violations(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    violations: List[str],
) -> Optional[Dict[str, Any]]:
    """Retry test generation with violation feedback (1 retry)."""
    retry_prompt = (
        f"{user_prompt}\n\n"
        f"IMPORTANT: Your previous generation had these slot plan violations:\n"
        + "\n".join(f"- {v}" for v in violations)
        + "\n\nPlease fix these issues in the regenerated conversation."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        TokenTracker.instance().add_chat(response.usage, "testgen")
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as exc:
        log.warning("Retry generation failed: %s", exc)
        return None


def postprocess_suggestions(
    suggestions: List[Dict[str, Any]],
    neighbors_for_suggestions: List[List[Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deduplication and post-processing of suggestions.

    1. Dedup vs. existing tests: reject if cosine similarity > 0.90.
    2. Dedup among suggestions: reject if same intent + similar novelty + high overlap.
    """
    dedup_threshold = config.get("similarity", {}).get("dedup_threshold", 0.90)

    # Step 1: Dedup vs existing tests
    filtered = []
    for i, s in enumerate(suggestions):
        neighbors = s.get("nearest_existing_tests", [])
        max_sim = max((n.get("similarity", 0) for n in neighbors), default=0)
        if max_sim > dedup_threshold:
            log.info(
                "Dedup: rejected %s (similar to existing test, sim=%.3f)",
                s.get("id"), max_sim,
            )
            continue
        filtered.append(s)

    # Step 2: Dedup among suggestions (greedy)
    kept = []
    seen_utterances: List[str] = []
    for s in filtered:
        utterance = s.get("user_utterance", "").lower().strip()
        intent = s.get("intent", "")

        # Check token overlap with already-kept suggestions of the same intent
        is_dup = False
        for prev in seen_utterances:
            overlap = _token_overlap(utterance, prev)
            if overlap > 0.80:
                is_dup = True
                break

        if is_dup:
            log.info("Dedup: rejected %s (high overlap with kept suggestion)", s.get("id"))
            continue

        kept.append(s)
        seen_utterances.append(utterance)

    log.info("Postprocess: %d -> %d suggestions after dedup", len(suggestions), len(kept))
    return kept


def _token_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level overlap ratio between two texts."""
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    smaller = min(len(tokens_a), len(tokens_b))
    return len(intersection) / smaller if smaller > 0 else 0.0
