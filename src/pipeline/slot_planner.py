"""Slot-plan builder for policy-driven test generation.

Builds a conversation policy (slot plan) that defines:
- What questions the agent must ask
- In what order
- What answer types to expect (boolean, enumerated, open_ended, confirmation)

Known intents use static policies from data/slot_policies.json.
NEW_* intents get a dynamically generated plan via LLM (cached per-run).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import TokenTracker

log = logging.getLogger(__name__)

# Thread-safe cache for dynamically generated slot plans (per-run)
_plan_cache: Dict[str, Dict[str, Any]] = {}
_plan_lock = threading.Lock()


def clear_plan_cache() -> None:
    """Clear the dynamic slot plan cache (call between runs)."""
    global _plan_cache
    with _plan_lock:
        _plan_cache.clear()


def build_slot_plan(
    intent: str,
    policy_path: Path,
    client: Any = None,
    model: str = "gpt-4o-mini",
    user_utterance: str = "",
    entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build or retrieve a slot plan for the given intent.

    For known intents: loads from slot_policies.json.
    For NEW_* intents: generates via LLM (cached per-run, thread-safe).

    Returns a dict with:
    - slots: list of slot definitions
    - confirmation: confirmation template string (or None)
    """
    intent_lower = intent.lower()

    # Try static policy first
    static_plan = _load_static_policy(intent_lower, policy_path)
    if static_plan is not None:
        return static_plan

    # Check dynamic plan cache
    with _plan_lock:
        if intent_lower in _plan_cache:
            return dict(_plan_cache[intent_lower])

    # Generate dynamic plan via LLM
    if client is None:
        return _default_plan(intent)

    dynamic_plan = _generate_dynamic_plan(
        client, model, intent, user_utterance, entities or {}
    )

    # Cache the plan
    with _plan_lock:
        _plan_cache[intent_lower] = dynamic_plan

    return dynamic_plan


def _load_static_policy(intent: str, policy_path: Path) -> Optional[Dict[str, Any]]:
    """Load a static slot policy from the JSON file."""
    if not policy_path.exists():
        return None

    try:
        with policy_path.open("r", encoding="utf-8") as f:
            policies = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load slot policies from %s: %s", policy_path, exc)
        return None

    # Try exact match, then without NEW_ prefix
    policy = policies.get(intent)
    if policy is None and intent.startswith("new_"):
        policy = policies.get(intent[4:])

    return policy


def _generate_dynamic_plan(
    client: Any,
    model: str,
    intent: str,
    user_utterance: str,
    entities: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a slot plan for a NEW_* intent using LLM."""
    system_prompt = """You are a conversation policy designer for a property management chatbot.

Given a new intent, design a slot plan that defines what information the agent needs to collect.

Rules:
- Generate 1-6 slots (be realistic, not all intents need many slots)
- Order slots logically
- Each slot must have: name, type, required (bool), prompt
- Slot types: "boolean" (yes/no only), "enumerated" (fixed options), "open_ended" (free text), "confirmation" (final step)
- At least 1 slot should be required
- Include a confirmation step at the end

Respond with valid JSON:
{
  "slots": [
    {"name": "...", "type": "boolean|enumerated|open_ended", "required": true/false, "prompt": "...", "options": [...]}
  ],
  "confirmation": "I'll ... {slot_name} ... Correct?"
}"""

    user_prompt = (
        f"Intent: {intent}\n"
        f"Example user message: {user_utterance}\n"
        f"Entities detected: {json.dumps(entities)}\n"
        f"\nDesign a slot plan for this intent."
    )

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
        plan = json.loads(content)
        plan.setdefault("slots", [])
        plan.setdefault("confirmation", None)
        return plan

    except Exception as exc:
        log.warning("Dynamic slot plan generation failed for '%s': %s", intent, exc)
        return _default_plan(intent)


def _default_plan(intent: str) -> Dict[str, Any]:
    """Return a minimal default slot plan."""
    return {
        "slots": [
            {
                "name": "description",
                "type": "open_ended",
                "required": True,
                "prompt": "Can you tell me more about what you need?",
            },
            {
                "name": "unit_number",
                "type": "open_ended",
                "required": True,
                "prompt": "What is your unit number?",
            },
        ],
        "confirmation": "I'll help you with your request. Is that correct?",
    }
