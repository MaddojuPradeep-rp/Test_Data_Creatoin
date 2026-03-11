"""NLU extraction — classifies conversation turns using GPT-4o-mini."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from ..models import TokenTracker

log = logging.getLogger(__name__)

# Thread-safe memoization cache for NLU results (keyed by utterance text)
_nlu_memo: Dict[str, Dict[str, Any]] = {}
_memo_lock = threading.Lock()


def clear_nlu_memo() -> None:
    """Clear the NLU memoization cache (call between runs)."""
    global _nlu_memo
    with _memo_lock:
        _nlu_memo.clear()


def extract_nlu(
    client: Any,
    model: str,
    user_text: str,
    taxonomy: Dict[str, Any],
    agent_text: str = "",
) -> Dict[str, Any]:
    """Classify a user utterance using GPT-4o-mini.

    Returns a dict with keys:
    - intent: str (domain-level intent from taxonomy or NEW_CamelCase)
    - entities: dict (extracted key-value pairs)
    - language: str (ISO 639-1)
    - sentiment: str (positive/neutral/negative)
    - urgency: str (low/medium/high)
    - confidence: float (0.0-1.0)
    - flags: list[str] (detected patterns)

    Results are memoized per utterance to avoid duplicate API calls.
    """
    # Check memo cache first (thread-safe)
    cache_key = user_text.strip().lower()
    with _memo_lock:
        if cache_key in _nlu_memo:
            TokenTracker.instance().record_nlu_cache_hit()
            return dict(_nlu_memo[cache_key])

    # Build intent list for the prompt
    known_intents = taxonomy.get("intents", [])
    sub_intents = taxonomy.get("sub_intents", {})
    known_flags = taxonomy.get("flags", [])
    known_entities = list(taxonomy.get("entities", {}).keys())

    intent_descriptions = []
    for intent in known_intents:
        subs = sub_intents.get(intent, [])
        if subs:
            intent_descriptions.append(f"- {intent} (sub-intents: {', '.join(subs)})")
        else:
            intent_descriptions.append(f"- {intent}")

    system_prompt = f"""You are an NLU classifier for a property management chatbot.

Classify the user's message into one of these intents:
{chr(10).join(intent_descriptions)}

If the message doesn't fit any known intent, create a NEW_ intent in CamelCase format (e.g., NEW_RentIncrease).

For sub-intents, always map to the parent intent (e.g., "plumbing" → "service_request").

Extract entities as key-value pairs. Normalize:
- Dates → ISO-8601 format
- Amounts → numeric values
- Known entity types: {', '.join(known_entities)}

Detect these flags if applicable: {', '.join(known_flags)}

Respond with valid JSON only:
{{
  "intent": "<intent_name>",
  "entities": {{}},
  "language": "<ISO 639-1 code>",
  "sentiment": "positive|neutral|negative",
  "urgency": "low|medium|high",
  "confidence": <0.0-1.0>,
  "flags": []
}}"""

    user_prompt = f"User message: {user_text}"
    if agent_text:
        user_prompt += f"\n\nAgent response (for context): {agent_text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        TokenTracker.instance().add_chat(response.usage, "nlu")

        content = response.choices[0].message.content
        result = json.loads(content)

        # Ensure all expected fields exist with defaults
        result.setdefault("intent", "unknown")
        result.setdefault("entities", {})
        result.setdefault("language", "en")
        result.setdefault("sentiment", "neutral")
        result.setdefault("urgency", "low")
        result.setdefault("confidence", 0.5)
        result.setdefault("flags", [])

        # Normalize intent — map sub-intents to parent
        intent_lower = result["intent"].lower()
        if not intent_lower.startswith("new_"):
            for parent, subs in sub_intents.items():
                if intent_lower in [s.lower() for s in subs]:
                    result["intent"] = parent
                    break

        # Store in memo cache
        with _memo_lock:
            _nlu_memo[cache_key] = dict(result)

        return result

    except Exception as exc:
        log.warning("NLU extraction failed for '%s...': %s", user_text[:50], exc)
        return {
            "intent": "unknown",
            "entities": {},
            "language": "en",
            "sentiment": "neutral",
            "urgency": "low",
            "confidence": 0.0,
            "flags": [],
        }
