"""Gap analysis — novelty/impact scoring and gap detection."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


def build_search_text(nlu_result: Dict[str, Any], user_text: str, agent_text: str = "") -> str:
    """Build a search text combining NLU info with the raw utterance for embedding search."""
    intent = nlu_result.get("intent", "")
    entities = nlu_result.get("entities", {})
    entity_str = " ".join(f"{k}:{v}" for k, v in entities.items()) if entities else ""

    parts = [user_text]
    if intent and intent != "unknown":
        parts.append(f"intent:{intent}")
    if entity_str:
        parts.append(entity_str)
    if agent_text:
        parts.append(agent_text[:200])

    return " ".join(parts)


def analyze_gap(
    index: Any,
    nlu: Dict[str, Any],
    user_text: str,
    config: Dict[str, Any],
    known_entity_combos: Set[Tuple[str, ...]],
    agent_text: str = "",
    precomputed_neighbors: Optional[List[tuple]] = None,
    prior_turns: Optional[List[Dict[str, Any]]] = None,
    known_intent_channel: Optional[Set[Tuple[str, str]]] = None,
    known_intent_policy: Optional[Set[Tuple[str, str]]] = None,
    known_intent_language: Optional[Set[Tuple[str, str]]] = None,
    known_intent_escalation: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Analyze whether a conversation turn represents a gap in the test set.

    Returns a dict with:
    - is_candidate: bool
    - reasons: list of gap reason strings
    - neighbors: list of neighbor info dicts
    - max_similarity: float
    - avg_similarity: float
    - novelty_score: float (0.0-1.0)
    - impact_score: float
    - is_paraphrase: bool
    - has_structural_novelty: bool
    - constraint_count: int
    - multi_turn_patterns: list of pattern names
    - unseen_combos: list of combo descriptions
    - newness_pct: int (0-100)
    - newness_label: str
    """
    novelty_cfg = config.get("novelty", {})
    scoring_cfg = config.get("scoring", {})
    multi_turn_cfg = config.get("multi_turn", {})

    novelty_max_weight = novelty_cfg.get("novelty_max_weight", 0.6)
    novelty_avg_weight = novelty_cfg.get("novelty_avg_weight", 0.4)
    sim_threshold = config.get("similarity", {}).get("threshold_small", 0.70)
    paraphrase_ceiling = novelty_cfg.get("paraphrase_novelty_ceiling", 0.50)

    intent = nlu.get("intent", "unknown").lower()
    entities = nlu.get("entities", {})
    flags = nlu.get("flags", [])
    language = nlu.get("language", "en")
    channel = config.get("_current_channel", "chat")

    # Get neighbors (either precomputed or from index search)
    if precomputed_neighbors is not None:
        neighbors_raw = precomputed_neighbors
    else:
        search_text = build_search_text(nlu, user_text, agent_text)
        top_k = config.get("similarity", {}).get("top_k", 5)
        neighbors_raw = index.search(search_text, top_k)

    # Parse neighbor results
    neighbors = []
    similarities = []
    for n in neighbors_raw:
        if isinstance(n, tuple) and len(n) >= 2:
            sim, nid = n[0], n[1]
            meta = n[2] if len(n) > 2 else {}
            sim_pct = f"{sim * 100:.1f}%"
            neighbors.append({
                "id": nid,
                "similarity": round(sim, 4),
                "similarity_pct": sim_pct,
                "intent": meta.get("intent", ""),
            })
            similarities.append(sim)

    max_sim = max(similarities) if similarities else 0.0
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

    # Compute novelty score
    novelty = novelty_max_weight * (1 - max_sim) + novelty_avg_weight * (1 - avg_sim)
    novelty = max(0.0, min(1.0, novelty))

    # Compute newness percentage and label
    newness_pct = int(novelty * 100)
    if newness_pct >= 75:
        newness_label = "Highly Novel"
    elif newness_pct >= 50:
        newness_label = "Moderately Novel"
    elif newness_pct >= 25:
        newness_label = "Slightly Novel"
    else:
        newness_label = "Very Similar"

    # --- Gap reasons ---
    reasons: List[str] = []
    unseen_combos: List[str] = []

    # 1. Low similarity
    if max_sim < sim_threshold:
        reasons.append("low_similarity")

    # 2. Unseen entity combo
    if entities:
        entity_combo = tuple(sorted(entities.keys()))
        if entity_combo not in known_entity_combos:
            reasons.append("unseen_entity_combo")

    # 3. Combinatorial novelty
    if known_intent_channel is not None and (intent, channel) not in known_intent_channel:
        reasons.append("combinatorial_novelty")
        unseen_combos.append(f"intent+channel({intent},{channel})")

    if known_intent_policy is not None and "policy" in flags and (intent, "policy") not in known_intent_policy:
        reasons.append("combinatorial_novelty_policy")
        unseen_combos.append(f"intent+policy({intent})")

    if known_intent_language is not None and language != "en" and (intent, language) not in known_intent_language:
        reasons.append("combinatorial_novelty_language")
        unseen_combos.append(f"intent+language({intent},{language})")

    if known_intent_escalation is not None and "agent_failure" in flags and (intent, "escalation") not in known_intent_escalation:
        reasons.append("combinatorial_novelty_escalation")
        unseen_combos.append(f"intent+escalation({intent})")

    # 4. Policy sensitivity
    if "policy" in flags:
        reasons.append("policy_sensitive")

    # 5. Multi-intent
    if "multi_intent" in flags:
        reasons.append("multi_intent")

    # 6. Agent failure
    if "agent_failure" in flags:
        reasons.append("agent_failure")

    # 7. Edge-case heuristics
    edge_cases = _detect_edge_cases(user_text, nlu)
    if edge_cases:
        reasons.extend(edge_cases)

    # 8. High constraint density
    constraint_count = _count_constraints(nlu, flags, language, prior_turns)
    constraint_threshold = scoring_cfg.get("constraint_density_threshold", 4)
    if constraint_count >= constraint_threshold:
        reasons.append("high_constraint_density")

    # 9. Multi-turn structural patterns
    multi_turn_patterns: List[str] = []
    if prior_turns:
        multi_turn_patterns = _detect_multi_turn_patterns(
            prior_turns, user_text, agent_text, nlu, multi_turn_cfg
        )
        if multi_turn_patterns:
            reasons.append("multi_turn_structural")

    # --- Impact score ---
    impact = 0.0
    if "policy" in flags:
        impact += scoring_cfg.get("impact_policy", 0.2)
    if "agent_failure" in flags:
        impact += scoring_cfg.get("impact_failure", 0.2)
    if "unseen_entity_combo" in reasons:
        impact += scoring_cfg.get("impact_new_entity_combo", 0.1)
    if language != "en":
        impact += scoring_cfg.get("impact_language_variant", 0.1)
    if any(c.startswith("intent+channel") for c in unseen_combos):
        impact += scoring_cfg.get("impact_new_pair", 0.15)
    # Triple bonus: new intent+channel+policy
    if (
        any(c.startswith("intent+channel") for c in unseen_combos)
        and any(c.startswith("intent+policy") for c in unseen_combos)
    ):
        impact += scoring_cfg.get("impact_new_triple", 0.25)
    if constraint_count >= constraint_threshold:
        impact += scoring_cfg.get("impact_constraint_density", 0.15)

    # --- Structural novelty ---
    has_structural_novelty = bool(
        reasons
        and any(
            r not in ("low_similarity",) for r in reasons
        )
    )
    if multi_turn_patterns:
        has_structural_novelty = True

    # --- Paraphrase detection ---
    is_paraphrase = False
    if (
        not has_structural_novelty
        and novelty < paraphrase_ceiling
        and len(reasons) <= 1
        and "low_similarity" not in reasons
    ):
        # Same intent + same entities + low novelty + no structural novelty → paraphrase
        if neighbors and intent:
            same_intent_neighbors = [
                n for n in neighbors if n.get("intent", "").lower() == intent
            ]
            if same_intent_neighbors:
                is_paraphrase = True

    is_candidate = bool(reasons) and not is_paraphrase

    return {
        "is_candidate": is_candidate,
        "reasons": reasons,
        "neighbors": neighbors,
        "max_similarity": round(max_sim, 4),
        "avg_similarity": round(avg_sim, 4),
        "novelty_score": round(novelty, 4),
        "impact_score": round(impact, 4),
        "is_paraphrase": is_paraphrase,
        "has_structural_novelty": has_structural_novelty,
        "constraint_count": constraint_count,
        "multi_turn_patterns": multi_turn_patterns,
        "unseen_combos": unseen_combos,
        "newness_pct": newness_pct,
        "newness_label": newness_label,
    }


def _detect_edge_cases(user_text: str, nlu: Dict[str, Any]) -> List[str]:
    """Detect edge-case heuristics: numeric extremes, date boundaries, code-switching."""
    edge_cases = []

    # Numeric extremes
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", user_text)
    for n_str in numbers:
        try:
            n = float(n_str)
            if n > 100000 or n == 0:
                edge_cases.append("numeric_extreme")
                break
        except ValueError:
            pass

    # Date boundaries
    if re.search(r"\b(?:12/31|01/01|02/29|leap\s*year|end\s*of\s*month|year[\s-]?end)\b", user_text, re.IGNORECASE):
        edge_cases.append("date_boundary")

    # Code-switching (detected by NLU flags)
    if "code_switching" in nlu.get("flags", []):
        edge_cases.append("code_switching")

    return edge_cases


def _count_constraints(
    nlu: Dict[str, Any],
    flags: List[str],
    language: str,
    prior_turns: Optional[List[Dict[str, Any]]],
) -> int:
    """Count the number of constraints in a turn (entities, flags, multi-turn, language)."""
    count = 0
    count += len(nlu.get("entities", {}))
    count += len([f for f in flags if f in ("policy", "multi_intent", "agent_failure")])
    if language != "en":
        count += 1
    if prior_turns and len(prior_turns) > 0:
        count += 1
    return count


def _detect_multi_turn_patterns(
    prior_turns: List[Dict[str, Any]],
    user_text: str,
    agent_text: str,
    nlu: Dict[str, Any],
    multi_turn_cfg: Dict[str, Any],
) -> List[str]:
    """Detect multi-turn structural patterns in the conversation."""
    patterns = []

    if not prior_turns:
        return patterns

    # Escalation transition: user repeats issue or demands escalation
    if multi_turn_cfg.get("detect_escalation", True):
        escalation_words = re.compile(
            r"\b(?:manager|supervisor|escalat|transfer|speak\s+to|complain|unacceptable)\b",
            re.IGNORECASE,
        )
        if escalation_words.search(user_text):
            patterns.append("escalation_transition")

    # Agent reversal: agent contradicts a previous statement
    if multi_turn_cfg.get("detect_agent_reversal", True):
        if len(prior_turns) >= 2:
            prev_agent = prior_turns[-1].get("agent_text", "")
            reversal_words = re.compile(
                r"\b(?:actually|correction|I\s+was\s+wrong|let\s+me\s+correct|sorry.*mistake)\b",
                re.IGNORECASE,
            )
            if agent_text and reversal_words.search(agent_text):
                patterns.append("agent_reversal")

    # User restatement: user repeats something they already said
    if multi_turn_cfg.get("detect_user_restatement", True):
        for pt in prior_turns:
            prev_user = pt.get("user_text", "").lower()
            if prev_user and _text_overlap(prev_user, user_text.lower()) > 0.6:
                patterns.append("user_restatement")
                break

    # State correction: user corrects a detail from a prior turn
    if multi_turn_cfg.get("detect_state_change", True):
        correction_re = re.compile(
            r"\b(?:sorry.*meant|actually.*not|correction|I\s+meant|wrong\s+(?:unit|number|date|address))\b",
            re.IGNORECASE,
        )
        if correction_re.search(user_text):
            patterns.append("state_correction")

    return patterns


def _text_overlap(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard overlap between two texts."""
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
