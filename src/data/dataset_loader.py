"""Test set loading, normalization, hashing, and entity-combo extraction."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

log = logging.getLogger(__name__)


def _load_test_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load a single JSON file and unpack nested test_cases format if present.

    Handles two structures:
    1. Flat array of test objects: ``[{test_case_id, turns, ...}, ...]``
    2. Nested container: ``{intent, channel, test_cases: {single_turn: [...], multi_turn: [...]}}``
    """
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return list(data)

    if not isinstance(data, dict):
        return [data]

    # Nested container format — unpack test_cases
    test_cases_block = data.get("test_cases")
    if isinstance(test_cases_block, dict):
        file_intent = data.get("intent", filepath.stem).lower()
        file_channel = data.get("channel", "chat")
        unpacked: List[Dict[str, Any]] = []

        for section in ("single_turn", "multi_turn"):
            is_multi = section == "multi_turn"
            for tc in test_cases_block.get(section, []):
                turns = tc.get("turns", [])
                ref_outputs = tc.get("reference_output", [])
                # Build a conversation array from turns
                conversation = []
                for i, turn_text in enumerate(turns):
                    conversation.append({"role": "user", "message": turn_text})
                    ref = ref_outputs[i] if i < len(ref_outputs) else ""
                    if ref:
                        conversation.append({"role": "assistant", "message": ref})

                flat = {
                    "id": tc.get("test_case_id", ""),
                    "intent": file_intent,
                    "channel": file_channel,
                    "utterance": turns[0] if turns else "",
                    "conversation": conversation,
                    "is_multi_turn": is_multi,
                    "turn_count": len(turns),
                }
                unpacked.append(flat)
        return unpacked

    # Single dict that doesn't follow the container format
    return [data]


def load_testset(path: Path) -> List[Dict[str, Any]]:
    """Load existing test cases from a single JSON file or a channel/intent directory tree.

    Supports:
    - Single JSON file containing an array of test case objects.
    - Directory tree: ``testsets/chat/payments.json``, ``testsets/email/complaints.json``, etc.

    Each test case is normalized to include at minimum:
    ``utterance``, ``intent``, ``channel``, ``entities``.
    """
    tests: List[Dict[str, Any]] = []

    if path.is_file():
        tests = _load_test_file(path)
    elif path.is_dir():
        for json_file in sorted(path.rglob("*.json")):
            # Infer channel and intent from directory structure
            rel = json_file.relative_to(path)
            parts = rel.parts
            channel = parts[0].lower() if len(parts) > 1 else "chat"
            intent_from_file = json_file.stem.lower()

            file_tests = _load_test_file(json_file)
            for t in file_tests:
                if not t.get("channel"):
                    t["channel"] = channel
                if not t.get("intent"):
                    t["intent"] = intent_from_file
            tests.extend(file_tests)
    else:
        log.warning("Test set path does not exist: %s", path)
        return []

    # Normalize each test case
    for t in tests:
        _normalize_test(t)

    log.info("Loaded %d test cases from %s", len(tests), path)
    return tests


def _normalize_test(t: Dict[str, Any]) -> None:
    """Normalize a test case to ensure standard fields are present."""
    # Derive utterance from conversation or email_thread if missing
    if not t.get("utterance"):
        conversation = t.get("conversation", [])
        email_thread = t.get("email_thread", [])
        messages = conversation or email_thread
        user_messages = [m.get("message", "") for m in messages if m.get("role") == "user"]
        t["utterance"] = " ".join(user_messages).strip() if user_messages else ""

    # Ensure entities dict exists
    if "entities" not in t:
        t["entities"] = {}

    # Ensure intent is lowercase
    if t.get("intent"):
        t["intent"] = t["intent"].strip().lower()

    # Ensure channel defaults
    if not t.get("channel"):
        t["channel"] = "chat"
    t["channel"] = t["channel"].strip().lower()

    # Ensure language defaults
    if not t.get("language"):
        t["language"] = "en"

    # Pull metadata fields to top level if present
    metadata = t.get("metadata", {})
    if metadata:
        if not t.get("difficulty"):
            t["difficulty"] = metadata.get("difficulty", "medium")


def build_test_index(testset: List[Dict[str, Any]], index: Any) -> None:
    """Embed all test case utterances and add them to the FAISS index.

    Args:
        testset: List of normalized test case dicts (must have ``utterance``).
        index: An ``EmbeddingIndex`` instance with an ``add_texts`` method.
    """
    texts = []
    ids = []
    metadata_list = []

    for t in testset:
        utterance = t.get("utterance", "").strip()
        if not utterance:
            continue
        texts.append(utterance)
        ids.append(t.get("id", ""))
        metadata_list.append({
            "intent": t.get("intent", ""),
            "channel": t.get("channel", "chat"),
            "language": t.get("language", "en"),
        })

    if texts:
        index.add_texts(texts, ids=ids, metadata=metadata_list)
        log.info("Built index with %d test utterances", len(texts))


def extract_entity_combos(testset: List[Dict[str, Any]]) -> Set[Tuple[str, ...]]:
    """Extract all unique entity-key combinations from the test set.

    Returns a set of tuples, where each tuple is a sorted set of entity keys
    found in a single test case.
    """
    combos: Set[Tuple[str, ...]] = set()
    for t in testset:
        entities = t.get("entities", {})
        if entities:
            combo = tuple(sorted(entities.keys()))
            combos.add(combo)
    return combos


def testset_hash(path: Path) -> str:
    """Compute a SHA-256 hash of the test set files for cache invalidation.

    If ``path`` is a directory, hashes all JSON files recursively.
    If ``path`` is a single file, hashes that file.
    """
    h = hashlib.sha256()

    if path.is_file():
        h.update(path.read_bytes())
    elif path.is_dir():
        for json_file in sorted(path.rglob("*.json")):
            h.update(json_file.read_bytes())
    else:
        h.update(b"empty")

    return h.hexdigest()[:16]
