import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .data.dataset_loader import load_testset, build_test_index, extract_entity_combos, testset_hash
from .data.embedding_index import EmbeddingIndex
from .pipeline.conversation_parser import parse_conversations
from .pipeline.nlu import extract_nlu, clear_nlu_memo
from .pipeline.gap_analysis import analyze_gap, build_search_text
from .pipeline.test_generation import generate_test_case, postprocess_suggestions, select_test_type
from .pipeline.slot_planner import build_slot_plan, clear_plan_cache
from .models import PipelineConfig, TokenTracker

log = logging.getLogger(__name__)


def _next_run_dir(out_root: Path) -> Path:
    """Create and return the next date-based run folder under out/.

    Naming: out/<YYYY-MM-DD>_001/, out/<YYYY-MM-DD>_002/, etc.
    Scans existing folders for today's date and increments the counter.
    """
    today = date.today().isoformat()
    pattern = re.compile(rf"^{re.escape(today)}_(\d{{3}})$")
    max_serial = 0
    if out_root.exists():
        for d in out_root.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    max_serial = max(max_serial, int(m.group(1)))
    run_dir = out_root / f"{today}_{max_serial + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Intent-grouped output: write test cases into run_dir/existing_intents|new_intents/
# ---------------------------------------------------------------------------
def _write_intent_grouped_output(
    all_kept: Dict[str, List[Dict[str, Any]]],
    existing_intents: Set[str],
    run_dir: Path,
) -> Path:
    """Write test cases grouped by channel then intent into the run folder.

    Structure:
        out/<YYYY-MM-DD>_001/
            existing_intents/
                chat/complaints.json
                email/payments.json
            new_intents/
                chat/NEW_rent_increase.json
                email/NEW_security_deposit.json
            report.md

    Returns the run output directory path.
    """
    dir_existing = run_dir / "existing_intents"
    dir_new = run_dir / "new_intents"

    # Group by channel → intent
    # channel_intent: { channel: { intent: [cases] } }
    channel_intent: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for channel, kept in all_kept.items():
        for s in kept:
            intent = s.get("intent", "unknown").strip() or "unknown"
            channel_intent.setdefault(channel, {}).setdefault(intent, []).append(s)

    existing_count = 0
    new_count = 0
    for channel, intents in sorted(channel_intent.items()):
        for intent, cases in sorted(intents.items()):
            intent_lower = intent.lower()
            is_new_intent = (
                intent.startswith("NEW_")
                or intent_lower not in existing_intents
            )

            if is_new_intent:
                intent_name = intent[4:] if intent.startswith("NEW_") else intent
                if not intent_name:
                    intent_name = "unknown"
                filename = f"NEW_{intent_name}.json"
                target_dir = dir_new / channel
            else:
                filename = f"{intent_lower}.json"
                target_dir = dir_existing / channel

            target_dir.mkdir(parents=True, exist_ok=True)
            filepath = target_dir / filename
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(cases, f, indent=2, ensure_ascii=False)

            if is_new_intent:
                new_count += 1
            else:
                existing_count += 1

    print(f"\n[output] Intent-grouped output:")
    print(f"[output]   existing_intents/ : {existing_count} intent files")
    print(f"[output]   new_intents/      : {new_count} intent files")
    print(f"[output] Run directory: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Combinatorial novelty: extract existing (intent x dimension) combos (#3)
# ---------------------------------------------------------------------------
def _extract_known_combos(
    testset: List[Dict[str, Any]],
) -> Tuple[
    Set[Tuple[str, str]],  # intent x channel
    Set[Tuple[str, str]],  # intent x policy
    Set[Tuple[str, str]],  # intent x language
    Set[Tuple[str, str]],  # intent x escalation
]:
    """Scan the existing test set to build sets of observed (intent, X) pairs."""
    intent_channel: Set[Tuple[str, str]] = set()
    intent_policy: Set[Tuple[str, str]] = set()
    intent_language: Set[Tuple[str, str]] = set()
    intent_escalation: Set[Tuple[str, str]] = set()

    for t in testset:
        intent = t.get("intent", "").lower()
        if not intent:
            continue
        channel = t.get("channel", "chat")
        intent_channel.add((intent, channel))

        lang = t.get("language", "en")
        intent_language.add((intent, lang))

        tags = t.get("tags", [])
        metadata = t.get("metadata", {})
        # Policy
        if (
            "policy" in tags
            or "policy" in intent
            or metadata.get("category", "").lower() in ("policy", "compliance")
        ):
            intent_policy.add((intent, "policy"))

        # Escalation / failure
        if (
            "failure" in tags
            or "escalation" in tags
            or "error" in intent
            or "failure" in intent
        ):
            intent_escalation.add((intent, "escalation"))

    return intent_channel, intent_policy, intent_language, intent_escalation
# Trivial-utterance filter — skip API calls entirely for short acks
# ---------------------------------------------------------------------------
_TRIVIAL_RE = re.compile(
    r"^(?:yes|no|ok(?:ay)?|sure|thanks?|thank\s*you|hi|hello|hey|bye"
    r"|got\s*it|alright|yep|nope|hmm+|hm+|k|yea[h]?|na[h]?"
    r"|please|great|perfect|wow|oh|ah|nice|cool|fine|right"
    r"|done|understood|correct|exactly|absolutely|definitely|agreed"
    r"|good|welcome|morning|evening|noted)[.!?]*$",
    re.IGNORECASE,
)

# HTML junk detection — emails with raw HTML/CSS that aren't real utterances
_HTML_JUNK_RE = re.compile(
    r"<(?:!--|style|meta|link|font-face|div\.|p\.|span\.)[^>]*>"
    r"|@font-face\s*\{"
    r"|font-family:\s*[\"']"
    r"|mso-[a-z-]+:"
    r"|/\*.*?\*/",
    re.IGNORECASE | re.DOTALL,
)


def _is_trivial(text: str) -> bool:
    """Return True if the utterance is a trivial acknowledgment or HTML junk."""
    stripped = text.strip()
    if len(stripped) < 2:
        return True
    if _HTML_JUNK_RE.search(stripped):
        return True
    # After stripping HTML tags, check if there's any real content left
    text_only = re.sub(r"<[^>]+>", " ", stripped).strip()
    text_only = re.sub(r"\s+", " ", text_only).strip()
    if len(text_only) < 3:
        return True
    return bool(_TRIVIAL_RE.match(stripped))


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Silence noisy HTTP-level loggers to keep output readable
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # Validate with Pydantic — raises on bad / missing keys
    PipelineConfig(**raw)
    log.info("Config validated successfully from %s", path)
    return raw


# ---------------------------------------------------------------------------
#  Interactive CLI
# ---------------------------------------------------------------------------

def _prompt_choice(header: str, options: List[str]) -> int:
    """Display a numbered menu and return the 1-based choice (validated)."""
    print(f"\n{'─' * 50}")
    print(f"  {header}")
    print(f"{'─' * 50}")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        try:
            choice = int(input(f"\n  Enter choice (1-{len(options)}): ").strip())
            if 1 <= choice <= len(options):
                return choice
        except (ValueError, EOFError):
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def _prompt_yes_no(question: str) -> bool:
    """Ask a yes / no question and return True for yes."""
    while True:
        ans = input(f"\n  {question} (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please enter y or n.")


class _InteractiveArgs:
    """Container for interactive CLI answers (drop-in for argparse.Namespace)."""
    def __init__(self) -> None:
        self.mode: str = "conservative"
        self.filter_style: str = "combo"     # "strict" or "combo"
        self.clear_cache: bool = False
        self.testset_modified: bool = False
        self.novelty_min: float | None = None


def _interactive_cli() -> _InteractiveArgs:
    """Run the interactive CLI flow and return user choices."""
    args = _InteractiveArgs()

    print("\n" + "=" * 50)
    print("   T E S T   G A P   M I N E R")
    print("=" * 50)

    # 1. Mode selection
    mode_choice = _prompt_choice(
        "Select novelty mode",
        ["Conservative  \u2014 strict filtering, only highly novel scenarios (novelty >= 0.48)",
         "Exploration   \u2014 broader coverage, catch more gaps (novelty >= 0.45, impact overrides)"],
    )
    args.mode = "conservative" if mode_choice == 1 else "exploration"

    # 2. Filter style
    filter_choice = _prompt_choice(
        "Select filtering approach",
        ["Only Strict New       — pure novelty threshold, no combo bypass",
         "Combination of New    — also keep new intent+channel combos & NEW_* intents"],
    )
    args.filter_style = "strict" if filter_choice == 1 else "combo"

    # 3. Clear cache?
    args.clear_cache = _prompt_yes_no("Do you want to clear the suggestion cache?")

    # 4. Testset modified?
    args.testset_modified = _prompt_yes_no("Is the test set modified since last run?")

    # Summary
    print(f"\n{'─' * 50}")
    print(f"  Mode            : {args.mode}")
    print(f"  Filter style    : {args.filter_style}")
    print(f"  Clear cache     : {'yes' if args.clear_cache else 'no'}")
    print(f"  Testset modified: {'yes' if args.testset_modified else 'no'}")
    print(f"{'─' * 50}\n")

    return args


def run() -> None:
    args = _interactive_cli()
    root = Path(__file__).resolve().parents[1]

    # Load .env file if present
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Logging (honour LOG_LEVEL from env)
    _setup_logging(os.getenv("LOG_LEVEL", "INFO"))

    config_path = root / "config" / "settings.yaml"
    config = load_config(config_path)

    api_key_env = config["openai"]["api_key_env"]
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"OpenAI API key not found in env var {api_key_env}")
    client = OpenAI(api_key=api_key)

    embedding_model = config["openai"]["embedding_model"]
    chat_model = config["openai"]["chat_model"]

    index = EmbeddingIndex(client=client, model=embedding_model)

    testset_path = root / config["paths"]["testset"]
    taxonomy_path = root / config["paths"]["taxonomy"]
    slot_policy_path = root / config["paths"].get("slot_policies", "data/slot_policies.json")
    test_type_dist = config.get("test_types", {"policy_flow": 0.70, "edge_case": 0.20, "stress": 0.10})

    testset = load_testset(testset_path)

    # ------------------------------------------------------------------
    # FAISS index cache — rebuild only when testset.json changes
    # ------------------------------------------------------------------
    cache_dir = root / ".cache"
    ts_hash = testset_hash(testset_path)
    # If user said testset is modified, force index rebuild
    if args.testset_modified:
        print(f"[cache] Testset marked as modified — rebuilding index (hash={ts_hash})...")
        build_test_index(testset, index)
        index.save(cache_dir, ts_hash)
        print(f"[cache] Index saved to {cache_dir}.")
    else:
        cached = EmbeddingIndex.load_cached(cache_dir, ts_hash, client, embedding_model)
        if cached is not None:
            index = cached
            print(f"[cache] Loaded index from disk (testset unchanged, hash={ts_hash}).")
        else:
            print(f"[cache] Building new index (hash={ts_hash}) -- this may take a minute...")
            build_test_index(testset, index)
            index.save(cache_dir, ts_hash)
            print(f"[cache] Index saved to {cache_dir}.")

    # CLI --clear-cache: remove suggestion caches before running
    if args.clear_cache:
        import glob as _glob
        for pattern in ["*.suggestions.json", "*.turn_cache.json"]:
            for f in _glob.glob(str(cache_dir / pattern)):
                os.remove(f)
        print("[cache] Suggestion and turn caches cleared.")

    known_entity_combos = extract_entity_combos(testset)

    # Extract combinatorial novelty baselines (#3)
    known_intent_channel, known_intent_policy, known_intent_language, known_intent_escalation = (
        _extract_known_combos(testset)
    )

    with taxonomy_path.open("r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    # Resolve novelty mode from interactive CLI
    novelty_mode = args.mode
    # CLI --novelty-min overrides the relevant threshold in config
    if args.novelty_min is not None:
        if novelty_mode == "exploration":
            config.setdefault("novelty", {})["keep_min_novelty"] = args.novelty_min
        else:
            config.setdefault("novelty", {})["exploration_min_novelty"] = args.novelty_min

    # Stash filter_style so the keep/filter loop can read it
    combo_bypass_enabled = args.filter_style == "combo"
    print(f"[config] Novelty mode: {novelty_mode}")
    print(f"[config] Filter style: {args.filter_style} ({'combo bypass ON' if combo_bypass_enabled else 'strict threshold only'})")
    effective_threshold = (
        config.get("novelty", {}).get("keep_min_novelty", 0.45)
        if novelty_mode == "exploration"
        else config.get("novelty", {}).get("exploration_min_novelty", 0.50)
    )
    print(f"[config] Effective novelty threshold: {effective_threshold}")

    # Reset token tracker and NLU memo for this run
    TokenTracker.instance().reset()
    clear_nlu_memo()

    # ------------------------------------------------------------------
    # Channel definitions — each channel has its own input & output
    # ------------------------------------------------------------------
    channels = []

    # Auto-discover conversation files in the conversations directory
    conv_dir = root / config["paths"].get("conversations", "data/conversations/")
    if conv_dir.is_dir():
        # Text conversation logs (primary source)
        txt_files = sorted(conv_dir.glob("*.txt"))
        for tf in txt_files:
            tag = "email" if "email" in tf.stem.lower() else "chat"
            channels.append({
                "name": tf.stem.lower(),
                "conversations_path": tf,
                "channel_tag": tag,
            })

        # HTML evaluation reports (secondary source)
        html_files = sorted(conv_dir.glob("*.html")) + sorted(conv_dir.glob("*.htm"))
        for hf in html_files:
            ch_name = hf.stem.replace(" ", "_").lower()[:40]
            channels.append({
                "name": f"html_{ch_name}",
                "conversations_path": hf,
                "channel_tag": "chat",
            })
            print(f"[config] Found HTML report: {hf.name}")

    if not channels:
        print("[config] WARNING: No conversation files found in", conv_dir)
        print("[config] Drop .html evaluation reports into data/conversations/")

    all_kept: Dict[str, List[Dict[str, Any]]] = {}

    # Determine run output folder: out/<date>_<serial>/
    out_root = root / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = _next_run_dir(out_root)
    print(f"[config] Run output folder: {out_dir.name}")

    for ch in channels:
        ch_name = ch["name"]
        conversations_path = ch["conversations_path"]

        # Skip channels whose input file is empty or missing
        if not conversations_path.exists() or conversations_path.stat().st_size == 0:
            print(f"[{ch_name}] No conversations found at {conversations_path} — skipping.")
            all_kept[ch_name] = []
            continue

        print(f"\n{'='*60}")
        print(f"  Processing channel: {ch_name.upper()}")
        print(f"  Input: {conversations_path}")
        print(f"{'='*60}")

        turns = parse_conversations(conversations_path)

        # ------------------------------------------------------------------
        # Conversation result cache — skip NLU / gap / testgen API calls
        # when the same conversations + testset combo was processed before
        # ------------------------------------------------------------------
        conv_hash = hashlib.sha256(conversations_path.read_bytes()).hexdigest()[:16]
        combo_key = f"{ts_hash}_{conv_hash}_{ch_name}"
        conv_cache_file = cache_dir / f"{combo_key}.suggestions.json"

        if conv_cache_file.exists():
            with conv_cache_file.open("r", encoding="utf-8") as fh:
                suggestions = json.load(fh)
            print(f"[{ch_name}][cache] Loaded {len(suggestions)} pre-filter suggestions from cache.")
            print(f"        To force re-analysis, delete {conv_cache_file}")
            turn_cache = {}  # not needed when loading from conv cache
        else:
            # Clean old caches for this channel
            for old_sug in cache_dir.glob(f"*_{ch_name}.suggestions.json"):
                if old_sug.name != f"{combo_key}.suggestions.json":
                    old_sug.unlink(missing_ok=True)

            # ----------------------------------------------------------
            # Per-turn result cache — avoid re-processing unchanged turns
            # ----------------------------------------------------------
            turn_cache_file = cache_dir / f"{combo_key}.turn_cache.json"
            if turn_cache_file.exists():
                with turn_cache_file.open("r", encoding="utf-8") as fh:
                    turn_cache: Dict[str, Dict[str, Any]] = json.load(fh)
            else:
                turn_cache = {}

            def _turn_cache_key(user_text: str, agent_text: str) -> str:
                """Deterministic hash of a turn's content for caching."""
                blob = f"{user_text}|||{agent_text}".encode("utf-8")
                return hashlib.sha256(blob).hexdigest()[:24]

            tracker = TokenTracker.instance()

            # ==========================================================
            # PHASE 1: Pre-filter trivial turns + check per-turn cache
            # ==========================================================
            non_trivial: List[Dict[str, Any]] = []
            cached_suggestions: List[Dict[str, Any]] = []
            trivial_count = 0
            cached_hit_count = 0

            for t in turns:
                user_text = t.get("user_text", "").strip()
                agent_text = t.get("agent_text", "").strip()
                if not user_text:
                    continue

                # Skip trivial acks — no API calls needed
                if _is_trivial(user_text):
                    trivial_count += 1
                    tracker.record_trivial_skip()
                    continue

                # Check per-turn cache
                tck = _turn_cache_key(user_text, agent_text)
                cached_result = turn_cache.get(tck)
                if cached_result is not None:
                    sug = dict(cached_result)
                    sug["id"] = f"SUG-{t.get('session_id', 'unknown')}-{t.get('turn_index', 0):04d}"
                    sug["source_session"] = t.get("session_id", "unknown")
                    sug["channel"] = ch.get("channel_tag", "chat")
                    cached_suggestions.append(sug)
                    cached_hit_count += 1
                    continue

                non_trivial.append(t)

            print(f"[{ch_name}] {len(turns)} turns -> {trivial_count} trivial skipped, "
                  f"{cached_hit_count} cached, {len(non_trivial)} to process")

            # ==========================================================
            # PHASE 2: NLU extraction — concurrent with memoization
            # ==========================================================
            nlu_results: Dict[int, Dict[str, Any]] = {}

            if non_trivial:
                max_workers = min(4, len(non_trivial)) or 1
                print(f"[{ch_name}] Phase 2: NLU extraction ({len(non_trivial)} turns, workers={max_workers})...")

                def _run_nlu(idx_turn: tuple) -> tuple:
                    idx, t = idx_turn
                    return idx, extract_nlu(
                        client, chat_model,
                        t["user_text"].strip(), taxonomy,
                        agent_text=t.get("agent_text", "").strip(),
                    )

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_run_nlu, (i, t)): i
                        for i, t in enumerate(non_trivial)
                    }
                    with tqdm(total=len(non_trivial), desc=f"NLU {ch_name}") as pbar:
                        for future in as_completed(futures, timeout=300):
                            pbar.update(1)
                            try:
                                idx, nlu = future.result(timeout=60)
                            except Exception as exc:
                                log.warning("NLU future failed: %s", exc)
                                continue
                            nlu_results[idx] = nlu

            # ==========================================================
            # PHASE 2b: Session-level intent resolution
            # Resolve NEW_* intents using the dominant taxonomy intent
            # from the same session. Intents should reflect the overall
            # conversation topic, not a single turn's detail.
            # ==========================================================
            if nlu_results:
                taxonomy_intents = set(
                    i.lower() for i in taxonomy.get("intents", [])
                ) | set(
                    i.lower() for i in taxonomy.get("domains", [])
                )
                # Also include sub_intents if present in taxonomy
                for sub_list in taxonomy.get("sub_intents", {}).values():
                    if isinstance(sub_list, list):
                        taxonomy_intents.update(i.lower() for i in sub_list)

                # Group NLU results by session_id
                session_intents: Dict[str, List[str]] = {}
                for idx, nlu in nlu_results.items():
                    sid = non_trivial[idx].get("session_id", "unknown")
                    intent = nlu.get("intent", "")
                    session_intents.setdefault(sid, []).append(intent)

                # For each session, find the dominant taxonomy intent
                session_dominant: Dict[str, str] = {}
                for sid, intents in session_intents.items():
                    known = [i for i in intents if i.lower() in taxonomy_intents]
                    if known:
                        # Most common taxonomy intent in this session
                        intent_counts = Counter(known)
                        session_dominant[sid] = intent_counts.most_common(1)[0][0]

                # Override NEW_* intents with the session's dominant intent
                overridden = 0
                for idx, nlu in nlu_results.items():
                    intent = nlu.get("intent", "")
                    if intent.startswith("NEW_"):
                        sid = non_trivial[idx].get("session_id", "unknown")
                        dominant = session_dominant.get(sid)
                        if dominant:
                            log.info(
                                "Intent override: %s -> %s (session %s dominant)",
                                intent, dominant, sid,
                            )
                            nlu["intent"] = dominant
                            overridden += 1
                if overridden:
                    print(f"[{ch_name}] Phase 2b: Resolved {overridden} NEW_* intents to session-level taxonomy intents")

            # ==========================================================
            # PHASE 3: Batch embedding — ONE API call for all search texts
            # ==========================================================
            top_k = config["similarity"]["top_k"]
            batch_neighbors: List[List[tuple]] = []

            if non_trivial:
                search_texts = [
                    build_search_text(
                        nlu_results[i],
                        non_trivial[i]["user_text"].strip(),
                        non_trivial[i].get("agent_text", "").strip(),
                    )
                    for i in range(len(non_trivial))
                ]
                print(f"[{ch_name}] Phase 3: Batch embedding ({len(search_texts)} texts in 1 API call)...")
                batch_neighbors = index.search_batch(search_texts, top_k)

            # ==========================================================
            # PHASE 4: Gap analysis — local only, no API calls
            # ==========================================================
            gap_candidates: List[tuple] = []  # (turn_idx, turn, nlu, gap_info)

            # Inject current channel into config for combo tracking
            config["_current_channel"] = ch.get("channel_tag", "chat")

            for i, t in enumerate(non_trivial):
                nlu = nlu_results[i]
                neighbors = batch_neighbors[i] if i < len(batch_neighbors) else []
                prior_turns = t.get("prior_turns", [])
                gap_info = analyze_gap(
                    index, nlu, t["user_text"].strip(), config,
                    known_entity_combos,
                    agent_text=t.get("agent_text", "").strip(),
                    precomputed_neighbors=neighbors,
                    prior_turns=prior_turns if prior_turns else None,
                    known_intent_channel=known_intent_channel,
                    known_intent_policy=known_intent_policy,
                    known_intent_language=known_intent_language,
                    known_intent_escalation=known_intent_escalation,
                )
                # Skip paraphrases (#5) — reject before testgen to save API calls
                if gap_info.get("is_paraphrase", False):
                    log.info("Rejected paraphrase: %s (novelty=%.3f)",
                             nlu.get("intent"), gap_info["novelty_score"])
                    continue
                if gap_info["is_candidate"]:
                    gap_candidates.append((i, t, nlu, gap_info))

            print(f"[{ch_name}] Phase 4: {len(gap_candidates)} gap candidates out of {len(non_trivial)} turns")

            # ==========================================================
            # PHASE 5: Test generation — concurrent, only for candidates
            # ==========================================================
            new_suggestions: List[Dict[str, Any]] = []

            if gap_candidates:
                max_workers = 1  # sequential — eliminates OpenAI rate-limit cascades
                print(f"[{ch_name}] Phase 5: Test generation ({len(gap_candidates)} candidates, sequential)...")

                def _run_testgen(item: tuple) -> Dict[str, Any] | None:
                    i, t, nlu_result, gap_info = item
                    session_id = t.get("session_id", "unknown")
                    user_text = t["user_text"].strip()
                    agent_text = t.get("agent_text", "").strip()
                    prior_turns = t.get("prior_turns", [])

                    # Select test type for this candidate
                    test_type, difficulty = select_test_type(test_type_dist)

                    # Build slot plan for this intent
                    intent = nlu_result.get("intent", "unknown")
                    slot_plan = build_slot_plan(
                        intent=intent,
                        policy_path=slot_policy_path,
                        client=client,
                        model=chat_model,
                        user_utterance=user_text,
                        entities=nlu_result.get("entities", {}),
                    )

                    gen = generate_test_case(
                        client=client,
                        model=chat_model,
                        session_id=session_id,
                        user_utterance=user_text,
                        agent_utterance=agent_text,
                        nlu_result=nlu_result,
                        gap_info=gap_info,
                        prior_turns=prior_turns if prior_turns else None,
                        slot_plan=slot_plan,
                        test_type=test_type,
                    )

                    # Build conversation array
                    conversation = []
                    for pt in prior_turns:
                        if pt.get("user_text"):
                            conversation.append({"role": "user", "message": pt["user_text"]})
                        if pt.get("agent_text"):
                            conversation.append({"role": "assistant", "message": pt["agent_text"]})
                    conversation.append({"role": "user", "message": user_text})
                    if agent_text:
                        conversation.append({"role": "assistant", "message": agent_text})

                    gen_conversation = gen.get("conversation", [])
                    final_conversation = gen_conversation if gen_conversation else conversation

                    # Classify based on the GENERATED conversation, not source
                    gen_turn_count = len(final_conversation)
                    is_multi_turn = gen_turn_count > 2
                    turn_count = gen_turn_count

                    suggestion = {
                        "id": f"SUG-{session_id}-{t.get('turn_index', 0):04d}",
                        "reason": gen.get("reason", "Covers a gap identified in conversation analysis."),
                        "source_session": session_id,
                        "channel": ch.get("channel_tag", "chat"),
                        "user_utterance": user_text,
                        "intent": nlu_result.get("intent"),
                        "entities": nlu_result.get("entities", {}),
                        "language": nlu_result.get("language"),
                        "test_type": test_type,
                        "difficulty": difficulty,
                        "newness_pct": gap_info["newness_pct"],
                        "newness_label": gap_info["newness_label"],
                        "novelty_score": round(gap_info["novelty_score"], 4),
                        "impact_score": round(gap_info["impact_score"], 4),
                        "nearest_existing_tests": gap_info["neighbors"],
                        "gap_reason": gen.get("gap_reason", gap_info["reasons"]),
                        "expected_behavior": gen.get("expected_behavior"),
                        "expected_answer": gen.get("expected_answer"),
                        "acceptance_criteria": gen.get("acceptance_criteria", []),
                        "tags": gen.get("tags", []),
                        "confidence": round(1.0 - gap_info["max_similarity"], 4),
                        "conversation": final_conversation,
                        "is_multi_turn": is_multi_turn,
                        "turn_count": turn_count,
                        # New enrichment fields
                        "has_structural_novelty": gap_info.get("has_structural_novelty", False),
                        "constraint_count": gap_info.get("constraint_count", 0),
                        "multi_turn_patterns": gap_info.get("multi_turn_patterns", []),
                        "unseen_combos": gap_info.get("unseen_combos", []),
                        "avg_similarity": round(gap_info.get("avg_similarity", 0), 4),
                    }

                    # Derive a meaningful name for NEW_ intents with empty names
                    raw_intent = suggestion.get("intent", "") or ""
                    if raw_intent == "NEW_" or raw_intent.startswith("NEW_") and len(raw_intent) <= 4:
                        # Try to derive from reason, then gap_reason, then tags
                        source_text = (
                            suggestion.get("reason", "")
                            or " ".join(suggestion.get("gap_reason", []))
                            or " ".join(suggestion.get("tags", []))
                        )
                        words = re.sub(r"[^a-zA-Z0-9\s]", "", source_text).split()
                        # Pick up to 4 meaningful words (skip short filler words)
                        name_words = [w for w in words if len(w) > 2][:4]
                        intent_name = "_".join(w.lower() for w in name_words) if name_words else "unknown"
                        suggestion["intent"] = f"NEW_{intent_name}"

                    # Store in per-turn cache
                    tck = _turn_cache_key(user_text, agent_text)
                    turn_cache[tck] = suggestion
                    return suggestion

                # Sequential loop — no concurrency, no rate-limit cascades
                with tqdm(total=len(gap_candidates), desc=f"TestGen {ch_name}") as pbar:
                    for idx, item in enumerate(gap_candidates):
                        try:
                            result = _run_testgen(item)
                        except Exception as exc:
                            log.warning("TestGen call failed: %s", exc)
                            result = None
                        if result is not None:
                            new_suggestions.append(result)
                        pbar.update(1)
                        # 1.5 s pause between calls to stay under RPM limits
                        if idx < len(gap_candidates) - 1:
                            time.sleep(1.5)

            # Merge cached + new suggestions
            suggestions = cached_suggestions + new_suggestions

            # Sort by original order
            suggestions.sort(key=lambda s: (s.get("source_session", ""), s.get("id", "")))

            # Collect neighbors for postprocessing
            neighbors_for_suggestions = [s.get("nearest_existing_tests", []) for s in suggestions]

            suggestions = postprocess_suggestions(suggestions, neighbors_for_suggestions, config)

            # Save pre-filter suggestions to conv cache (so mode/filter changes
            # don't require re-running NLU/gap/testgen API calls)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with conv_cache_file.open("w", encoding="utf-8") as fh:
                json.dump(suggestions, fh, indent=2, ensure_ascii=False)
            with turn_cache_file.open("w", encoding="utf-8") as fh:
                json.dump(turn_cache, fh, indent=2, ensure_ascii=False)
            print(f"[{ch_name}][cache] Saved {len(suggestions)} pre-filter suggestions + "
                  f"{len(turn_cache)} turn results to cache.")

        # ==============================================================
        # KEEP / FILTER — always runs (even from cache) so mode & filter
        # style choices take effect without re-running API calls.
        # ==============================================================
        novelty_cfg = config.get("novelty", {})
        is_exploration = novelty_mode == "exploration"

        if is_exploration:
            # Exploration: broader coverage, lower threshold + impact override
            min_novelty = novelty_cfg.get("keep_min_novelty", 0.45)
        else:
            # Conservative: strict, higher threshold
            min_novelty = novelty_cfg.get("exploration_min_novelty", 0.50)

        min_impact = novelty_cfg.get("keep_min_impact", 0.20)
        impact_override_min_novelty = novelty_cfg.get("impact_override_min_novelty", 0.30)

        # --- Combo bypass: always keep unseen intent+channel combos ---
        # Minimum novelty floor for combo bypass (avoid true duplicates)
        # Conservative mode uses a higher floor so results differ from exploration
        if is_exploration:
            combo_bypass_floor = novelty_cfg.get("combo_bypass_min_novelty", 0.25)
        else:
            combo_bypass_floor = novelty_cfg.get(
                "conservative_combo_bypass_min_novelty",
                novelty_cfg.get("combo_bypass_min_novelty", 0.25),
            )

        kept: List[Dict[str, Any]] = []
        rejected = 0
        rejected_structural = 0
        rejected_conservative = 0
        combo_bypassed = 0
        for s in suggestions:
            nov = s.get("novelty_score", 0)
            imp = s.get("impact_score", 0)
            structural = s.get("has_structural_novelty", True)  # default True for cached
            unseen = s.get("unseen_combos", [])

            # Combo bypass: if the intent+channel pair is unseen, keep it
            # as long as it clears the mode-appropriate floor.
            # Only active when user chose "Combination of New"
            if combo_bypass_enabled:
                has_new_intent_channel = any(
                    c.startswith("intent+channel(") for c in unseen
                )
                has_new_intent = s.get("intent", "").startswith("NEW_")

                if (has_new_intent_channel or has_new_intent) and nov >= combo_bypass_floor:
                    kept.append(s)
                    combo_bypassed += 1
                    continue

            if is_exploration:
                # Exploration mode: broader coverage — lower threshold
                # Keep if novelty >= 0.45
                # OR (impact >= 0.20 AND novelty >= 0.30)
                if nov >= min_novelty:
                    kept.append(s)
                elif imp >= min_impact and nov >= impact_override_min_novelty:
                    kept.append(s)
                else:
                    rejected += 1

                # Structural novelty gate — reject paraphrased single-intent
                # cases that only survived on novelty but have no structural novelty
                if kept and kept[-1] is s and not structural and nov < 0.50:
                    kept.pop()
                    rejected += 1
                    rejected_structural += 1
            else:
                # Conservative mode: strict novelty + structural required
                # Impact override: high-impact suggestions (policy, failure, etc.)
                # with novelty near the threshold still deserve consideration
                conservative_impact_floor = novelty_cfg.get("conservative_impact_override_min_novelty", 0.42)
                if nov >= min_novelty and structural:
                    kept.append(s)
                elif structural and imp >= 0.30 and nov >= conservative_impact_floor:
                    kept.append(s)
                else:
                    rejected += 1
                    if not structural:
                        rejected_conservative += 1
        if combo_bypassed:
            print(f"[{ch_name}] {combo_bypassed} suggestions kept via combo bypass "
                  f"(new intent or unseen intent+channel, floor={combo_bypass_floor})")

        if rejected_structural:
            print(f"[{ch_name}] Rejected {rejected_structural} suggestions lacking structural novelty")
        if rejected_conservative:
            print(f"[{ch_name}] Rejected {rejected_conservative} suggestions in conservative mode (no structural novelty)")

        print(f"[{ch_name}] Filter result: {len(kept)} kept, {len(suggestions)} pre-filter, {rejected} rejected")

        # --- Coverage saturation detection (#9) ---
        if kept:
            novelty_scores = [s.get("novelty_score", 0) for s in kept]
            median_novelty = sorted(novelty_scores)[len(novelty_scores) // 2]
            high_novelty_count = sum(1 for n in novelty_scores if n > 0.50)
            high_novelty_pct = (high_novelty_count / len(novelty_scores)) * 100
            edge_scraping_count = sum(1 for n in novelty_scores if 0.35 <= n <= 0.45)
            edge_scraping_pct = (edge_scraping_count / len(novelty_scores)) * 100 if novelty_scores else 0

            print(f"[{ch_name}] Saturation metrics: median_novelty={median_novelty:.3f}, "
                  f">{0.50:.0%}_novelty={high_novelty_pct:.1f}%, "
                  f"edge_scraping(0.35-0.45)={edge_scraping_pct:.1f}%")
            if edge_scraping_pct > 50:
                print(f"[{ch_name}] WARNING: >50% of suggestions are edge-scraping (novelty 0.35-0.45). "
                      f"Consider switching to exploration mode or adding adversarial/synthetic prompts.")
            if median_novelty < 0.40:
                print(f"[{ch_name}] WARNING: Median novelty is low ({median_novelty:.3f}). "
                      f"Coverage may be nearing saturation for this conversation set.")

        print(f"[{ch_name}] Filter result complete: {len(kept)} kept")

        all_kept[ch_name] = kept

    # Combine all channel results for reporting
    kept = []
    for channel_kept in all_kept.values():
        kept.extend(channel_kept)

    # ------------------------------------------------------------------
    # Write intent-grouped output into out/<date>_<serial>/existing_intents|new_intents/
    # ------------------------------------------------------------------
    existing_intents = {t.get("intent", "").lower() for t in testset}
    existing_intents.discard("")
    intent_output_dir = _write_intent_grouped_output(all_kept, existing_intents, out_dir)

    report_lines: List[str] = []
    today = date.today().isoformat()
    report_lines.append("# Test Gap Miner Report")
    report_lines.append(f"\nGenerated: {today}")
    report_lines.append(f"\nTotal suggested tests: **{len(kept)}**")

    # --- Per-channel summary ---
    report_lines.append("\n## Suggestions by Channel")
    for ch_name, ch_kept in all_kept.items():
        report_lines.append(f"- **{ch_name}**: {len(ch_kept)} suggestions")

    # --- Coverage delta ---
    new_intents = {s.get("intent", "").lower() for s in kept}
    new_intents.discard("")
    added_intents = new_intents - existing_intents
    report_lines.append("\n## Coverage Delta")
    report_lines.append(f"- Existing test cases: {len(testset)}")
    report_lines.append(f"- Existing intents/domains: {len(existing_intents)}")
    report_lines.append(f"- Intents in suggestions: {len(new_intents)}")
    report_lines.append(f"- Net-new intents: {len(added_intents)}")
    if added_intents:
        for ai in sorted(added_intents):
            report_lines.append(f"  - `{ai}`")

    # --- By channel ---
    existing_channels = Counter(t.get("channel", "unknown") for t in testset)
    report_lines.append("\n## Existing Test Coverage by Channel")
    for ch, cnt in existing_channels.most_common():
        report_lines.append(f"- {ch}: {cnt}")

    # --- By domain ---
    existing_domains = Counter(
        t.get("metadata", {}).get("domain", t.get("metadata", {}).get("category", "unknown"))
        for t in testset
    )
    report_lines.append("\n## Existing Test Coverage by Domain")
    for dom, cnt in existing_domains.most_common():
        report_lines.append(f"- {dom}: {cnt}")

    # --- By intent ---
    by_intent: Dict[str, int] = {}
    for s in kept:
        intent = s.get("intent", "unknown")
        by_intent[intent] = by_intent.get(intent, 0) + 1
    report_lines.append("\n## By Intent")
    for intent, count in sorted(by_intent.items(), key=lambda x: -x[1]):
        report_lines.append(f"- {intent}: {count}")

    # --- By language ---
    by_lang = Counter(s.get("language", "unknown") for s in kept)
    report_lines.append("\n## By Language")
    for lang, count in by_lang.most_common():
        report_lines.append(f"- {lang}: {count}")

    # --- Multi-turn summary ---
    multi_turn_tests = [s for s in kept if s.get("is_multi_turn", False)]
    single_turn_tests = [s for s in kept if not s.get("is_multi_turn", False)]
    report_lines.append(f"\n## Multi-Turn Coverage")
    report_lines.append(f"- Single-turn test cases: {len(single_turn_tests)}")
    report_lines.append(f"- Multi-turn test cases: {len(multi_turn_tests)}")
    if multi_turn_tests:
        avg_turns = sum(s.get("turn_count", 1) for s in multi_turn_tests) / len(multi_turn_tests)
        max_turns = max(s.get("turn_count", 1) for s in multi_turn_tests)
        report_lines.append(f"- Avg turns per multi-turn case: {avg_turns:.1f}")
        report_lines.append(f"- Max turns in a single case: {max_turns}")

    # --- Failure-derived tests ---
    failure_tests = [
        s for s in kept
        if any("failure" in r.lower() or "agent" in r.lower() for r in s.get("gap_reason", []))
    ]
    report_lines.append(f"\n## Failure-Derived Tests ({len(failure_tests)})")
    for ft in failure_tests:
        report_lines.append(f"- **{ft['id']}** | {ft.get('intent')} | novelty={ft.get('novelty_score', 0):.2f}")

    # --- Multi-turn structural patterns (#7) ---
    mt_pattern_tests = [s for s in kept if s.get("multi_turn_patterns")]
    if mt_pattern_tests:
        report_lines.append(f"\n## Multi-Turn Structural Patterns ({len(mt_pattern_tests)})")
        for mt in mt_pattern_tests:
            patterns = ", ".join(mt.get("multi_turn_patterns", []))
            report_lines.append(
                f"- **{mt['id']}** | {mt.get('intent')} | patterns: {patterns} | "
                f"novelty={mt.get('novelty_score', 0):.2f}"
            )

    # --- Combinatorial novelty summary (#3) ---
    combo_tests = [s for s in kept if s.get("unseen_combos")]
    if combo_tests:
        report_lines.append(f"\n## Combinatorial Novelty ({len(combo_tests)})")
        for ct in combo_tests:
            combos = ", ".join(ct.get("unseen_combos", []))
            report_lines.append(
                f"- **{ct['id']}** | {ct.get('intent')} | unseen: {combos} | "
                f"novelty={ct.get('novelty_score', 0):.2f}"
            )

    # --- High constraint density (#6) ---
    high_constraint_tests = [
        s for s in kept
        if s.get("constraint_count", 0) >= config["scoring"].get("constraint_density_threshold", 4)
    ]
    if high_constraint_tests:
        report_lines.append(f"\n## High Constraint Density ({len(high_constraint_tests)})")
        for hc in high_constraint_tests:
            report_lines.append(
                f"- **{hc['id']}** | {hc.get('intent')} | constraints={hc.get('constraint_count', 0)} | "
                f"novelty={hc.get('novelty_score', 0):.2f}"
            )

    # --- Accept / reject checklist (per channel) ---
    for ch_name, ch_kept in all_kept.items():
        if not ch_kept:
            continue
        report_lines.append(f"\n## Accept / Reject Checklist -- {ch_name.upper()}")
        report_lines.append("| # | ID | Intent | Newness | Novelty | Impact | Turns | Structural | Constraints | Closest Existing Test | Accept? |")
        report_lines.append("|---|-----|--------|---------|---------|--------|-------|------------|-------------|----------------------|---------|")
        for i, s in enumerate(ch_kept, 1):
            nearest = s.get("nearest_existing_tests", [])
            closest = nearest[0] if nearest else {}
            closest_txt = f"{closest.get('id', '--')} ({closest.get('similarity_pct', '--')}, {closest.get('intent', '')})"
            turn_count = s.get("turn_count", 1)
            turn_label = f"{turn_count} {'(M)' if s.get('is_multi_turn') else ''}"
            structural = "Y" if s.get("has_structural_novelty") else "-"
            constraints = str(s.get("constraint_count", 0))
            report_lines.append(
                f"| {i} | {s['id']} | {s.get('intent', '')} | "
                f"{s.get('newness_pct', 0)}% | "
                f"{s.get('novelty_score', 0):.2f} | "
                f"{s.get('impact_score', 0):.2f} | "
                f"{turn_label} | "
                f"{structural} | "
                f"{constraints} | "
                f"{closest_txt} | [ ] |"
            )

    # --- Metrics summary ---
    new_entity_combos = set()
    for s in kept:
        ents = s.get("entities", {})
        if ents:
            new_entity_combos.add(tuple(sorted(ents.keys())))
    added_entity_combos = new_entity_combos - known_entity_combos

    failure_rate = len(failure_tests) / len(kept) * 100 if kept else 0.0

    # Saturation metrics for the report (#9)
    all_novelty_scores = [s.get("novelty_score", 0) for s in kept]
    if all_novelty_scores:
        sorted_novelties = sorted(all_novelty_scores)
        median_novelty_all = sorted_novelties[len(sorted_novelties) // 2]
        high_novelty_all = sum(1 for n in all_novelty_scores if n > 0.50)
        high_novelty_all_pct = (high_novelty_all / len(all_novelty_scores)) * 100
        edge_scraping_all = sum(1 for n in all_novelty_scores if 0.35 <= n <= 0.45)
        edge_scraping_all_pct = (edge_scraping_all / len(all_novelty_scores)) * 100
    else:
        median_novelty_all = 0.0
        high_novelty_all_pct = 0.0
        edge_scraping_all_pct = 0.0

    report_lines.append("\n## Metrics")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| Novelty mode | {novelty_mode} |")
    report_lines.append(f"| Coverage Delta (intents) | +{len(added_intents)} |")
    report_lines.append(f"| Entity combo coverage | {len(known_entity_combos)} existing -> +{len(added_entity_combos)} new |")
    report_lines.append(f"| Language coverage | {', '.join(by_lang.keys())} |")
    report_lines.append(f"| Failure-to-test rate | {failure_rate:.1f}% |")
    report_lines.append(f"| Median novelty | {median_novelty_all:.3f} |")
    report_lines.append(f"| Suggestions with novelty > 0.50 | {high_novelty_all_pct:.1f}% |")
    report_lines.append(f"| Edge-scraping (0.35-0.45) | {edge_scraping_all_pct:.1f}% |")
    report_lines.append(f"| Structural novelty count | {sum(1 for s in kept if s.get('has_structural_novelty'))} |")
    report_lines.append(f"| Multi-turn patterns found | {len(mt_pattern_tests)} |")
    report_lines.append(f"| Combinatorial novelty found | {len(combo_tests)} |")
    report_lines.append(f"| High constraint density | {len(high_constraint_tests)} |")
    for ch_name, ch_kept in all_kept.items():
        report_lines.append(f"| {ch_name} suggestions | {len(ch_kept)} |")

    # Saturation warning in report
    if edge_scraping_all_pct > 50:
        report_lines.append("\n> **WARNING**: >50% of suggestions are edge-scraping (novelty 0.35-0.45). "
                            "Consider switching to `exploration` mode or adding adversarial/synthetic conversation data.")
    if median_novelty_all < 0.40 and kept:
        report_lines.append(f"\n> **WARNING**: Median novelty is low ({median_novelty_all:.3f}). "
                            "Coverage may be nearing saturation for this conversation set.")

    # --- Intent-grouped output summary ---
    report_lines.append(f"\n## Intent-Grouped Output")
    report_lines.append(f"- Output directory: `{intent_output_dir.relative_to(root)}`")

    dir_existing = intent_output_dir / "existing_intents"
    dir_new = intent_output_dir / "new_intents"

    # Collect files per channel subfolder
    def _collect_channel_files(base_dir: Path) -> Dict[str, List[Path]]:
        result: Dict[str, List[Path]] = {}
        if not base_dir.exists():
            return result
        for ch_dir in sorted(base_dir.iterdir()):
            if ch_dir.is_dir():
                files = sorted(ch_dir.glob("*.json"))
                if files:
                    result[ch_dir.name] = files
        return result

    existing_by_channel = _collect_channel_files(dir_existing)
    new_by_channel = _collect_channel_files(dir_new)

    total_existing = sum(len(f) for f in existing_by_channel.values())
    total_new = sum(len(f) for f in new_by_channel.values())
    report_lines.append(f"- `existing_intents/` — {total_existing} intent files across {len(existing_by_channel)} channel(s)")
    report_lines.append(f"- `new_intents/` — {total_new} intent files across {len(new_by_channel)} channel(s)")

    for category, by_channel in [("existing_intents", existing_by_channel), ("new_intents", new_by_channel)]:
        if by_channel:
            report_lines.append(f"\n### {category}/")
            for ch_name, files in by_channel.items():
                report_lines.append(f"\n#### {ch_name}/")
                report_lines.append("| File | Test Cases |")
                report_lines.append("|------|------------|")
                for fp in files:
                    try:
                        with fp.open("r", encoding="utf-8") as f:
                            count = len(json.load(f))
                    except (json.JSONDecodeError, OSError):
                        count = 0
                    report_lines.append(f"| {fp.name} | {count} |")

    # --- OpenAI API usage summary ---
    usage = TokenTracker.instance().summary()
    report_lines.append("\n## OpenAI API Usage")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| Chat prompt tokens | {usage['prompt_tokens']:,} |")
    report_lines.append(f"| Chat completion tokens | {usage['completion_tokens']:,} |")
    report_lines.append(f"| Embedding tokens | {usage['embedding_tokens']:,} |")
    report_lines.append(f"| **Total tokens** | **{usage['total_tokens']:,}** |")
    report_lines.append(f"| Estimated cost (USD) | ${usage['estimated_cost_usd']:.4f} |")
    report_lines.append(f"| NLU API calls | {usage['api_calls'].get('nlu', 0)} |")
    report_lines.append(f"| NLU cache hits (saved) | {usage['api_calls'].get('nlu_cache_hit', 0)} |")
    report_lines.append(f"| Test-gen API calls | {usage['api_calls'].get('testgen', 0)} |")
    report_lines.append(f"| Embedding API calls | {usage['api_calls'].get('embedding', 0)} |")
    report_lines.append(f"| Trivial turns skipped | {usage['trivial_turns_skipped']} |")

    print(f"\n{'='*60}")
    print(f"  OpenAI API Usage Summary")
    print(f"  Total tokens: {usage['total_tokens']:,}")
    print(f"  Estimated cost: ${usage['estimated_cost_usd']:.4f}")
    print(f"  NLU calls: {usage['api_calls'].get('nlu', 0)} "
          f"(+{usage['api_calls'].get('nlu_cache_hit', 0)} memo hits)")
    print(f"  TestGen calls: {usage['api_calls'].get('testgen', 0)}")
    print(f"  Embedding calls: {usage['api_calls'].get('embedding', 0)}")
    print(f"  Trivial skipped: {usage['trivial_turns_skipped']}")
    print(f"{'='*60}")

    # Write report into the run folder only
    run_report = out_dir / "report.md"
    with run_report.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved: {run_report}")

if __name__ == "__main__":
    run()
