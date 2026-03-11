# Test Gap Miner

An AI-powered pipeline that analyzes real property-management conversations (chat & email) against an existing test suite and generates new test case suggestions for uncovered scenarios.

---

## How It Works

```
Real Conversations ──►  NLU Extraction  ──►  Gap Analysis  ──►  Test Generation
  (chat & email)       (intent, lang,      (FAISS similarity    (GPT-4o-mini
                        entities, flags)    + novelty scoring)    structured output)
                                                  │
                                 ┌────────────────┘
                                 ▼
                          Existing Test Set
                          (1,233 cases)
```

1. **Parse** — Splits raw conversation logs into user/agent turn pairs.
2. **NLU** — Classifies each turn: intent, language, policy flags, entities (GPT-4o-mini). Uses taxonomy sub-intents and normalizes to parent domain.
3. **Session-Level Intent Resolution** — Resolves spurious `NEW_*` intents by checking the session's dominant taxonomy intent. Ensures intents reflect the overall conversation topic, not a single turn's detail.
4. **Embedding & Search** — Embeds each turn and finds its 5 nearest neighbors in the existing test index (FAISS + `text-embedding-3-small`).
5. **Gap Analysis** — Computes a **novelty score** and **impact score** for each turn. Detects multi-turn patterns, constraint density, and combinatorial novelty.
6. **Test Generation** — For each gap candidate, selects a test type (policy_flow / edge_case / stress), builds a slot plan, and generates a structured test case with conversation flow and acceptance criteria.
7. **Filtering** — Applies the user-selected mode and filter style to decide which suggestions to keep.
8. **Output Organization** — Writes results into a date-based run folder, grouped by channel and intent.

---

## Pipeline Deep Dive

### Phase 0 — Initialization

1. Load `config/settings.yaml` and validate via Pydantic (`PipelineConfig`).
2. Load all existing test cases from `data/testsets/` (supports single JSON or `channel/intent/` directory tree).
3. Normalize each test case to a standard schema: `utterance`, `intent`, `channel`, `entities`.
4. Build or load the **FAISS embedding index** from `.cache/`. The index embeds every existing test utterance using `text-embedding-3-small` (1536 dims, cosine similarity). A SHA-256 hash of the test set is used as a cache key — the index is rebuilt only when the test set changes.
5. Extract known **(intent × dimension)** combinations from the test set: intent×channel, intent×policy, intent×language, intent×escalation. These are used later to detect combinatorial novelty.

### Phase 1 — Conversation Parsing

`conversation_parser.py` converts raw chat/email logs into structured turns:

```
Raw text log  ──►  Session boundary detection  ──►  Turn extraction  ──►  Noise filtering
```

- **Session detection**: Splits on explicit `Session <ID>` headers, or double-blank-line boundaries.
- **Role detection**: Identifies `User:` / `Agent:` / `Bot:` / `Assistant:` / `Support:` prefixes.
- **HTML/CSS cleaning**: Strips `<!-- comments -->`, HTML tags, `@font-face`, `mso-*` properties, email boilerplate.
- **Noise filtering**: Removes trivial acknowledgments ("ok", "thanks"), email headers, standalone URLs, unsubscribe footers.
- **Multi-turn context**: Each turn carries a `prior_turns` array of all preceding turns in the same session.

**Output per turn:**
```json
{
  "session_id": "AUTO-2026-03-10-0001",
  "turn_index": 0,
  "user_text": "My sink is leaking badly",
  "agent_text": "I'm sorry to hear that. Can you tell me your unit number?",
  "prior_turns": []
}
```

### Phase 2 — NLU Extraction

Each non-trivial turn is sent to GPT-4o-mini for classification:

| Field | Description | Example |
|-------|-------------|---------|
| `intent` | Domain-level intent from taxonomy or `NEW_CamelCase` | `service_request`, `NEW_RentIncrease` |
| `entities` | Extracted key-value pairs (dates → ISO-8601, amounts → numeric) | `{"unit": "402", "issue": "leaking sink"}` |
| `language` | ISO 639-1 language code | `en`, `es` |
| `sentiment` | `positive`, `neutral`, or `negative` | `negative` |
| `urgency` | `low`, `medium`, or `high` | `high` |
| `confidence` | NLU confidence (0.0–1.0) | `0.92` |
| `flags` | Detected patterns | `["policy", "agent_failure"]` |

**Flags detected:**
- `multi_intent` — User asks about multiple topics
- `policy` — Touches policy-sensitive areas (fair housing, payments)
- `agent_failure` — Agent response appears wrong or incomplete
- `code_switching` — User mixes languages
- `ambiguous` — Intent unclear
- `typo` — Spelling/grammar errors

Results are **memoized** per utterance (thread-safe) to avoid duplicate API calls within a run.

### Phase 3 — Gap Analysis

For each NLU-classified turn, gap analysis determines *what is missing* from the existing test set:

```
Turn + NLU  ──►  FAISS search (top-5 neighbors)  ──►  Novelty scoring  ──►  Candidate decision
                                                  ──►  Impact scoring
                                                  ──►  Multi-factor detection
```

**Novelty score** (0.0–1.0):
$$\text{novelty} = 0.6 \times (1 - \text{max\_sim}) + 0.4 \times (1 - \text{avg\_sim})$$

**Gap candidate criteria** (any one triggers candidacy):

| # | Criterion | What it catches |
|---|-----------|-----------------|
| 1 | Low similarity | `max_sim < 0.70` — semantically distant from all existing tests |
| 2 | Unseen entity combo | First time this combination of entity keys appears |
| 3 | Combinatorial novelty | New intent×channel, intent×policy, intent×language, or intent×escalation pair |
| 4 | Policy sensitivity | Utterance touches policy-sensitive areas |
| 5 | Multi-intent | User asks about 2+ topics in one message |
| 6 | Agent failure | Agent gave wrong/incomplete/contradictory response |
| 7 | Edge-case heuristics | Numeric extremes, date boundaries, code-switching |
| 8 | High constraint density | ≥ 4 constraints in a single turn |
| 9 | Multi-turn structural patterns | Escalation transitions, agent reversals, user restatements, state corrections |

**Paraphrase rejection**: If same intent + same entities + novelty < 0.50 + no structural novelty → rejected as a paraphrase.

### Phase 4 — Test Case Generation

This is where the actual test case content is created. Three systems work together:

#### Step 1: Select Test Type

Each gap candidate is randomly assigned a test type based on configurable weights:

| Type | Default % | Difficulty | Generated Conversation Style |
|------|-----------|------------|------------------------------|
| `policy_flow` | 70% | easy | Happy-path: user answers correctly, agent follows slot plan exactly |
| `edge_case` | 20% | medium | User deviates: vague answers, wrong types, corrections, partial info |
| `stress` | 10% | hard | Disruptions: off-topic, multi-intent, frustrated user, irrelevant info |

#### Step 2: Build Slot Plan

The **slot planner** defines the conversation policy — what questions the agent must ask, in what order, and what answer types to expect:

```
Intent  ──►  Static lookup (slot_policies.json)  ──►  Slot plan
             OR
        ──►  LLM-generated plan (NEW_* intents)  ──►  Slot plan (cached)
```

**Static policies** are defined for all 12 known intents in `data/slot_policies.json`. Example for `service_request`:

```
1. [open_ended]   service_type  (REQUIRED) — "What type of service do you need?"
2. [open_ended]   location      (REQUIRED) — "Where in the unit is the issue?"
3. [enumerated]   urgency       (REQUIRED) — options: [emergency, high, normal]
4. [boolean]      access_perm   (REQUIRED) — "Can maintenance enter your unit?"
5. [open_ended]   preferred_date (optional)
6. [open_ended]   additional_details (optional)
FINAL: [confirmation] — "I'll submit a service request for {service_type} at {location}. Correct?"
```

**Slot question types:**
- `boolean` → yes/no question only
- `enumerated` → presents fixed options (e.g., "Is this emergency, high, or normal?")
- `open_ended` → free-text answer
- `confirmation` → final verification step

For `NEW_*` intents with no static policy, the LLM generates a dynamic plan (1–6 slots, logical ordering). Dynamic plans are **cached per-run** (thread-safe).

#### Step 3: Generate Test Case via LLM

The system prompt is assembled from three layers:

```
Base prompt (QA test designer role)
  + Slot plan (conversation policy with STRICT RULES)
  + Test-type overlay (policy_flow | edge_case | stress instructions)
  + Quality rules (acceptance criteria, priority, category requirements)
```

**For edge_case**, the LLM is instructed to choose ONE user deviation:
- Incomplete answer ("I don't know")
- Wrong answer type ("maybe" to a boolean slot)
- User correction ("Sorry, I meant 420")
- Vague/ambiguous answer
- Partial multi-slot answer

**For stress**, the LLM is instructed to choose ONE disruption:
- Off-topic injection ("Also when does the pool open?")
- Multi-intent message (two issues at once)
- Frustrated user ("I already told you that!")
- Irrelevant context/attachments

The LLM call uses `gpt-4o-mini` at temperature 0.2, returning structured JSON.

#### Step 4: Validate Against Slot Plan

**Only for `policy_flow` tests** (edge_case/stress intentionally deviate):

| Check | Rule |
|-------|------|
| Required slots | Each required slot must be addressed in the agent's turns |
| Boolean format | Boolean slots must use yes/no phrasing, not open-ended questions |
| Enumerated options | Enumerated slots must present the listed options |
| Confirmation step | If plan requires confirmation, last agent turn must confirm |

If validation fails, the LLM is re-prompted with the violation list (up to 1 retry). On final failure, the test case is accepted but tagged with `slot_plan_violations`.

#### Step 5: Assemble Suggestion

The final test suggestion combines NLU data, gap analysis, and generated content:

```json
{
  "id": "SUG-session123-0001",
  "intent": "service_request",
  "channel": "chat",
  "test_type": "edge_case",
  "difficulty": "medium",
  "reason": "Tests agent handling when user gives vague location for maintenance request",
  "novelty_score": 0.6234,
  "impact_score": 0.35,
  "gap_reason": ["unseen_entity_combo", "policy_sensitive"],
  "conversation": [
    {"role": "user", "message": "My sink is leaking"},
    {"role": "assistant", "message": "What type of service do you need?"},
    {"role": "user", "message": "I think it's plumbing... not sure"},
    {"role": "assistant", "message": "No problem. Where in the unit is the issue?"},
    {"role": "user", "message": "Somewhere in the kitchen I think"}
  ],
  "acceptance_criteria": [
    "Agent asks clarifying question about exact location",
    "Agent presents urgency options: emergency, high, normal",
    "Agent asks boolean yes/no for maintenance access"
  ],
  "expected_behavior": "Agent handles vague answers gracefully and re-asks for specifics",
  "priority": "P1",
  "category": "edge_case",
  "tags": ["maintenance", "vague_answer", "generated"]
}
```

### Phase 5 — Filtering & Deduplication

Before writing output, suggestions are filtered through multiple gates:

1. **Dedup vs. existing tests**: Reject if cosine similarity > 0.90 to any existing test.
2. **Dedup among suggestions**: Reject if same intent + similar novelty + high token overlap.
3. **Novelty threshold** (depends on mode):
   - Conservative: novelty ≥ 0.48
   - Exploration: novelty ≥ 0.45
4. **Impact override** (conservative only): Keep if novelty ≥ 0.42 AND impact ≥ 0.30.
5. **Combo bypass** (if "Combination of New" filter selected): Keep unseen intent×channel combos and `NEW_*` intents above a floor (conservative ≥ 0.35, exploration ≥ 0.25).

### Phase 6 — Output

Results are written to `out/<YYYY-MM-DD>_<NNN>/`:

```
out/2026-03-10_001/
  existing_intents/
    chat/complaints.json
    email/payments.json
  new_intents/
    chat/NEW_rent_increase.json
    email/NEW_security_deposit.json
  report.md
```

A `report.md` is generated with:
- Per-channel statistics (turns parsed, candidates found, suggestions kept)
- Novelty score distribution breakdown
- Intent-grouped output summary with per-channel file counts
- OpenAI API usage and cost estimate

---

## Interactive CLI

When you run the pipeline, an interactive menu guides you through four choices:

```
══════════════════════════════════════════════════
   T E S T   G A P   M I N E R
══════════════════════════════════════════════════

  Select novelty mode
    1. Conservative
    2. Exploration

  Select filtering approach
    1. Only Strict New
    2. Combination of New

  Do you want to clear the suggestion cache? (y/n)
  Is the test set modified since last run? (y/n)
```

---

## Modes Explained

### Conservative vs. Exploration

These terms follow their **conventional meanings** in strategy and decision-making:

| Aspect | Conservative | Exploration |
|--------|-------------|-------------|
| **Philosophy** | "Be strict — only show me what's truly new" | "Explore broadly — don't miss any potential gap" |
| **Novelty threshold** | ≥ 0.50 (higher bar) | ≥ 0.45 (lower bar) |
| **Impact override** | No — novelty threshold is strict, no overrides | Yes — high-impact scenarios (policy violations, failures) with novelty ≥ 0.30 are kept even if below 0.45 |
| **Structural gate** | Requires structural novelty for all suggestions | Rejects paraphrases below 0.50 novelty |
| **Result volume** | Fewer suggestions (~2 chat at strict) | More suggestions (~48–55 chat) |
| **Best for** | Mature test suites, high-precision, zero noise | Initial gap analysis, thorough coverage, catching edge cases |

**Why "Conservative"?** — You're being cautious about what you accept. Higher threshold, stricter filtering, fewer results, lower risk of including noise. Like a conservative investor — only safe bets.

**Why "Exploration"?** — You're casting a wider net to discover more gaps. Lower threshold, impact overrides, more variety. Like an explorer — you'd rather find something imperfect than miss something important.

### Only Strict New vs. Combination of New

| Aspect | Only Strict New | Combination of New |
|--------|----------------|-------------------|
| **Combo bypass** | OFF — every suggestion must pass the mode's novelty threshold on its own | ON — suggestions with new intent names or unseen intent+channel combinations get an automatic pass. Floor depends on mode: conservative ≥ 0.35, exploration ≥ 0.25 |
| **What it catches** | Only semantically distinct scenarios | Semantically distinct scenarios + all new intent labels + all new intent+channel pairs |
| **Result volume** | Fewer (strict threshold only) | More (threshold + combo bypass) |
| **Best for** | When you want zero overlap with existing tests | When you want to ensure every new intent and channel combination has coverage |

---

## All Four Combinations

| # | Mode | Filter | Chat | Email | Total | Description |
|---|------|--------|------|-------|-------|-------------|
| 1 | Conservative | Strict New | ~2 | ~0 | **~2** | **Most strict.** Only truly unique scenarios (novelty ≥ 0.48). Zero compromise. |
| 2 | Conservative | Combination of New | varies | varies | **varies** | Strict threshold + combo bypass with higher floor (novelty ≥ 0.35). Fewer combos kept than exploration. |
| 3 | Exploration | Strict New | varies | varies | **varies** | Lower threshold (0.45) + impact override, but no combo bypass. Balanced coverage. |
| 4 | Exploration | Combination of New | varies | varies | **varies** | **Most inclusive.** Lower threshold + impact override + combo bypass (floor 0.25). |

```
Strictness:  Most ◄─────────────────────────────────────────► Least

  Conservative      Conservative       Exploration        Exploration
  + Strict          + Combo            + Strict           + Combo
     ~2                ~69                ~62                ~71
     │                  │                  │                  │
     │                  │                  │                  └─ Everything + new combos
     │                  │                  └─ Relaxed novelty + impact override
     │                  └─ Strict novelty BUT keep all new intents/combos
     └─ Only truly unique scenarios (novelty ≥ 0.48)
```

---

## Novelty Score

Each conversation turn gets a novelty score (0.0 – 1.0) measuring how different it is from the existing test set:

```
novelty = 0.6 × (1 - max_similarity) + 0.4 × (1 - avg_similarity)
```

- **max_similarity**: Cosine similarity to the single closest existing test
- **avg_similarity**: Average similarity to the top-5 nearest tests
- Weights: 60% max, 40% average (configurable in `settings.yaml`)

| Novelty | Meaning |
|---------|---------|
| 0.00 | Identical to an existing test |
| 0.25 | Very similar — likely a paraphrase |
| 0.45 | Moderately different — conservative threshold |
| 0.50 | Meaningfully different — exploration threshold |
| 0.75+ | Highly novel — rarely seen |
| 1.00 | Completely unlike anything in the test set |

---

## Impact Score

Bonus points that make a suggestion more likely to be kept (especially in conservative mode):

| Factor | Bonus | Description |
|--------|-------|-------------|
| Policy-related | +0.20 | Touches policy (payments, fair housing, etc.) |
| Agent failure | +0.20 | Agent gave wrong/incomplete response |
| New entity combo | +0.10 | Unseen entity combination |
| Language variant | +0.10 | Non-English conversation |
| New intent+channel pair | +0.15 | e.g., `parking` + `email` never tested |
| New intent+channel+policy triple | +0.25 | Three-way combo never tested |
| High constraint density | +0.15 | ≥ 4 constraints (entities, policy, multi-turn, language) |

---

## Project Structure

```
├── .env                           # OpenAI API key (not committed)
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml              # All configurable thresholds and model settings
├── data/
│   ├── taxonomy.json              # Intent/entity taxonomy for NLU classification
│   ├── slot_policies.json         # Slot plans per intent (conversation policies)
│   ├── conversations/             # Raw conversation logs for analysis
│   │   ├── chat_logs.txt
│   │   └── email_logs.txt
│   └── testsets/                  # Existing test cases (1,238) by channel & intent
│       ├── chat/                  # 12 intent JSON files
│       │   ├── complaints.json
│       │   ├── events.json
│       │   ├── fair_housing.json
│       │   ├── handoff.json
│       │   ├── language_support.json
│       │   ├── packages.json
│       │   ├── parking.json
│       │   ├── payments.json
│       │   ├── prompt_injection.json
│       │   ├── property_info.json
│       │   ├── self_service.json
│       │   └── service_request.json
│       └── email/                 # 12 intent JSON files
│           ├── complaints.json
│           ├── events.json
│           ├── fair_housing.json
│           ├── handoff.json
│           ├── language_support.json
│           ├── packages.json
│           ├── parking.json
│           ├── payments.json
│           ├── prompt_injection.json
│           ├── property_info.json
│           ├── self_service.json
│           └── service_request.json
├── src/
│   ├── __init__.py
│   ├── __main__.py                # Package entrypoint (`python -m src`)
│   ├── main.py                    # Pipeline orchestrator + interactive CLI
│   ├── models.py                  # Pydantic models + token tracking
│   ├── pipeline/
│   │   ├── conversation_parser.py # Conversation log parser
│   │   ├── nlu.py                 # NLU extraction (GPT-4o-mini)
│   │   ├── gap_analysis.py        # Novelty/impact scoring, gap detection
│   │   ├── test_generation.py     # Test case generation (3 types) + deduplication
│   │   ├── slot_planner.py        # Slot-plan builder for policy-driven generation
│   │   └── pii.py                 # PII detection utilities
│   └── data/
│       ├── dataset_loader.py      # Test set loading (file or directory) + hashing
│       └── embedding_index.py     # FAISS embedding index management
├── scripts/
│   └── check_api.py               # OpenAI API health check utility
├── reports/                       # Evaluation reports (reference)
├── .cache/                        # Auto-generated: embeddings + suggestion caches
└── out/                           # Auto-generated: pipeline output
    └── <YYYY-MM-DD>_NNN/          # Date-based run folder (e.g. 2026-03-10_001/)
        ├── existing_intents/      # Test cases for known taxonomy intents
        │   ├── chat/
        │   └── email/
        ├── new_intents/           # Test cases for newly discovered intents
        │   ├── chat/
        │   └── email/
        └── report.md              # Run analysis report
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
#    Create a .env file in the project root:
echo OPENAI_API_KEY=sk-your-key-here > .env

# 3. Run the pipeline
python -m src

# Optional: verify OpenAI API setup
python scripts/check_api.py
```

---

## Output Organization

Every run creates a date-based folder under `out/` with an auto-incrementing serial number:

```
out/
  2026-03-10_001/              ← first run of the day
    existing_intents/          ← test cases for known taxonomy intents
      chat/
        complaints.json
        payments.json
      email/
        complaints.json
        handoff.json
    new_intents/               ← test cases for newly discovered intents
      chat/
        NEW_rent_increase.json
      email/
        NEW_security_deposit.json
    report.md
  2026-03-10_002/              ← second run same day
    ...
```

Inside each run folder, test cases are split into:
- **`existing_intents/<channel>/`** — One file per known taxonomy intent, grouped by channel (e.g., `chat/payments.json`).
- **`new_intents/<channel>/`** — One file per genuinely new intent, grouped by channel (prefixed with `NEW_`).

Each test case JSON object includes `channel`, `test_type`, and `difficulty` fields.

**Previous runs are never overwritten.**

---

## Test Types

Every generated test case is assigned one of three types (configurable distribution in `settings.yaml`):

| Type | Default % | Difficulty | Purpose |
|------|-----------|------------|---------|
| `policy_flow` | 70% | easy | Clean slot-following happy-path conversation |
| `edge_case` | 20% | medium | User deviations: incomplete/vague/wrong answers, corrections |
| `stress` | 10% | hard | Off-topic, multi-intent, frustrated users, irrelevant info |

**policy_flow** — Validates the agent correctly collects all required slots in order, uses proper question types, and confirms at the end.

**edge_case** — Tests agent recovery when the user:
- Gives an incomplete answer ("I don't know")
- Provides the wrong answer type ("maybe" to a yes/no question)
- Corrects themselves ("Sorry, I meant 420")
- Gives vague or partial information

**stress** — Tests agent reasoning under pressure:
- Off-topic interjections ("Also when does the pool open?")
- Multi-intent messages (two issues in one message)
- Frustrated/emotional users ("I already told you that!")
- Irrelevant context that should be filtered

Configure the distribution:
```yaml
# config/settings.yaml
test_types:
  policy_flow: 0.70
  edge_case: 0.20
  stress: 0.10
```

---

## Caching

The pipeline caches results at multiple levels to avoid redundant API calls:

| Cache | What it stores | Invalidated when |
|-------|---------------|------------------|
| **FAISS index** | Embeddings of all 1,233 existing tests | Any file in `testset_by_channel/` changes (hash-based) |
| **Turn cache** | Per-turn NLU + gap + testgen results | Conversation file content changes |
| **Suggestion cache** | Pre-filter suggestions (post-dedup) | Conversation or testset content changes |

Changing **mode** or **filter style** does NOT require clearing the cache — the filter is re-applied instantly from cached pre-filter suggestions.

Use "Clear cache" in the CLI (or delete files in `.cache/`) when you want to force full re-analysis.

Use "Testset modified" when you've added/changed test cases in `testset_by_channel/` to force an index rebuild.

---

## Configuration

All thresholds are configurable in `config/settings.yaml`:

```yaml
novelty:
  exploration_min_novelty: 0.50   # Conservative mode threshold (strict)
  keep_min_novelty: 0.45          # Exploration mode threshold (broad)
  combo_bypass_min_novelty: 0.25  # Floor for combo bypass (exploration)
  conservative_combo_bypass_min_novelty: 0.35  # Floor for combo bypass (conservative)
  novelty_max_weight: 0.6         # Weight for max similarity
  novelty_avg_weight: 0.4         # Weight for avg similarity
  paraphrase_novelty_ceiling: 0.50

scoring:
  impact_policy: 0.2
  impact_failure: 0.2
  impact_new_pair: 0.15
  impact_new_triple: 0.25
```

---

## Tech Stack

- **Python 3.11**
- **OpenAI GPT-4o-mini** — NLU classification + test case generation
- **OpenAI text-embedding-3-small** — 1,536-dim embeddings for similarity search
- **FAISS** — Fast approximate nearest neighbor search
- **Pydantic** — Configuration validation
- **tqdm** — Progress bars





Test Case Generation Pipeline

The system generates new evaluation test cases using a 6-phase pipeline that processes conversation logs, identifies gaps in existing tests, and generates structured conversations to evaluate the agent.

Phase 0 – Initialization

Component: main.py

Loads configuration and system settings

Loads the existing test dataset (~1,200 tests)

Builds or loads the FAISS vector index for similarity search

Extracts known intent × dimension combinations from the existing test set

This prepares the system for identifying coverage gaps.

Phase 1 – Conversation Parsing

Component: conversation_parser.py

Raw conversation logs (chat/email) are converted into structured turns.

Processing includes:

HTML and CSS stripping for email conversations

Removing formatting noise (styles, comments, templates)

Extracting clean user/agent turns

Preserving session context

This produces clean conversational data for analysis.

Phase 2 – NLU Analysis

Component: nlu.py

Each user turn is analyzed using GPT-4o-mini to extract:

Intent

Entities

Language

Sentiment

Urgency

Additional behavioral flags

Results are memoized/cached to avoid repeated LLM calls.

Phase 3 – Gap Analysis

Component: gap_analysis.py

The system determines whether a conversation pattern is already covered in the existing test set.

Process:

Compute embeddings

Run FAISS top-5 similarity search

Calculate a novelty score

Calculate an impact score

Candidates are evaluated against 9 criteria, including:

Low similarity to existing tests

New entity combinations

Policy coverage gaps

Multi-intent scenarios

Agent failures

Multi-turn conversation patterns

Only promising candidates move forward.

Phase 4 – Test Case Generation

Components: test_generation.py, slot_planner.py

The system generates a new test conversation using GPT-4o-mini.

Test Type Selection

Each candidate is assigned a generation type:

70% Policy Flow – standard policy-compliant conversations

20% Edge Case – unusual but realistic variations

10% Stress Tests – challenging or unexpected scenarios

Slot Planning

A slot-based conversation plan is used to guide generation.

Known intents → slot plan from slot_policies.json

New intents → slot plan generated via LLM

Each slot defines:

question type (boolean / open / enumerated / confirmation)

required fields

prompt hints

Validation

For policy flow tests:

Generated conversations are validated against the slot plan

Violations trigger one regeneration attempt

Remaining issues are tagged for visibility

Phase 5 – Filtering & Output

Component: main.py

Before saving:

Deduplicate similar tests

Apply novelty thresholds

Allow impact overrides for important cases

Allow combo bypass for new entity combinations

Accepted tests are written to:

out/<date_run>/
  existing_intents/<channel>/
  new_intents/<channel>/

A report summarizing the run is also generated.