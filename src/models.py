"""Pydantic models for validation and type-safety across the pipeline."""

from __future__ import annotations

import threading
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Pipeline configuration — validates config/settings.yaml
# ---------------------------------------------------------------------------
class OpenAIConfig(BaseModel):
    api_key_env: str = "OPENAI_API_KEY"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"


class PathsConfig(BaseModel):
    testset: str = "data/testsets"
    conversations: str = "data/conversations/"
    taxonomy: str = "data/taxonomy.json"
    slot_policies: str = "data/slot_policies.json"


class EmbeddingConfig(BaseModel):
    use_large: bool = False
    faiss_nlist: int = 100


class SimilarityConfig(BaseModel):
    top_k: int = 5
    threshold_small: float = 0.70
    threshold_large: float = 0.70
    dedup_threshold: float = 0.90


class NoveltyConfig(BaseModel):
    mode: str = "conservative"
    keep_min_novelty: float = 0.45
    keep_min_impact: float = 0.20
    impact_override_min_novelty: float = 0.30
    exploration_min_novelty: float = 0.48
    novelty_max_weight: float = 0.6
    novelty_avg_weight: float = 0.4
    paraphrase_novelty_ceiling: float = 0.50
    combo_bypass_min_novelty: float = 0.25
    conservative_combo_bypass_min_novelty: float = 0.35
    conservative_impact_override_min_novelty: float = 0.42


class ScoringConfig(BaseModel):
    impact_policy: float = 0.2
    impact_failure: float = 0.2
    impact_new_entity_combo: float = 0.1
    impact_language_variant: float = 0.1
    impact_new_pair: float = 0.15
    impact_new_triple: float = 0.25
    constraint_density_threshold: int = 4
    impact_constraint_density: float = 0.15


class MultiTurnConfig(BaseModel):
    detect_escalation: bool = True
    detect_agent_reversal: bool = True
    detect_user_restatement: bool = True
    detect_state_change: bool = True


class TestTypesConfig(BaseModel):
    policy_flow: float = 0.70
    edge_case: float = 0.20
    stress: float = 0.10


class PipelineConfig(BaseModel):
    """Top-level configuration — mirrors config/settings.yaml."""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    multi_turn: MultiTurnConfig = Field(default_factory=MultiTurnConfig)
    test_types: TestTypesConfig = Field(default_factory=TestTypesConfig)


# ---------------------------------------------------------------------------
# Thread-safe OpenAI token usage tracker (singleton)
# ---------------------------------------------------------------------------
class TokenTracker:
    """Tracks OpenAI API token consumption across threads."""

    _instance: TokenTracker | None = None
    _init_lock = threading.Lock()

    @classmethod
    def instance(cls) -> TokenTracker:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0
        self.calls: Dict[str, int] = {
            "nlu": 0, "nlu_cache_hit": 0,
            "testgen": 0, "embedding": 0,
        }
        self.trivial_skipped = 0

    # --- recording helpers ---
    def add_chat(self, usage: Any, call_type: str = "nlu") -> None:
        with self._lock:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.calls[call_type] = self.calls.get(call_type, 0) + 1

    def add_embedding(self, usage: Any) -> None:
        with self._lock:
            self.embedding_tokens += usage.total_tokens
            self.calls["embedding"] = self.calls.get("embedding", 0) + 1

    def record_trivial_skip(self) -> None:
        with self._lock:
            self.trivial_skipped += 1

    def record_nlu_cache_hit(self) -> None:
        with self._lock:
            self.calls["nlu_cache_hit"] = self.calls.get("nlu_cache_hit", 0) + 1

    # --- reporting ---
    def summary(self) -> Dict[str, Any]:
        total = self.prompt_tokens + self.completion_tokens + self.embedding_tokens
        # gpt-4o-mini pricing (per 1M tokens)
        cost_in = self.prompt_tokens * 0.15 / 1_000_000
        cost_out = self.completion_tokens * 0.60 / 1_000_000
        # text-embedding-3-small pricing
        cost_emb = self.embedding_tokens * 0.02 / 1_000_000
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": total,
            "estimated_cost_usd": round(cost_in + cost_out + cost_emb, 6),
            "api_calls": dict(self.calls),
            "trivial_turns_skipped": self.trivial_skipped,
        }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.embedding_tokens = 0
            self.calls = {
                "nlu": 0, "nlu_cache_hit": 0,
                "testgen": 0, "embedding": 0,
            }
            self.trivial_skipped = 0


# ---------------------------------------------------------------------------
# Existing test-set schema (supports both chat/email and legacy flat formats)
# ---------------------------------------------------------------------------
class ConversationMessage(BaseModel):
    role: str
    message: str = ""


class TestCaseMetadata(BaseModel):
    domain: Optional[str] = None
    category: Optional[str] = None
    difficulty: str = "medium"
    expected_intent: Optional[str] = None
    expected_action: Optional[str] = None
    tone: Optional[str] = None
    original_test_case_id: Optional[str] = None


class TestCase(BaseModel):
    id: str
    source: str = "chat"
    # Chat schema
    conversation: List[ConversationMessage] = Field(default_factory=list)
    # Email schema
    subject: Optional[str] = None
    email_thread: List[ConversationMessage] = Field(default_factory=list)
    # Metadata
    metadata: TestCaseMetadata = Field(default_factory=TestCaseMetadata)
    # Legacy flat fields (populated during normalization)
    intent: str = ""
    utterance: str = ""
    entities: Dict[str, Any] = Field(default_factory=dict)
    language: str = "en"
    channel: str = "chat"
    difficulty: str = "medium"
    expected_behavior: str = ""
    expected_answer: str = ""
    acceptance_criteria: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    last_verified: Optional[str] = None

    @field_validator("intent", mode="before")
    @classmethod
    def lowercase_intent(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v


# ---------------------------------------------------------------------------
# Parsed conversation turn
# ---------------------------------------------------------------------------
class TurnMeta(BaseModel):
    timestamp: Optional[datetime] = None
    language: str = "en"
    channel: str = "chat"


class ConversationTurn(BaseModel):
    session_id: str = "unknown"
    turn_index: int = 0
    user_text: str = ""
    agent_text: str = ""
    meta: TurnMeta = Field(default_factory=TurnMeta)


# ---------------------------------------------------------------------------
# NLU extraction result
# ---------------------------------------------------------------------------
class NluResult(BaseModel):
    intent: str = "unknown"
    entities: Dict[str, Any] = Field(default_factory=dict)
    language: str = "unknown"
    flags: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Gap analysis output
# ---------------------------------------------------------------------------
class NeighborInfo(BaseModel):
    id: Optional[str] = None
    similarity: float = 0.0


class GapInfo(BaseModel):
    is_candidate: bool = False
    reasons: List[str] = Field(default_factory=list)
    neighbors: List[NeighborInfo] = Field(default_factory=list)
    max_similarity: float = 0.0
    novelty_score: float = 0.0
    impact_score: float = 0.0
    intent: str = ""


# ---------------------------------------------------------------------------
# Suggested new test output
# ---------------------------------------------------------------------------
class SuggestedTest(BaseModel):
    id: str
    source_session: str
    user_utterance: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    language: Optional[str] = None
    novelty_score: float = 0.0
    impact_score: float = 0.0
    nearest_existing_tests: List[NeighborInfo] = Field(default_factory=list)
    gap_reason: List[str] = Field(default_factory=list)
    expected_behavior: Optional[str] = None
    expected_answer: Optional[str] = None
    acceptance_criteria: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    # Multi-turn conversation context
    conversation: List[ConversationMessage] = Field(default_factory=list)
    is_multi_turn: bool = False
    turn_count: int = 1

    @field_validator("acceptance_criteria", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [v]
        return v or []
