from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import logging
import sys
import os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)
from src.emotion.analyzer import EmotionalAppraisal, VAD

logger = logging.getLogger('AMN')

@dataclass
class MemoryEntry:
    id: str
    content: str  # Interaction text (user + agent)
    vad: VAD
    timestamp: datetime
    recency_score: float  # 1.0 recent → 0.0 old
    importance: float  # Peak-end: 1.0 if arousal>0.7 else 0.5 (locked)
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class WorkingMemory:
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.memories: List[MemoryEntry] = []  # Ordered: recent first
        self.appraiser = EmotionalAppraisal()

    def add(self, content: str, user_vad: Optional[VAD] = None) -> MemoryEntry:
        if len(self.memories) >= self.capacity:
            evicted = self.memories.pop()  # LRU eviction
            logger.info(f"Evicted WM: {evicted.id}")
        vad = user_vad or self.appraiser.analyze(content)['vad']
        importance = 1.0 if vad.arousal > 0.7 else 0.5  # Peak-end binary (locked)
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            vad=vad,
            timestamp=datetime.now(),
            recency_score=1.0,
            importance=importance
        )
        self.memories.insert(0, entry)  # Recent first
        self._decay_recency()
        logger.info(f"Added to WM: {entry.id} | VAD: {entry.vad}")
        return entry

    def _decay_recency(self):
        now = datetime.now()
        for mem in self.memories:
            age_hours = (now - mem.timestamp).total_seconds() / 3600
            mem.recency_score = max(0.1, 1.0 / (1 + age_hours))  # Decay func

    def get_all(self) -> List[MemoryEntry]:
        self._decay_recency()
        return self.memories

class EpisodicMemory:
    def __init__(self):
        self.memories: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry):
        self.memories.insert(0, entry)  # Chronological recent first
        logger.info(f"Added to EM: {entry.id}")

    def consolidate(self, entry: MemoryEntry) -> bool:
        # Prep for Phase 2: Trigger if arousal>0.7 or goal>0.8
        if entry.importance > 0.8:  # Stub; full in Day 14
            logger.info(f"Consolidation trigger: {entry.id}")
            return True
        return False

    def get_recent(self, n: int = 50) -> List[MemoryEntry]:
        return self.memories[:n]
