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
from src.emotion.analyzer import FullEmotionalAppraisal, LazarusAppraisal

logger = logging.getLogger('AMN')

@dataclass
class MemoryEntry:
    id: str
    content: str
    appraisal: LazarusAppraisal
    timestamp: datetime
    recency_score: float
    importance: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class WorkingMemory:
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.memories: List[MemoryEntry] = []
        self.appraiser = FullEmotionalAppraisal()

    def add(self, content: str) -> MemoryEntry:
        appraisal_dict = self.appraiser.full_appraisal(content)
        appraisal = appraisal_dict['lazarus']
        if len(self.memories) >= self.capacity:
            evicted = self.memories.pop()
            logger.info(f"Evicted: {evicted.id}")
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            appraisal=appraisal,
            timestamp=datetime.now(),
            recency_score=1.0,
            importance=1.0 if appraisal_dict['consolidate'] else 0.5
        )
        self.memories.insert(0, entry)
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
