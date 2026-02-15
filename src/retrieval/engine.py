import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from src.memory.core import MemoryEntry, WorkingMemory, EpisodicMemory
from src.emotion.analyzer import EmotionalAppraisal
import logging

logger = logging.getLogger('AMN')

class RetrievalEngine:
    WEIGHTS = {
        'semantic': 0.25,
        'emotional': 0.30,  # KEY: Complementary
        'goal': 0.20,       # Stub Phase 2
        'peak_end': 0.15,
        'recency': 0.10
    }  # Locked totals 1.0

    def __init__(self, wm: WorkingMemory, em: EpisodicMemory, k: int = 5):
        self.wm = wm
        self.em = em
        self.k = k
        self.appraiser = EmotionalAppraisal()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._fit_vectorizer()

    def _fit_vectorizer(self):
        dummy_texts = ["happy sad angry fear excited calm project work friend family"]
        self.vectorizer.fit(dummy_texts)

    def _semantic_score(self, query: str, entry: MemoryEntry) -> float:
        q_vec = self.vectorizer.transform([query])
        e_vec = self.vectorizer.transform([entry.content])
        return cosine_similarity(q_vec, e_vec)[0][0]

    def _emotional_resonance(self, query_vad, entry_vad) -> float:
        valence_diff = 1 - abs(query_vad.valence - entry_vad.valence)
        arousal_align = 1 - abs(query_vad.arousal - entry_vad.arousal) * 0.5
        return (valence_diff + arousal_align) / 2

    def _goal_align(self, query_appraisal, entry_appraisal) -> float:
        goal_sim = 1 - abs(query_appraisal.goal_relevance - entry_appraisal.goal_relevance)
        agency_align = 1 - abs(query_appraisal.agency - entry_appraisal.agency)
        return (goal_sim + agency_align) / 2

    def retrieve(self, query: str, query_vad: 'VAD') -> List[Tuple[MemoryEntry, float]]:
        all_mems = self.wm.get_all() + self.em.get_recent(50)
        if not all_mems:
            return []

        scores = []
        for mem in all_mems:
            sem = self._semantic_score(query, mem)
            emo = self._emotional_resonance(query_vad, mem.vad)
            goal = self._goal_align(query_appraisal, mem.appraisal)
            peak = mem.importance
            rec = mem.recency_score
            total = (
                self.WEIGHTS['semantic'] * sem +
                self.WEIGHTS['emotional'] * emo +
                self.WEIGHTS['goal'] * goal +
                self.WEIGHTS['peak_end'] * peak +
                self.WEIGHTS['recency'] * rec
            )
            scores.append((mem, total))

        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:self.k]
        logger.info(f"Retrieved top-1: {top_k[0][0].id} score={top_k[0][1]:.3f}")
        return top_k
