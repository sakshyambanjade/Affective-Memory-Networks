import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from src.memory.core import MemoryEntry, WorkingMemory, EpisodicMemory
from src.emotion.analyzer import FullEmotionalAppraisal
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
        self.appraiser = FullEmotionalAppraisal()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._fit_vectorizer()

    def _fit_vectorizer(self):
        dummy_texts = ["happy sad angry fear excited calm project work friend family"]
        self.vectorizer.fit(dummy_texts)

    def _semantic_score(self, query: str, entry: MemoryEntry) -> float:
        q_vec = self.vectorizer.transform([query])
        e_vec = self.vectorizer.transform([entry.content])
        return cosine_similarity(q_vec, e_vec)[0][0]

    def _emotional_resonance(self, query_vad, memory_vad) -> float:
        """
        Implements complementary retrieval for therapeutic reframing (paper Section 3.3)
        When user is negative → retrieve positive memories for reframing
        """
        # Valence: opposite when user is negative
        if query_vad.valence < -0.2:          # user in distress
            valence_sim = 1 - abs(query_vad.valence + memory_vad.valence)   # push toward positive
        else:
            valence_sim = 1 - abs(query_vad.valence - memory_vad.valence)
        # Arousal: always similarity (high arousal memories stay salient)
        arousal_sim = 1 - 0.5 * abs(query_vad.arousal - memory_vad.arousal)
        return 0.5 * valence_sim + 0.5 * arousal_sim

    def _goal_align(self, query_appraisal, entry_appraisal) -> float:
        goal_sim = 1 - abs(query_appraisal.goal_relevance - entry_appraisal.goal_relevance)
        agency_align = 1 - abs(query_appraisal.agency - entry_appraisal.agency)
        return (goal_sim + agency_align) / 2

    def retrieve(self, query: str, query_vad: 'VAD') -> List[Tuple[MemoryEntry, float]]:
        all_mems = self.wm.get_all() + self.em.get_recent(50)
        if not all_mems:
            return []

        # Compute full appraisal for query
        query_appraisal_dict = self.appraiser.full_appraisal(query)
        query_appraisal = query_appraisal_dict['lazarus']

        scores = []
        for mem in all_mems:
            sem = self._semantic_score(query, mem)
            # Use mem.appraisal.vad for emotional resonance
            emo = self._emotional_resonance(query_vad, mem.appraisal.vad)
            # Use goal_relevance as proxy for goal alignment
            goal = 1 - abs(query_appraisal.goal_relevance - mem.appraisal.goal_relevance)
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
        if top_k:
            logger.info(f"Retrieved top-1: {top_k[0][0].id} score={top_k[0][1]:.3f}")
        return top_k
