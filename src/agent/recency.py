from src.agent.baseline import BaselineAgent
from src.memory.core import WorkingMemory, EpisodicMemory
from src.retrieval.engine import RetrievalEngine
import logging

logger = logging.getLogger('AMN')

class RecencyAgent:
    def __init__(self):
        self.baseline = BaselineAgent()
        self.wm = WorkingMemory()
        self.em = EpisodicMemory()
        self.retriever = RetrievalEngine(self.wm, self.em, k=3)
    
    def step(self, user_input: str) -> str:
        vad = self.baseline.appraiser.analyze(user_input)['vad']
        retrieved = self.retriever.retrieve(user_input, vad)
        retrieved = sorted(retrieved, key=lambda x: x[0].recency_score, reverse=True)
        context = self.retriever._format_context(retrieved)
        prompt = f"""MEMORIES (RECENCY ONLY): {context}\nCURRENT: {user_input}\nRespond:"""
        resp = self.baseline.client.messages.create(
            model="claude-3.5-sonnet-20240620", max_tokens=200, 
            messages=[{"role": "user", "content": prompt}]
        )
        response = resp.content[0].text.strip()
        full_turn = f"User: {user_input}\nAgent: {response}"
        entry = self.wm.add(full_turn, vad)
        self.em.add(entry)
        return response
