

import logging
from src.agent.baseline import BaselineAgent
from src.memory.core import WorkingMemory, EpisodicMemory
from src.retrieval.engine import RetrievalEngine
from src.agent.ollama_client import ollama_chat

logger = logging.getLogger('AMN')

class RecencyAgent:
    def __init__(self):
        self.baseline = BaselineAgent()
        self.wm = WorkingMemory()
        self.em = EpisodicMemory()
        self.retriever = RetrievalEngine(self.wm, self.em, k=3)
    
    def _format_context(self, retrieved):
        ctx = []
        for mem, score in retrieved:
            vad = getattr(mem, 'vad', None)
            if vad is None and hasattr(mem, 'appraisal') and hasattr(mem.appraisal, 'vad'):
                vad = mem.appraisal.vad
            vad_str = str(vad) if vad is not None else 'N/A'
            ctx.append(f"PAST: {mem.content[:150]}... [VAD:{vad_str}] (score:{score:.2f})")
        return "\n".join(ctx)

    def step(self, user_input: str) -> str:
        vad = self.baseline.appraiser.analyze(user_input)['vad']
        retrieved = self.retriever.retrieve(user_input, vad)
        retrieved = sorted(retrieved, key=lambda x: x[0].recency_score, reverse=True)
        context = self._format_context(retrieved)
        prompt = f"MEMORIES (RECENCY ONLY): {context}\nCURRENT: {user_input}\nRespond:"
        reply = ollama_chat(
            prompt,
            model="tinyllama",
            max_tokens=200,
            temperature=0.7
        )
        full_turn = f"User: {user_input}\nAgent: {reply}"
        entry = self.wm.add(full_turn)
        self.em.add(entry)
        return reply
