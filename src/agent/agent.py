
import logging
from typing import List
from src.memory.core import WorkingMemory, EpisodicMemory
from src.retrieval.engine import RetrievalEngine
from src.emotion.analyzer import EmotionalAppraisal
from src.agent.ollama_client import ollama_chat

logger = logging.getLogger('AMN')

class AMNAgent:
    def __init__(self):
        self.wm = WorkingMemory()
        self.em = EpisodicMemory()
        self.retriever = RetrievalEngine(self.wm, self.em, k=3)
        self.appraiser = EmotionalAppraisal()

    def _format_context(self, retrieved: List) -> str:
        ctx = []
        for mem, score in retrieved:
            vad = getattr(mem, 'vad', None)
            if vad is None and hasattr(mem, 'appraisal') and hasattr(mem.appraisal, 'vad'):
                vad = mem.appraisal.vad
            vad_str = str(vad) if vad is not None else 'N/A'
            ctx.append(f"PAST: {mem.content[:150]}... [VAD:{vad_str}] (score:{score:.2f})")
        return "\n".join(ctx)

    def step(self, user_input: str) -> str:
        vad = self.appraiser.analyze(user_input)['vad']
        logger.info(f"User VAD: {vad}")
        retrieved = self.retriever.retrieve(user_input, vad)
        context = self._format_context(retrieved)
        prompt = f"You are an emotionally aware agent. Use these memories to respond empathetically:\n\nMEMORIES:\n{context}\n\nCURRENT: {user_input}\n\nRespond naturally, referencing relevant past emotions/experiences when helpful. Be concise."
        reply = ollama_chat(
            prompt,
            model="tinyllama",
            system_prompt="You maintain emotional continuity across conversations.",
            max_tokens=200,
            temperature=0.7
        )
        full_turn = f"User: {user_input}\nAgent: {reply}"
        entry = self.wm.add(full_turn)
        self.em.add(entry)
        logger.info(f"Response: {reply[:50]}...")
        return reply
