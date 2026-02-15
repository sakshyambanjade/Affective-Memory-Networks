import os
from anthropic import Anthropic
from typing import List
from src.memory.core import WorkingMemory, EpisodicMemory
from src.retrieval.engine import RetrievalEngine
from src.emotion.analyzer import EmotionalAppraisal
import logging

logger = logging.getLogger('AMN')

class AMNAgent:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.wm = WorkingMemory()
        self.em = EpisodicMemory()
        self.retriever = RetrievalEngine(self.wm, self.em, k=3)
        self.appraiser = EmotionalAppraisal()

    def _format_context(self, retrieved: List) -> str:
        ctx = []
        for mem, score in retrieved:
            ctx.append(f"PAST: {mem.content[:150]}... [VAD:{mem.vad}] (score:{score:.2f})")
        return "\n".join(ctx)

    def step(self, user_input: str) -> str:
        vad = self.appraiser.analyze(user_input)['vad']
        logger.info(f"User VAD: {vad}")
        retrieved = self.retriever.retrieve(user_input, vad)
        context = self._format_context(retrieved)
        prompt = f"""You are an emotionally aware agent. Use these memories to respond empathetically:

MEMORIES:
{context}

CURRENT: {user_input}

Respond naturally, referencing relevant past emotions/experiences when helpful. Be concise."""
        resp = self.client.messages.create(
            model="claude-3.5-sonnet-20240620",
            max_tokens=200,
            temperature=0.7,
            system="You maintain emotional continuity across conversations.",
            messages=[{"role": "user", "content": prompt}]
        )
        response = resp.content[0].text.strip()
        full_turn = f"User: {user_input}\nAgent: {response}"
        entry = self.wm.add(full_turn, vad)
        self.em.add(entry)
        logger.info(f"Response: {response[:50]}...")
        return response
