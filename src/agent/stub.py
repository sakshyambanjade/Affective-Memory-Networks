import sys
import os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)
from src.memory.core import WorkingMemory, EpisodicMemory
from src.retrieval.engine import RetrievalEngine
from src.emotion.analyzer import EmotionalAppraisal

class AMNAgent:
    def __init__(self):
        self.wm = WorkingMemory()
        self.em = EpisodicMemory()
        self.retriever = RetrievalEngine(self.wm, self.em)
        self.appraiser = EmotionalAppraisal()

    def step(self, user_input: str) -> str:
        vad = self.appraiser.analyze(user_input)['vad']
        context = self.retriever.retrieve(user_input, vad)
        ctx_str = "\n".join([f"{m.content} [VAD:{m.vad}] (score:{s:.2f})" for m, s in context])
        response = f"Retrieved: {ctx_str[:100]}... Considering emotions, I say: That's tough. Remember that happy project time?"
        full_turn = f"User: {user_input} | Agent: {response}"
        entry = self.wm.add(full_turn, vad)
        self.em.add(entry)
        return response
