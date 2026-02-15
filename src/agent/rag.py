

import numpy as np
from src.agent.baseline import BaselineAgent
from src.memory.core import WorkingMemory, EpisodicMemory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from src.agent.ollama_client import ollama_chat

logger = logging.getLogger('AMN')

class SemanticRAGAgent:
    def __init__(self):
        self.baseline = BaselineAgent()
        self.wm = WorkingMemory()
        self.em = EpisodicMemory()
    
    def step(self, user_input: str) -> str:
        vad = self.baseline.appraiser.analyze(user_input)['vad']
        all_mems = self.wm.get_all() + self.em.get_recent(50)
        if not all_mems:
            # No memories yet, just respond to the user input
            prompt = f"SEMANTIC MEMORIES: (none)\nCURRENT: {user_input}\nRespond:"
            reply = ollama_chat(
                prompt,
                model="llama2",
                max_tokens=200,
                temperature=0.7
            )
            full_turn = f"User: {user_input}\nAgent: {reply}"
            entry = self.wm.add(full_turn)
            self.em.add(entry)
            return reply
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        vectorizer.fit([user_input] + [m.content for m in all_mems])
        q_vec = vectorizer.transform([user_input])
        mem_vecs = vectorizer.transform([m.content for m in all_mems])
        similarities = cosine_similarity(q_vec, mem_vecs)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        context = "\n".join([f"PAST: {all_mems[i].content[:150]}..." for i in top_indices])
        prompt = f"SEMANTIC MEMORIES: {context}\nCURRENT: {user_input}\nRespond:"
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
