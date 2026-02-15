


import logging
from src.emotion.analyzer import EmotionalAppraisal
from src.agent.ollama_client import ollama_chat

logger = logging.getLogger('AMN')

class BaselineAgent:
    def __init__(self):
        self.appraiser = EmotionalAppraisal()

    def step(self, user_input: str) -> str:
        vad = self.appraiser.analyze(user_input)['vad']
        prompt = f"Respond empathetically to: {user_input}"
        reply = ollama_chat(
            prompt,
            model="tinyllama",
            max_tokens=200,
            temperature=0.7
        )
        return reply
