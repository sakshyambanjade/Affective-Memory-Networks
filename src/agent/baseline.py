


import logging
from src.emotion.analyzer import EmotionalAppraisal
from src.agent.gpt_oss_client import gpt_oss_cloud_chat

logger = logging.getLogger('AMN')

class BaselineAgent:
    def __init__(self, model="gpt-oss:120b-cloud"):
        self.appraiser = EmotionalAppraisal()
        self.model = model

    def step(self, user_input: str) -> str:
        vad = self.appraiser.analyze(user_input)['vad']
        prompt = f"Respond empathetically to: {user_input}"
        reply = gpt_oss_cloud_chat(
            prompt,
            model=self.model,
            max_tokens=200,
            temperature=0.7
        )
        return reply
