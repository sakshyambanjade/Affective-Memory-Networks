import os
from anthropic import Anthropic
from src.emotion.analyzer import EmotionalAppraisal
import logging

logger = logging.getLogger('AMN')

class BaselineAgent:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.appraiser = EmotionalAppraisal()

    def step(self, user_input: str) -> str:
        vad = self.appraiser.analyze(user_input)['vad']
        prompt = f"Respond empathetically to: {user_input}"
        resp = self.client.messages.create(
            model="claude-3.5-sonnet-20240620",
            max_tokens=200,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
