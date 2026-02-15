import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import yaml
import logging
from typing import Tuple, Dict, NamedTuple
import os
from anthropic import Anthropic

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

logger = logging.getLogger('AMN')

class VAD(NamedTuple):
    valence: float  # -1 (unpleasant) to 1 (pleasant)
    arousal: float  # 0 (calm) to 1 (excited)
    dominance: float  # 0 (submissive) to 1 (dominant)

class LazarusAppraisal(NamedTuple):
    vad: VAD
    goal_relevance: float
    agency: float
    certainty: float
    novelty: float
    pleasantness: float
    control: float

class EmotionalAppraisal:
    def __init__(self, config_path: str = None):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = Anthropic(api_key=api_key) if api_key else None
        self.stop_words = set(stopwords.words('english'))
        # Simplified lexicon (expand with Warriner/ANEW data later; download CSV for prod)
        self.lexicon = {
            # Valence high: joy=0.9, love=0.95; low: hate=-0.9, fear=-0.7
            # Add ~1400 words from https://github.com/bagustris/text-vad or Warriner
            'happy': (0.9, 0.6, 0.5), 'sad': (-0.8, 0.4, 0.3),
            'angry': (-0.6, 0.9, 0.8), 'fear': (-0.7, 0.8, 0.2),
            'excited': (0.7, 0.95, 0.7), 'calm': (0.4, 0.1, 0.6),
            # ... load full from file in Phase 2
        }
        if config_path is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
            config_path = os.path.join(root, 'config', 'defaults.yaml')
        self.load_config(config_path)

    def load_config(self, path: str):
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)
        os.makedirs('config', exist_ok=True)
        if not os.path.exists(path):
            default = {'llm_threshold': 0.3, 'lexicon_weight': 0.7}
            with open(path, 'w') as f:
                yaml.dump(default, f)

    def text_to_vad_lexicon(self, text: str) -> VAD:
        sentences = sent_tokenize(text.lower())
        v, a, d = [], [], []
        for sent in sentences:
            tokens = [w for w in word_tokenize(sent) if w.isalpha() and w not in self.stop_words]
            sent_v, sent_a, sent_d = 0.0, 0.0, 0.0
            count = 0
            for token in tokens:
                syns = wordnet.synsets(token)
                lemma = token if not syns else syns[0].lemmas()[0].name()
                if lemma in self.lexicon:
                    lv, la, ld = self.lexicon[lemma]
                    sent_v += lv; sent_a += la; sent_d += ld; count += 1
            if count > 0:
                v.append(sent_v / count); a.append(sent_a / count); d.append(sent_d / count)
        return VAD(np.mean(v) if v else 0.0, np.mean(a) if a else 0.0, np.mean(d) if d else 0.0)

    def llm_vad_fallback(self, text: str) -> VAD:
        if not self.client:
            logger.warning("LLM fallback skipped: ANTHROPIC_API_KEY not set")
            return VAD(0.0, 0.0, 0.0)
        prompt = f"""Analyze this text for VAD emotions (Valence -1 unpleasant to 1 pleasant, Arousal 0 calm to 1 excited, Dominance 0 submissive to 1 dominant). Respond ONLY with three floats separated by commas, e.g., \"0.5,0.7,0.4\": \"{text}\" """
        resp = self.client.messages.create(model="claude-3.5-sonnet-20240620", max_tokens=20, messages=[{"role": "user", "content": prompt}])
        try:
            scores = [float(x) for x in resp.content[0].text.strip().split(',')]
            return VAD(*scores)
        except:
            logger.warning("LLM VAD parse failed")
            return VAD(0.0, 0.0, 0.0)

    def analyze(self, text: str) -> Dict:
        lexicon_vad = self.text_to_vad_lexicon(text)
        llm_vad = self.llm_vad_fallback(text) if abs(lexicon_vad.valence) < self.config['llm_threshold'] else lexicon_vad
        # Weighted avg (Phase 2: add Lazarus appraisal)
        final_vad = VAD(
            self.config['lexicon_weight'] * lexicon_vad.valence + (1 - self.config['lexicon_weight']) * llm_vad.valence,
            lexicon_vad.arousal,  # Lexicon strong for arousal
            lexicon_vad.dominance
        )
        logger.info(f"Text: {text[:50]}... | VAD: ({final_vad.valence:.2f}, {final_vad.arousal:.2f}, {final_vad.dominance:.2f})")
        return {'vad': final_vad, 'lexicon': lexicon_vad, 'llm': llm_vad}

class FullEmotionalAppraisal(EmotionalAppraisal):
    def __init__(self):
        super().__init__()
        self.lazarus_prompt = """Analyze text using Lazarus appraisal theory. Return 7 floats: VAD(3), goal_relevance, agency, certainty, novelty, pleasantness, control. Range -1 to 1 or 0-1 appropriately. Example: \"0.5,0.7,0.4,0.8,0.6,0.3,0.9\": \"{text}\" """

    def full_appraisal(self, text: str) -> Dict:
        vad_dict = self.analyze(text)
        vad = vad_dict['vad']
        resp = self.client.messages.create(
            model="claude-3.5-sonnet-20240620",
            max_tokens=30,
            messages=[{"role": "user", "content": self.lazarus_prompt.format(text=text[:500])}]
        )
        try:
            scores = [float(x) for x in resp.content[0].text.strip().split(',')]
            lazarus = LazarusAppraisal(
                vad=vad,
                goal_relevance=scores[3],
                agency=scores[4],
                certainty=scores[5],
                novelty=scores[6],
                pleasantness=scores[7] if len(scores)>7 else 0.5,
                control=scores[8] if len(scores)>8 else 0.5
            )
        except:
            lazarus = LazarusAppraisal(
                vad=vad,
                goal_relevance=max(0, vad.valence),
                agency=vad.dominance,
                certainty=1-vad.arousal,
                novelty=vad.arousal,
                pleasantness=(vad.valence+1)/2,
                control=vad.dominance
            )
        consolidate = (lazarus.vad.arousal > 0.7) or (lazarus.goal_relevance > 0.8)
        logger.info(f"Appraisal: VAD={lazarus.vad}, Goal={lazarus.goal_relevance:.2f}, Consolidate={consolidate}")
        return {
            'lazarus': lazarus,
            'consolidate': consolidate,
            'vad': vad_dict
        }

# Test script in analyzer.py if __name__
if __name__ == "__main__":
    appraiser = EmotionalAppraisal()
    tests = [
        "I'm so excited about this new project!",
        "I feel sad and lost after the breakup.",
        "This makes me angry and frustrated.",
        "I'm calm and relaxed right now.",
        "I fear failing this important task."
    ]
    for text in tests:
        result = appraiser.analyze(text)
        print(result)
