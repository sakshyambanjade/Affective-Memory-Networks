
# --- AMN Data Package Integration ---

import os
import sys
import logging
from pathlib import Path
import numpy as np
import yaml
from typing import Dict, NamedTuple
import sys
import os
from pathlib import Path
# Ensure amn_data_package/scripts is in sys.path for import
DATA_PACKAGE_PATH = os.getenv(
    'AMN_DATA_PACKAGE_PATH',
    str(Path(__file__).parent.parent.parent / 'amn_data_package')
)
SCRIPTS_PATH = str(Path(DATA_PACKAGE_PATH) / 'scripts')
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)
from load_data import AMNDataLoader

# Add data package scripts to path for import
DATA_PACKAGE_PATH = os.getenv(
    'AMN_DATA_PACKAGE_PATH',
    str(Path(__file__).parent.parent.parent / 'amn_data_package')
)
SCRIPTS_PATH = str(Path(DATA_PACKAGE_PATH) / 'scripts')
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

logger = logging.getLogger('AMN')



class VAD(NamedTuple):
    valence: float  # -1 (unpleasant) to 1 (pleasant)
    arousal: float  # 0 (calm) to 1 (excited)
    dominance: float  # 0 (submissive) to 1 (dominant)



class LazarusAppraisal(NamedTuple):
    vad: 'VAD'
    goal_relevance: float
    agency: float
    certainty: float
    novelty: float
    pleasantness: float
    control: float



class EmotionalAppraisal:
    def __init__(self, config_path: str = None,
                 data_package_dir: str = DATA_PACKAGE_PATH):
        """
        EmotionalAppraisal loads a comprehensive VAD lexicon from the AMN data
        package.
        Args:
            config_path: Path to YAML config (optional)
            data_package_dir: Path to amn_data_package (default: autodetect or env var)
        """
        self.data_loader = AMNDataLoader(data_package_dir)
        self.lexicon = self.data_loader.load_vad_lexicon()
        if config_path is None:
            root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            config_path = os.path.join(root, 'config', 'defaults.yaml')
        self.load_config(config_path)



    def load_config(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        os.makedirs('config', exist_ok=True)
        if not os.path.exists(path):
            default = {'lexicon_weight': 1.0}
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(default, f)


    def text_to_vad_lexicon(self, text: str) -> VAD:
        """
        Compute VAD for input text using the loaded 209-word lexicon.
        Tokenizes, matches words, and averages VAD values.
        """
        import re
        tokens = re.findall(r"\b\w+\b", text.lower())
        v, a, d = [], [], []
        for token in tokens:
            if token in self.lexicon:
                lv, la, ld = self.lexicon[token]
                v.append(lv)
                a.append(la)
                d.append(ld)
        return VAD(
            np.mean(v) if v else 0.0,
            np.mean(a) if a else 0.0,
            np.mean(d) if d else 0.0
        )


    def analyze(self, text: str) -> Dict:
        """
        Analyze text and return VAD using the data package lexicon only.
        """
        lexicon_vad = self.text_to_vad_lexicon(text)
        final_vad = VAD(
            self.config.get('lexicon_weight', 1.0) * lexicon_vad.valence,
            lexicon_vad.arousal,
            lexicon_vad.dominance
        )
        logger.info(
            "Text: %s... | VAD: (%.2f, %.2f, %.2f)",
            text[:50], final_vad.valence, final_vad.arousal, final_vad.dominance
        )
        return {'vad': final_vad, 'lexicon': lexicon_vad}




class FullEmotionalAppraisal(EmotionalAppraisal):
    def __init__(self, data_package_dir: str = DATA_PACKAGE_PATH):
        super().__init__(data_package_dir=data_package_dir)

    def full_appraisal(self, text: str) -> Dict:
        vad_dict = self.analyze(text)
        vad = vad_dict['vad']
        # Use local rules for LazarusAppraisal
        lazarus = LazarusAppraisal(
            vad=vad,
            goal_relevance=max(0, vad.valence),
            agency=vad.dominance,
            certainty=1 - vad.arousal,
            novelty=vad.arousal,
            pleasantness=(vad.valence + 1) / 2,
            control=vad.dominance
        )
        return {
            'lazarus': lazarus,
            'vad': vad,
            'consolidate': vad.arousal > 0.7 or max(0, vad.valence) > 0.8
        }
