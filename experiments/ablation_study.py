"""
Ablation Study for AMN
Tests importance of each retrieval factor by removing one at a time
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from amn_data_package.scripts.load_data import AMNDataLoader
from src.agent.agent import AMNAgent
from src.memory.core import WorkingMemory, EpisodicMemory
from src.retrieval.engine import RetrievalEngine
from src.emotion.analyzer import EmotionalAppraisal

# Set up logging
RESULTS_DIR = PROJECT_ROOT / 'results' / 'ablation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
log_time = datetime.now().strftime('%Y%m%d_%H%M')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / f'ablation_{log_time}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ABLATION')

# Define ablation configurations
ABLATION_CONFIGS = {
    'full': {
        'semantic': 0.25,
        'emotional': 0.30,
        'goal': 0.20,
        'peak_end': 0.15,
        'recency': 0.10
    },
    'no_emotional': {
        'semantic': 0.286,
        'emotional': 0.0,
        'goal': 0.286,
        'peak_end': 0.214,
        'recency': 0.214
    },
    'no_goal': {
        'semantic': 0.3125,
        'emotional': 0.375,
        'goal': 0.0,
        'peak_end': 0.1875,
        'recency': 0.125
    },
    'no_peak': {
        'semantic': 0.294,
        'emotional': 0.353,
        'goal': 0.235,
        'peak_end': 0.0,
        'recency': 0.118
    },
    'no_recency': {
        'semantic': 0.278,
        'emotional': 0.333,
        'goal': 0.222,
        'peak_end': 0.167,
        'recency': 0.0
    }
}

class AblationAMNAgent(AMNAgent):
    """AMN agent with configurable retrieval weights"""
    def __init__(self, weights_config):
        # Initialize components
        self.wm = WorkingMemory()
        self.em = EpisodicMemory()
        self.retriever = RetrievalEngine(self.wm, self.em, k=3)
        self.appraiser = EmotionalAppraisal()
        
        # Override weights
        self.retriever.WEIGHTS = weights_config
        logger.info(f"Initialized with weights: {weights_config}")

def run_ablation_variant(conversations, config_name, weights):
    """Run one ablation configuration on test conversations"""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING: {config_name}")
    logger.info(f"Weights: {weights}")
    logger.info(f"{'='*60}\n")
    
    agent = AblationAMNAgent(weights)
    results = []
    
    for i, convo in enumerate(conversations):
        logger.info(f"[{i+1}/{len(conversations)}] {convo['primary_emotion']}")
        turns = []
        
        for turn in convo.get('turns', []):
            if turn.get('speaker') == 'user':
                user_text = turn.get('text', '')
                try:
                    response = agent.step(user_text)
                    turns.append({
                        'user': user_text,
                        'agent': response
                    })
                    logger.info(f"  U: {user_text[:60]}...")
                    logger.info(f"  A: {response[:60]}...")
                except Exception as e:
                    logger.error(f"  ERROR: {e}")
                    turns.append({
                        'user': user_text,
                        'agent': None,
                        'error': str(e)
                    })
                
                time.sleep(1)  # Rate limiting
        
        results.append({
            'convo_id': convo.get('id', i),
            'primary_emotion': convo.get('primary_emotion', 'unknown'),
            'turns': turns
        })
    
    return results

def main():
    logger.info("="*60)
    logger.info("AMN ABLATION STUDY")
    logger.info("="*60)
    
    # Load conversations
    loader = AMNDataLoader(PROJECT_ROOT / 'amn_data_package')
    all_convos = loader.prepare_for_experiment(n_conversations=100)
    
    # Select 15 balanced conversations (1 per emotion)
    emotions_seen = set()
    test_convos = []
    for convo in all_convos:
        emotion = convo.get('primary_emotion', 'unknown')
        if emotion not in emotions_seen and emotion != 'unknown':
            test_convos.append(convo)
            emotions_seen.add(emotion)
            if len(test_convos) == 15:
                break
    
    logger.info(f"\n✓ Selected {len(test_convos)} test conversations")
    logger.info(f"✓ Emotions: {sorted(emotions_seen)}\n")
    
    # Run each ablation variant
    all_results = {}
    for config_name, weights in ABLATION_CONFIGS.items():
        variant_results = run_ablation_variant(test_convos, config_name, weights)
        all_results[config_name] = variant_results
        
        # Save progress
        progress_file = RESULTS_DIR / f'ablation_progress_{log_time}.json'
        with open(progress_file, 'w') as f:
            json.dump({
                'completed': list(all_results.keys()),
                'results': all_results
            }, f, indent=2)
    
    # Save final results
    output_file = RESULTS_DIR / f'ablation_results_{log_time}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'configs': ABLATION_CONFIGS,
            'results': all_results,
            'test_conversations': [
                {'id': c.get('id'), 'emotion': c.get('primary_emotion')} 
                for c in test_convos
            ]
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ ABLATION STUDY COMPLETE")
    logger.info(f"✅ Results: {output_file}")
    logger.info(f"{'='*60}")
    logger.info(f"\nNext step: python experiments/analyze_ablation.py")

if __name__ == '__main__':
    main()
