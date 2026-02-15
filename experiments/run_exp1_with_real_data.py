
import os
import sys
import json
import logging
from datetime import datetime


# Ensure project root is in sys.path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set up robust logging to a timestamped file in results/
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
log_time = datetime.now().strftime('%Y%m%d_%H%M')
log_path = os.path.join(RESULTS_DIR, f'exp1_run_{log_time}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('EXPERIMENT')

from amn_data_package.scripts.load_data import AMNDataLoader
from src.agent.agent import AMNAgent
from src.agent.baseline import BaselineAgent
from src.agent.recency import RecencyAgent
from src.agent.rag import SemanticRAGAgent


# Load up to 100 conversations from the data package
import time
from pathlib import Path
loader = AMNDataLoader(os.path.join(PROJECT_ROOT, 'amn_data_package'))
conversations = loader.prepare_for_experiment(n_conversations=100)

# Resume support: check for partial results
RESUME_FILE = os.path.join(RESULTS_DIR, 'exp1_resume.json')
if Path(RESUME_FILE).exists():
    with open(RESUME_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    start_idx = len(results)
    logger.info(f"Resuming from conversation {start_idx+1}")
else:
    results = []
    start_idx = 0

AGENTS = {
    'amn': AMNAgent(),
    'baseline': BaselineAgent(),
    'recency': RecencyAgent(),
    'semantic_rag': SemanticRAGAgent()
}



try:
    for i, convo in enumerate(conversations[start_idx:], start=start_idx):
        logger.info(f"Convo {i+1}/100: {convo['primary_emotion']}")
        convo_results = {}
        for condition, agent in AGENTS.items():
            logger.info(f"  Running {condition}...")
            turns = []
            for turn in convo['turns']:
                if turn['speaker'] == 'user':
                    user_input = turn['text']
                    try:
                        response = agent.step(user_input)
                        turns.append({
                            'user': user_input,
                            'agent': response,
                            'condition': condition
                        })
                        logger.info(f"    User: {user_input}")
                        logger.info(f"    Agent: {response}")
                    except Exception as e:
                        logger.error(f"    ERROR in agent '{condition}' on input '{user_input}': {e}", exc_info=True)
                        turns.append({
                            'user': user_input,
                            'agent': None,
                            'condition': condition,
                            'error': str(e)
                        })
                        # If API quota/rate limit error, save and exit for later resume
                        if 'quota' in str(e).lower() or 'rate limit' in str(e).lower() or 'exhaust' in str(e).lower():
                            logger.error("API quota/rate limit exhausted. Saving progress and exiting.")
                            with open(RESUME_FILE, 'w', encoding='utf-8') as f:
                                json.dump(results, f, indent=2)
                            sys.exit(99)
                    time.sleep(2)  # Delay between API calls
            convo_results[condition] = turns
        results.append({
            'convo_id': i+1,
            'primary_emotion': convo['primary_emotion'],
            **convo_results
        })
        # Save progress after each conversation
        with open(RESUME_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
except Exception as e:
    logger.error(f"FATAL ERROR during experiment: {e}", exc_info=True)
finally:
    # Always save results, even if partial
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = os.path.join(RESULTS_DIR, f'exp1_realdata_{timestamp}.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Experiment 1 with real data COMPLETE: {filename}")
    logger.info(f"Total conversations: {len(results)} × 4 conditions")
    # Remove resume file if finished
    if len(results) >= len(conversations) and Path(RESUME_FILE).exists():
        Path(RESUME_FILE).unlink()
    