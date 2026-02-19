#!/usr/bin/env python3
"""
AMN Experiment 1: One-command reproducibility
Usage: python run_experiments.py --output results/exp1_repro.json
"""

import argparse
from experiments.exp1_generator import run_experiment1
from experiments.eval_metrics import compute_all_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/exp1_repro.json')
    parser.add_argument('--full', action='store_true', help='Run full experiment (default: True)')
    parser.add_argument('--model', type=str, default='tinyllama:latest', help='Model name for agents')
    parser.add_argument('--n_convos', type=int, default=100, help='Number of conversations to run')
    args = parser.parse_args()
    run_experiment1(full=args.full, model=args.model, n_convos=args.n_convos, output=args.output)
    print(f"Repro complete: {args.output}")
