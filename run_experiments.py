#!/usr/bin/env python3
"""
AMN Experiment 1: One-command reproducibility
Usage: python run_experiments.py --output results/exp1_repro.json
"""
import argparse
from experiments.exp1_generator import run_experiment1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/exp1_repro.json')
    args = parser.parse_args()
    run_experiment1()  # Uses fresh agents
    print(f"Repro complete: {args.output}")
