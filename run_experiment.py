#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CrowdRank - Main experiment runner script
This script provides a centralized way to run all experiments in sequence.
"""

import os
import argparse
import time
import sys

def run_cmd(cmd):
    """Run a command and print its output"""
    print(f"Running: {cmd}")
    start_time = time.time()
    ret = os.system(cmd)
    duration = time.time() - start_time
    print(f"Command completed in {duration:.2f} seconds with return code {ret}")
    return ret

def main():
    parser = argparse.ArgumentParser(description='CrowdRank - Run experiments')
    parser.add_argument('--experiment', type=int, default=1, choices=[1, 2, 3],
                        help='Experiment ID (1, 2, or 3)')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['traditional', 'pointwise', 'pairwise', 'listwise', 'bayes',
                                'evaluation', 'earnings', 'feature', 'economic', 'all'],
                        help='Stage to run')
    args = parser.parse_args()
    
    experiment_id = args.experiment
    print(f"Running experiment {experiment_id}")
    
    # Define command paths
    cmd_traditional = f"python code/1-TraditionalML/AOQBaselineResults.py"
    cmd_pointwise = f"python code/2-PointwiseNN/PointNNLearning.py --experiment {experiment_id}"
    cmd_pairwise = f"python code/3-PairwiseNN/RankNNLearning.py --experiment {experiment_id}"
    cmd_listwise = f"python code/4-ListwiseNN/DeepListwiseLearning.py --experiment {experiment_id}"
    cmd_bayes = f"python code/5-BayesORN/BayesianORNLearning.py --experiment {experiment_id}"
    cmd_evaluation = f"python code/6-RankEvaluation/RankEvaluation.py --experiment {experiment_id}"
    cmd_earnings = f"python code/7-EarningsPrediction/EPSPred.py --experiment {experiment_id}"
    cmd_feature = f"python code/8-FeatureImportance/FeatureImportance.py --experiment {experiment_id}"
    cmd_economic = f"python code/9-EconomicValue/EconomicValue.py --experiment {experiment_id}"
    
    # Create output directories if they don't exist
    os.makedirs(f"output/table/Experiment-{experiment_id}", exist_ok=True)
    os.makedirs(f"output/figure/Experiment-{experiment_id}", exist_ok=True)
    
    # Run commands based on specified stage
    if args.stage == 'traditional' or args.stage == 'all':
        run_cmd(cmd_traditional)
    
    if args.stage == 'pointwise' or args.stage == 'all':
        run_cmd(cmd_pointwise)
    
    if args.stage == 'pairwise' or args.stage == 'all':
        run_cmd(cmd_pairwise)
    
    if args.stage == 'listwise' or args.stage == 'all':
        run_cmd(cmd_listwise)
    
    if args.stage == 'bayes' or args.stage == 'all':
        run_cmd(cmd_bayes)
    
    if args.stage == 'evaluation' or args.stage == 'all':
        run_cmd(cmd_evaluation)
    
    if args.stage == 'earnings' or args.stage == 'all':
        run_cmd(cmd_earnings)
    
    if args.stage == 'feature' or args.stage == 'all':
        run_cmd(cmd_feature)
    
    if args.stage == 'economic' or args.stage == 'all':
        run_cmd(cmd_economic)
    
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main() 