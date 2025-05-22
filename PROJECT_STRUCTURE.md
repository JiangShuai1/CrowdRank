# Project Structure

This document provides an overview of the CrowdRank project structure.

```
CrowdRank/
│
├── code/                           # Source code for all experiments
│   ├── 1-TraditionalML/            # Traditional machine learning Pointwise ranking models
│   │   ├── AOQBaselineModels.py    # Implementation of traditional ML models
│   │   └── AOQBaselineResults.py   # Training and evaluation script
│   │
│   ├── 2-PointwiseNN/              # Deep learning-based Pointwise ranking models
│   │   ├── BaseFunction.py         # Utility functions for data loading and processing
│   │   ├── Early_Stopping.py       # Early stopping implementation for model training
│   │   ├── PointNNLearning.py      # Training and evaluation script
│   │   └── PointNNModels.py        # Model architectures (EWDNN, ResNet, FM, etc.)
│   │
│   ├── 3-PairwiseNN/               # Deep learning-based Pairwise ranking models
│   │   ├── BaseFunction.py         # Utility functions for data loading and processing
│   │   ├── Early_Stopping.py       # Early stopping implementation for model training
│   │   ├── RankNNLearning.py       # Training and evaluation script
│   │   └── RankNNModels.py         # Model architectures for pairwise ranking
│   │
│   ├── 4-ListwiseNN/               # Listwise ranking models
│   │   ├── BaseFunction.py         # Utility functions for data loading and processing
│   │   ├── DeepListwiseFunc.py     # Utility functions specific to listwise models
│   │   ├── DeepListwiseLearning.py # Training and evaluation script
│   │   ├── Early_Stopping.py       # Early stopping implementation for model training
│   │   ├── LGBMRanker.py           # LGBM implementation for listwise ranking
│   │   ├── LTRBenchFunc.py         # Learning-to-rank benchmark functions
│   │   └── LTRData.py              # Data handling for learning-to-rank
│   │
│   ├── 5-BayesORN/                 # Our proposed BayesianORN model
│   │   ├── BaseFunction.py         # Utility functions for data loading and processing
│   │   ├── BayesianORNLearning.py  # Training script with hyperparameter tuning
│   │   ├── BayesianORNModels.py    # BayesianORN model architecture
│   │   ├── Early_Stopping.py       # Early stopping implementation for model training
│   │   └── ORNPrediction.py        # Prediction script for the trained model
│   │
│   ├── 6-RankEvaluation/           # Evaluation of model ranking performance
│   │   ├── AddedRankEvaluation.py  # Additional evaluation metrics
│   │   ├── AddedRankEvaluationFunc.py # Functions for additional evaluation metrics
│   │   ├── EvaluationFunc.py       # Core evaluation functions
│   │   └── RankEvaluation.py       # Main evaluation script
│   │
│   ├── 7-EarningsPrediction/       # Validation through earnings prediction
│   │   ├── AccountingEPSPred.py    # Accounting-based EPS prediction
│   │   ├── AddedEPSPred.py         # Additional EPS prediction methods
│   │   ├── AddedRankEvaluationFunc.py # Functions for additional evaluation
│   │   ├── EPSPred.py              # Main EPS prediction script
│   │   └── EPSPredFunc.py          # Utility functions for EPS prediction
│   │
│   ├── 8-FeatureImportance/        # Feature importance analysis
│   │   ├── BaseFunction.py         # Utility functions
│   │   ├── BayesianORNModels.py    # BayesianORN model for feature analysis
│   │   └── FeatureImportance.py    # Feature importance quantification script
│   │
│   └── 9-EconomicValue/            # Economic value analysis
│       ├── EconomicValue.py        # Economic value assessment script
│       └── EPSPredFunc.py          # Utility functions for EPS prediction
│
├── data/                           # Experimental datasets
│   ├── Experiment-1/               # First experimental setup
│   │   ├── ReportData.csv          # Original analyst opinion dataset
│   │   ├── Train.json              # Training data for pairwise models
│   │   ├── Val.json                # Validation data
│   │   └── Test-{1-5}.json         # Test datasets
│   │
│   ├── Experiment-2/               # Second experimental setup
│   │   ├── ReportData.csv
│   │   ├── Train.json
│   │   ├── Val.json
│   │   └── Test-{1-5}.json
│   │
│   └── Experiment-3/               # Third experimental setup
│       ├── ReportData.csv
│       ├── Train.json
│       ├── Val.json
│       └── Test-{1-5}.json
│
├── output/                         # Experimental results
│   ├── table/                      # Tables and numerical results
│   │   ├── Experiment-1/
│   │   ├── Experiment-2/
│   │   └── Experiment-3/
│   │
│   └── figure/                     # Visualizations and plots
│       ├── Experiment-1/
│       ├── Experiment-2/
│       └── Experiment-3/
│
├── run_experiment.py               # Main script to run experiments
├── requirements.txt                # Python dependencies
├── README.md                       # Project description and usage instructions
├── LICENSE                         # MIT License
└── .gitignore                      # Git ignore file
```

## Directory Structure Details

### code/
Contains all implementation code, organized by model type and evaluation purpose.

### data/
Contains the experimental datasets for three experimental setups. Each experiment includes analyst opinion data (`ReportData.csv`) and preprocessed data for training and evaluation (`Train.json`, `Val.json`, `Test-*.json`).

### output/
Stores all experimental results, including performance metrics in tables and visualizations in figures.

## Key Files

- `run_experiment.py`: Main script for running experiments with command-line arguments
- `requirements.txt`: Lists all Python package dependencies
- `README.md`: Project documentation and usage instructions 