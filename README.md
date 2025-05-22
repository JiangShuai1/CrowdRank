# CrowdRank: Extracting Collective Wisdom Based on Opinion Quality Rank Learning

This repository contains the implementation of a novel deep learning model for integrating heterogeneous expert opinions, as presented in our paper "Which to Trust: Extracting Collective Wisdom Based on Opinion Quality Rank Learning" submitted to Informs Journal on Computing.

## Overview

CrowdRank addresses the challenge of aggregating opinions from diverse experts with varying expertise levels. Our approach transforms the opinion integration problem into an opinion quality ranking problem:

1. First, we identify the relative quality between pairs of opinions
2. Then, we derive a global ranking from these pairwise comparisons using Expectation Propagation
3. Finally, we use this ranking to solve the opinion integration problem

We validate our model using financial market data, specifically earnings forecasts from financial analysts.

## Project Structure

The repository is organized into three main directories:

- **data/**: Contains experimental datasets
  - `Experiment-1/`, `Experiment-2/`, `Experiment-3/`: Three experimental setups with the same data structure
  - Each experiment folder contains `ReportData.csv` (original analyst opinion dataset) and JSON files for training Pairwise ranking models

- **code/**: Contains all experimental code implementations
  1. `1-TraditionalML/`: Traditional machine learning Pointwise ranking models
  2. `2-PointwiseNN/`: Deep learning-based Pointwise ranking models
  3. `3-PairwiseNN/`: Deep learning-based Pairwise ranking models
  4. `4-ListwiseNN/`: Listwise ranking models
  5. `5-BayesORN/`: Our proposed model (BayesianORN) with hyperparameter tuning
  6. `6-RankEvaluation/`: Evaluation of ranking performance for different models
  7. `7-EarningsPrediction/`: Validation of model's impact on earnings prediction accuracy
  8. `8-FeatureImportance/`: Quantifying feature importance in our model
  9. `9-EconomicValue/`: Analysis of economic benefits from accurate opinion ranking

- **output/**: Stores all experimental results
  - `table/`: Performance metrics and numerical results
  - `figure/`: Visualizations and plots

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Data Preparation

The datasets are already included in the `data/` directory. Each experiment has the following files:
- `ReportData.csv`: Financial analyst opinion dataset
- Train/Test/Val JSON files: Preprocessed data for model training and evaluation

## Usage

Each module in the `code/` directory contains scripts that can be run independently:

```bash
# Run traditional machine learning models
python code/1-TraditionalML/AOQBaselineResults.py

# Train and evaluate deep Pointwise models
python code/2-PointwiseNN/PointNNLearning.py

# Train and evaluate our proposed BayesianORN model
python code/5-BayesORN/BayesianORNLearning.py

# Evaluate ranking performance
python code/6-RankEvaluation/RankEvaluation.py
```

## Citation

If you find this code useful for your research, please cite our paper:

```
@article{crowdrank2023,
  title={Which to Trust: Extracting Collective Wisdom Based on Opinion Quality Rank Learning},
  author={Jiang, Shuai and Others},
  journal={Informs Journal on Computing},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 