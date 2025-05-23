# CrowdRank: Extracting Collective Wisdom Based on Opinion Quality Rank Learning

This repository contains the implementation of a novel deep learning model for integrating heterogeneous expert opinions, as presented in our paper "Which to Trust: Extracting Collective Wisdom Based on Opinion Quality Rank Learning" submitted to Informs Journal on Computing.

## Overview

CrowdRank addresses the challenge of aggregating opinions from diverse experts with varying expertise levels. Our approach transforms the opinion integration problem into an opinion quality ranking problem:

1. First, we identify the relative quality between pairs of opinions  
2. Then, we derive a global ranking from these pairwise comparisons using Expectation Propagation  
3. Finally, we use this ranking to solve the opinion integration problem  

We validate our model using financial market data, specifically earnings forecasts from financial analysts.

## Key Features

- **Opinion Quality Ranking**: Transform traditional opinion integration into a ranking problem for more accurate aggregation.
- **Deep Pairwise Comparison Models**: Utilize deep learning techniques to learn the relative quality of expert opinions.
- **Bayesian Optimization**: Integrate Bayesian methods for hyperparameter tuning and uncertainty quantification.
- **Comprehensive Evaluation**: Evaluate models through multiple criteria including accuracy, economic value, and feature importance.
- **Modular Code Structure**: Well-organized codebase allowing independent execution of each module for flexibility and reusability.

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

Due to file size limitations, the full datasets are not included in this repository. You can download the experimental data via Baidu Netdisk:

🔗 [Download data via Baidu Netdisk](https://pan.baidu.com/s/121Y1g_jVhrGOzDQAOLi-4g)  
🔑 Extraction Code: `jcmb`

> ⚠️ Note: The link may be updated periodically to prevent data abuse. To obtain the latest download link, please send an email to `shuaijiangai@gmail.com`.

After downloading, place the extracted files under the corresponding experiment folders (`data/Experiment-1/`, `data/Experiment-2/`, etc.).

## Usage

Each module in the `code/` directory contains scripts that can be run independently. Below are usage examples for each core component:

### 1. Run Traditional Machine Learning Models
```bash
python code/1-TraditionalML/AOQBaselineResults.py
```

### 2. Train and Evaluate Deep Pointwise Models
```bash
python code/2-PointwiseNN/PointNNLearning.py
```

### 3. Train and Evaluate Deep Pairwise Models
```bash
python code/3-PairwiseNN/PairNNLearning.py
```

### 4. Train and Evaluate Listwise Ranking Models
```bash
python code/4-ListwiseNN/ListNNLearning.py
```

### 5. Train and Tune the Proposed BayesianORN Model
```bash
python code/5-BayesORN/BayesianORNLearning.py
```

### 6. Evaluate Ranking Performance Across Models
```bash
python code/6-RankEvaluation/RankEvaluation.py
```

### 7. Validate Earnings Prediction Accuracy Using Aggregated Opinions
```bash
python code/7-EarningsPrediction/EarningsPrediction.py
```

### 8. Analyze Feature Importance in Opinion Quality Ranking
```bash
python code/8-FeatureImportance/FeatureImportanceAnalysis.py
```

### 9. Assess Economic Value of Accurate Opinion Ranking
```bash
python code/9-EconomicValue/EconomicValueAssessment.py
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
