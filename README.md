# Stability and Reproducibility of Sparse Machine Learning Models in Small-Sample Metabolomics

## Overview

This project evaluates the reliability of sparse machine learning models applied to small metabolomics datasets.

Rather than focusing on discovering new biological biomarkers, the goal is to assess whether machine learning–derived feature selections and performance estimates are stable and reproducible under small-sample conditions.

## Objective

To determine:

* Whether sparse models (LASSO, Elastic Net) consistently select the same metabolites across repeated resampling.
* How much model performance varies under strict nested cross-validation.
* The extent of overfitting caused by naive validation strategies.
* How reliable ML-derived biomarker claims are in small-n metabolomics studies.

## Methodology

* Binary classification using metabolite concentration data.
* Strict nested cross-validation for unbiased performance estimation.
* Repeated resampling to measure variability.
* Quantification of feature selection stability.
* Comparison of naive vs nested validation performance.
* Evaluation of calibration and prediction uncertainty.

## Visual diagram of training process
Nested Cross-Validation Structure

Full Dataset
   │
   ▼
Outer CV (Model Evaluation)

For each outer fold:

1. Hold out one fold as the test set.
2. Use the remaining folds as the training set.
3. Run inner cross-validation on the training set to tune hyperparameters.
4. Train the model with the best hyperparameters.
5. Evaluate on the held-out outer test fold.

Example (5 folds):

Fold 1 → Test: [1] | Train: [2,3,4,5] → Inner CV
Fold 2 → Test: [2] | Train: [1,3,4,5] → Inner CV
Fold 3 → Test: [3] | Train: [1,2,4,5] → Inner CV
Fold 4 → Test: [4] | Train: [1,2,3,5] → Inner CV
Fold 5 → Test: [5] | Train: [1,2,3,4] → Inner CV

## Key Focus

* Stability of selected metabolites
* Variability of model performance
* Overfitting risk
* Reproducibility of ML conclusions

## Outcome

The project provides evidence-based recommendations for applying sparse machine learning models to small metabolomics datasets, with emphasis on rigor, reliability, and methodological best practice.
