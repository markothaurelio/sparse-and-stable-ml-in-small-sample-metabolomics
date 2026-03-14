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

Nested Cross-Validation (5 Outer Folds)

FULL DATASET
│
└── OUTER CROSS-VALIDATION (Model Evaluation)

    ├── Fold 1
    │   ├── Test: Fold 1
    │   └── Train: Folds 2,3,4,5
    │       └── INNER CV (Hyperparameter Tuning)

    ├── Fold 2
    │   ├── Test: Fold 2
    │   └── Train: Folds 1,3,4,5
    │       └── INNER CV

    ├── Fold 3
    │   ├── Test: Fold 3
    │   └── Train: Folds 1,2,4,5
    │       └── INNER CV

    ├── Fold 4
    │   ├── Test: Fold 4
    │   └── Train: Folds 1,2,3,5
    │       └── INNER CV

    └── Fold 5
        ├── Test: Fold 5
        └── Train: Folds 1,2,3,4
            └── INNER CV

## Key Focus

* Stability of selected metabolites
* Variability of model performance
* Overfitting risk
* Reproducibility of ML conclusions

## Outcome

The project provides evidence-based recommendations for applying sparse machine learning models to small metabolomics datasets, with emphasis on rigor, reliability, and methodological best practice.
