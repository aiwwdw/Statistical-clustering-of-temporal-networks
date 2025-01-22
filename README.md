# Statistical Clustering of Temporal Networks Using Optimizer

This repository contains the implementation of the research presented in the paper **"Statistical Clustering of Temporal Networks Through a Dynamic Stochastic Block Model Using Optimizer."** The paper explores an efficient method for clustering nodes in temporal networks by employing optimization techniques inspired by deep learning.

---

## Overview

Statistical network analysis is a critical field in areas like biology, sociology, and internet studies. This project focuses on clustering temporal networks by leveraging:
- Dynamic Stochastic Block Models (DSBM) to model temporal dependencies.
- Optimizers (e.g., gradient descent-based approaches) to efficiently estimate maximum likelihood parameters for clustering.
- Tensor operations to reduce computational complexity and improve scalability.

The proposed approach demonstrates improvements in speed, memory efficiency, and accuracy over traditional fixed-point equation methods, enabling practical use in large-scale temporal networks.

---

## Key Features

1. **Dynamic Stochastic Block Model**:
   - Represents temporal networks as weighted, undirected graphs.
   - Models node transitions using Markov Chains with time-varying interactions.

2. **Optimization-Based Parameter Estimation**:
   - Replaces traditional Expectation Maximization (EM) methods with optimizers for estimating parameters like transition probabilities and cluster assignments.
   - Employs softmax and sigmoid functions to enforce constraints (e.g., probabilities between 0 and 1).

3. **Computational Efficiency**:
   - Utilizes tensor operations to drastically reduce time complexity.
   - Achieves faster convergence while maintaining accuracy.

4. **Synthetic Experiments**:
   - Evaluates performance using synthetic datasets under various conditions (e.g., group stability, time stamps).
   - Demonstrates superiority of optimizer-based methods in Adjusted Rand Index (ARI) metrics.

---

## File Structure

- **`main.py`**: Entry point for training and evaluating the model.
- **`dsbm_model.py`**: Implementation of the Dynamic Stochastic Block Model.
- **`optimizer_utils.py`**: Utility functions for gradient descent optimization.
- **`synthetic_data_generator.py`**: Generates synthetic temporal network datasets.
- **`evaluation.py`**: Calculates evaluation metrics like ARI for clustering results.

---

## How to Run
