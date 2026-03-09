# Entropy-SGD optimizes the prior of a PAC-Bayes bound

**Bayesian Machine Learning — Project**  
*Alexandre Mallez · Swann Cordier*

---

## Overview

This notebook is a PyTorch implementation of **Entropy-SGLD**, the algorithm introduced in:

> Dziugaite & Roy, *"Entropy-SGD optimizes the prior of a PAC-Bayes bound: Generalization properties of Entropy-SGD and data-dependent priors"*, ICML 2018. ([arXiv:1712.09376](https://arxiv.org/abs/1712.09376))

The core idea is that Entropy-SGD implicitly optimizes a data-dependent prior within a PAC-Bayes framework, leading to tighter generalization bounds. This project focuses on reproducing the training algorithm and studying its generalization behaviour empirically.

---

## Model & Task

- **Architecture:** FC600 — a fully-connected network with two hidden layers of 600 units each (28×28 → 600 → 600 → 1, sigmoid output).
- **Task:** Binary classification on binarized MNIST (digits 0–4 → label 1, digits 5–9 → label 0).
- **Loss:** Bounded Binary Cross-Entropy with the transformation ψ(p) = e^{-L_max} + (1 − 2e^{-L_max}) · p, L_max = 4.

---

## Algorithm

The notebook implements **Algorithm 1: Entropy-SGLD**, which consists of:

- An **inner loop** of L = 20 SGLD steps exploring a local basin around the current weights, computing a running average μ.
- An **outer loop** performing a gradient ascent step on the entropy landscape, with decaying step size η_t ∝ t^{-0.6}.

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| m | 60,000 | Training set size |
| τ | √m ≈ 244.9 | Temperature scaling |
| γ | 1.0 | Proximal regularization |
| β | 100.0 | Inverse temperature (outer loop) |
| α | 0.75 | EMA coefficient for μ |
| L | 20 | Inner SGLD steps |
| K | 128 | Batch size |

> **Note:** Some hyperparameters (η_prime_base, β) were intentionally set away from the paper's values to obtain visible learning within a reasonable number of epochs.

---

## Experiments

### 1. Binary MNIST (50 epochs)
The model is trained on standard binarized MNIST. The test error consistently stays below the train error and keeps decreasing throughout training, suggesting strong generalization without overfitting.

### 2. Random Label Experiment (10 epochs)
Labels are randomized to check for overfitting behaviour. Entropy-SGLD is compared against standard SGD with momentum:
- **SGD** overfits significantly — train error drops while test error remains near 50%.
- **Entropy-SGLD** resists overfitting, confirming the regularization effect of the entropy term.

### 3. FashionMNIST (30 epochs)
The same binarized setup (classes 0–4 vs. 5–9) is applied to FashionMNIST. The model achieves a large error drop at the start and continues to improve, validating the algorithm's ability to generalize beyond MNIST.

---

## PAC-Bayes Bound (Attempted)

An attempt was made to compute the PAC-Bayes generalization bound using the KL divergence estimator described in Appendix C.3.2 of the paper (Monte Carlo estimation of KL(P_exp(−ℓ) ‖ P)). Due to instability in the estimator, including negative KL values, this part was left incomplete in favour of focusing on the main training algorithm and its empirical evaluation.

---

## Requirements

```bash
pip install torch torchvision matplotlib tqdm scipy numpy
```

A CUDA-compatible GPU is recommended for reasonable training times.

---

## Usage

Run the notebook cells sequentially. The notebook will automatically download MNIST and FashionMNIST via `torchvision.datasets`. Training progress (train/test error per epoch) is printed and plotted at the end of each experiment.

---

## References

- Dziugaite, G. K., & Roy, D. M. (2018). *Entropy-SGD optimizes the prior of a PAC-Bayes bound*. ICML 2018. [arXiv:1712.09376](https://arxiv.org/abs/1712.09376)
