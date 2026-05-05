# 🧠 Lab Manual – Generative Models & Deep Learning
### Experiments 1–10 | Python · PyTorch · NumPy · Scikit-learn

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Experiment Summaries](#experiment-summaries)
   - [Exp 1 – Random Sampling from Probability Distributions](#experiment-1--random-sampling-from-probability-distributions)
   - [Exp 2 – Maximum Likelihood Estimation (MLE)](#experiment-2--maximum-likelihood-estimation-mle)
   - [Exp 3 – Neural Network Training with Backpropagation](#experiment-3--neural-network-training-with-backpropagation)
   - [Exp 4 – Generative Adversarial Network (GAN)](#experiment-4--generative-adversarial-network-gan)
   - [Exp 5 – Flow-Based Generative Model](#experiment-5--flow-based-generative-model)
   - [Exp 6 – Evaluation Metrics](#experiment-6--evaluation-metrics)
   - [Exp 7 – Model Performance Evaluation on Validation Set](#experiment-7--model-performance-evaluation-on-validation-set)
   - [Exp 8 – Variational Autoencoder (VAE)](#experiment-8--variational-autoencoder-vae)
   - [Exp 9 – Autoregressive Model (PixelCNN)](#experiment-9--autoregressive-model-pixelcnn)
   - [Exp 10 – GAN Loss Function Analysis](#experiment-10--gan-loss-function-analysis)
4. [Project – Parameter Estimation for a Probability Distribution](#project--parameter-estimation-for-a-probability-distribution)
5. [Quick Reference Table](#quick-reference-table)

---

## Overview

This lab manual covers 10 hands-on experiments in the areas of **probability theory**, **deep learning**, and **generative modelling**. Each experiment is provided as a self-contained Jupyter Notebook (`.ipynb`) with step-by-step code cells, markdown explanations, and visualizations.

The experiments progress from foundational statistics (sampling, MLE) through to advanced generative architectures (GANs, VAEs, Normalizing Flows, PixelCNN).

---

## Prerequisites & Setup

### Required Libraries

Install all dependencies with:

```bash
pip install numpy scipy matplotlib scikit-learn scikit-image
pip install torch torchvision
```

### Running the Notebooks

1. Launch Jupyter: `jupyter notebook` or open in **VS Code / Google Colab**
2. Open any `.ipynb` file from this folder
3. Run cells top to bottom using **Shift + Enter**

> **Note:** Experiments 3, 4, 7, 8, and 9 download the MNIST dataset automatically on first run (~11 MB). An internet connection is required for the first run only.

---

## Experiment Summaries

---

### Experiment 1 – Random Sampling from Probability Distributions

**File:** `Experiment_1_Random_Sampling.ipynb`

**Aim:** Generate and visualize random samples from three common probability distributions using NumPy.

**Distributions Covered:**
- **Normal Distribution** – bell-shaped, defined by mean (μ) and standard deviation (σ)
- **Uniform Distribution** – equal probability across a range [low, high]
- **Exponential Distribution** – models time between events, defined by rate parameter λ

**What the Code Does:**
- Generates 1000 random samples from each distribution
- Prints summary statistics (mean, std, min, max)
- Plots histograms for all three distributions side by side

**Key Functions Used:**
```python
np.random.normal(mu, sigma, n)
np.random.uniform(low, high, n)
np.random.exponential(scale, n)
```

**Expected Output:** Three side-by-side histograms showing the shape of each distribution.

---

### Experiment 2 – Maximum Likelihood Estimation (MLE)

**File:** `Experiment_2_MLE.ipynb`

**Aim:** Implement MLE to estimate the parameters (μ, σ) of a Normal distribution from a synthetic dataset, and compare with the true parameters.

**Concept:**
MLE finds the parameter values that maximize the probability of observing the given data. For a Normal distribution:
- **MLE of μ** = sample mean
- **MLE of σ** = sample standard deviation (with N in the denominator)

**What the Code Does:**
- Generates a synthetic dataset with known true parameters (μ=5.0, σ=2.0)
- Computes MLE estimates using `np.mean()` and `np.std()`
- Calculates estimation error vs. true parameters
- Plots the data histogram alongside the true and estimated distributions

**Expected Output:** A plot showing the observed data, the true Normal curve, and the MLE-fitted curve — they should closely overlap.

---

### Experiment 3 – Neural Network Training with Backpropagation

**File:** `Experiment_3_Neural_Network_MNIST.ipynb`

**Aim:** Train a fully connected neural network on the MNIST handwritten digit dataset using stochastic gradient descent (SGD) and backpropagation.

**Architecture:**
```
Input (784) → FC Layer (128, ReLU) → FC Layer (64, ReLU) → Output (10, Softmax)
```

**What the Code Does:**
- Downloads and loads the MNIST dataset (60,000 training images)
- Defines a 3-layer neural network using `torch.nn`
- Trains for 5 epochs using SGD with momentum
- Evaluates accuracy on the test set
- Plots the training loss curve

**Key Concepts:**
- **Backpropagation**: `loss.backward()` computes gradients automatically
- **Optimizer step**: `optimizer.step()` updates weights using computed gradients
- **Cross-Entropy Loss**: used for multi-class classification

**Expected Output:** Training loss decreasing each epoch; test accuracy around **97%**.

---

### Experiment 4 – Generative Adversarial Network (GAN)

**File:** `Experiment_4_GAN_MNIST.ipynb`

**Aim:** Train a GAN on MNIST and monitor the competing losses of the Generator and Discriminator.

**How a GAN Works:**
- **Generator (G):** Takes random noise z ∈ ℝ¹⁰⁰ and generates fake images
- **Discriminator (D):** Tries to tell real images from fake ones
- They compete: G tries to fool D, D tries not to be fooled

**Architecture:**
```
Generator:    Noise(100) → FC(256) → FC(512) → FC(784) → Image(28×28)
Discriminator: Image(784) → FC(512) → FC(256) → FC(1) → Real/Fake Score
```

**What the Code Does:**
- Trains G and D alternately for 5 epochs
- Records Generator and Discriminator losses per epoch
- Plots the loss curves to observe the adversarial training dynamic
- Visualizes 16 generated digit images

**Expected Output:** G loss and D loss competing (not both decreasing), and generated images that resemble handwritten digits.

---

### Experiment 5 – Flow-Based Generative Model

**File:** `Experiment_5_Flow_Model.ipynb`

**Aim:** Implement a Normalizing Flow model using affine coupling layers and train it on 2D data (Two Moons).

**Key Concept:**
Normalizing Flows learn an **invertible mapping** between a simple distribution (Gaussian) and a complex target distribution. Unlike GANs and VAEs, they allow exact likelihood computation.

**Architecture:**
```
Data x → Coupling Layer 1 → Coupling Layer 2 → ... → Latent z ~ N(0,I)
```

Each **Affine Coupling Layer** splits input, computes scale (s) and shift (t), and applies:
```
y₂ = x₂ · exp(s(x₁)) + t(x₁)
```

**What the Code Does:**
- Generates a Two Moons dataset (sklearn)
- Stacks 4 coupling layers into a flow model
- Trains by minimizing negative log-likelihood
- Samples from the model by inverting the flow
- Visualizes original data vs. generated samples

**Expected Output:** Generated samples that resemble the crescent/moon shape of the original dataset.

---

### Experiment 6 – Evaluation Metrics

**File:** `Experiment_6_Evaluation_Metrics.ipynb`

**Aim:** Implement and interpret common evaluation metrics used for both classification models and generative models.

**Metrics Covered:**

| Metric | Formula | Used For |
|--------|---------|---------|
| Accuracy | Correct / Total | Classification |
| Precision | TP / (TP + FP) | Classification |
| Recall | TP / (TP + FN) | Classification |
| F1-Score | 2 × (P × R) / (P + R) | Classification |
| Perplexity | exp(−mean(log p(x))) | Language/Generative models |
| MSE | mean((y − ŷ)²) | Image reconstruction |
| SSIM | Structural similarity | Image quality |

**What the Code Does:**
- Simulates true and predicted binary labels
- Computes all classification metrics using `sklearn`
- Draws and annotates a confusion matrix
- Computes Perplexity from simulated log-probabilities
- Computes MSE and SSIM between two synthetic images

**Expected Output:** Printed metric values and a labelled confusion matrix heatmap.

---

### Experiment 7 – Model Performance Evaluation on Validation Set

**File:** `Experiment_7_Model_Evaluation.ipynb`

**Aim:** Train a neural network and track both training loss and validation loss/accuracy after every epoch to observe learning behaviour.

**Why Validation Matters:**
A separate validation set (never seen during training) reveals whether the model is **generalizing** or **overfitting**.

**What the Code Does:**
- Splits MNIST into 50,000 training and 10,000 validation samples
- Trains a simple FC network for 5 epochs
- After each epoch, evaluates on the validation set
- Plots training loss, validation loss, and validation accuracy side by side

**Signs to Look For:**
- If val loss decreases alongside train loss → **good generalization**
- If train loss decreases but val loss increases → **overfitting**

**Expected Output:** Two plots — Loss (train vs val) and Validation Accuracy over epochs.

---

### Experiment 8 – Variational Autoencoder (VAE)

**File:** `Experiment_8_VAE.ipynb`

**Aim:** Build and train a VAE on MNIST, then use it to reconstruct and generate new digit images.

**How a VAE Works:**
```
Input x → Encoder → (μ, σ) → Reparameterize → z → Decoder → Reconstruction x̂
```
- **Encoder** maps input to a distribution in latent space (not a single point)
- **Reparameterization Trick**: z = μ + ε·σ (allows gradients to flow)
- **Decoder** maps latent vector z back to image space

**Loss Function:**
```
VAE Loss = Reconstruction Loss (BCE) + KL Divergence
```
- Reconstruction loss: how similar is x̂ to x?
- KL Divergence: how close is the learned distribution to a standard Gaussian?

**What the Code Does:**
- Builds Encoder → (μ, log_var) → Decoder architecture
- Applies reparameterization trick during forward pass
- Trains for 5 epochs using combined VAE loss
- Displays 8 original images alongside their VAE reconstructions

**Expected Output:** Side-by-side grid showing original MNIST digits and their VAE-reconstructed versions.

---

### Experiment 9 – Autoregressive Model (PixelCNN)

**File:** `Experiment_9_PixelCNN.ipynb`

**Aim:** Implement a PixelCNN autoregressive model using masked convolutions to generate images one pixel at a time.

**Key Concept:**
Autoregressive models generate data **sequentially** — each output depends only on previously generated values. PixelCNN achieves this using **masked convolutions** that block access to future pixels.

```
p(x) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · ... · p(xₙ|x₁,...,xₙ₋₁)
```

**Two Mask Types:**
- **Mask A** (first layer): Current pixel is also masked out
- **Mask B** (subsequent layers): Current pixel can see itself

**What the Code Does:**
- Defines `MaskedConv2d` — a Conv2d layer with a binary mask applied to weights
- Stacks 3 masked conv layers to build PixelCNN
- Trains for 3 epochs on MNIST using pixel-level cross-entropy loss
- Generates 4 new images by sampling pixel-by-pixel (row by row)

**Expected Output:** Generated 28×28 images that somewhat resemble MNIST digits, created entirely from scratch pixel by pixel.

---

### Experiment 10 – GAN Loss Function Analysis

**File:** `Experiment_10_GAN_Loss_Functions.ipynb`

**Aim:** Implement and compare three major GAN loss formulations — Original Minimax, LSGAN, and WGAN — and also demonstrate feature matching loss.

**Loss Functions Compared:**

| Loss Type | Discriminator | Generator | Key Benefit |
|-----------|--------------|-----------|-------------|
| **Minimax GAN** | Binary Cross-Entropy | BCE | Original formulation |
| **LSGAN** | Mean Squared Error | MSE | Reduces vanishing gradients |
| **WGAN** | Earth Mover Distance | −D(fake) | Stable, meaningful metric |
| **Feature Matching** | Match feature stats | L2 distance | Stabilizes training |

**What the Code Does:**
- Simulates discriminator output scores for real and fake images
- Computes D loss and G loss under all three formulations
- Plots a grouped bar chart comparing loss values across methods
- Demonstrates feature matching loss using simulated feature activations

**Expected Output:** A bar chart showing how D loss and G loss values differ across Minimax, LSGAN, and WGAN formulations, alongside printed loss values.

---

## Project – Parameter Estimation for a Probability Distribution

**Description:** An end-to-end project applying the MLE algorithm (from Experiment 2) across varying dataset sizes to study estimation performance.

**Tasks:**
1. Choose a probability distribution (e.g., Normal or Exponential) and define its PDF/PMF
2. Generate a synthetic dataset with known true parameters
3. Implement MLE in Python to estimate the parameters
4. Calculate estimation error compared to the true parameters
5. Repeat for multiple dataset sizes (e.g., 10, 50, 100, 500, 1000) and plot how error changes with sample size

**Key Insight:** As sample size increases, MLE estimates converge to the true parameters — this property is called **consistency**.

---

## Quick Reference Table

| # | Experiment | Concepts | Libraries | Dataset |
|---|-----------|----------|-----------|---------|
| 1 | Random Sampling | PDF, sampling | NumPy, Matplotlib | Synthetic |
| 2 | MLE | Parameter estimation | NumPy, SciPy | Synthetic |
| 3 | Neural Network | Backpropagation, SGD | PyTorch | MNIST |
| 4 | GAN | Adversarial training | PyTorch | MNIST |
| 5 | Flow Model | Normalizing flows, invertible maps | PyTorch, sklearn | Synthetic |
| 6 | Eval Metrics | Accuracy, F1, SSIM, Perplexity | sklearn, skimage | Synthetic |
| 7 | Validation | Overfitting, generalization | PyTorch | MNIST |
| 8 | VAE | Encoder-decoder, KL divergence | PyTorch | MNIST |
| 9 | PixelCNN | Autoregressive, masked conv | PyTorch | MNIST |
| 10 | GAN Losses | Minimax, LSGAN, WGAN | PyTorch | Synthetic |

---

*Lab Manual – Generative Models & Deep Learning*
