# Positive Definite Sliced Wasserstein Graph Kernel for Brain Disease Diagnosis

## Independent Algorithmic Reproduction & Synthetic Validation Study

### Based on MICCAI 2023 Research

![Domain](https://img.shields.io/badge/Domain-Medical%20AI-blue)
![Method](https://img.shields.io/badge/Method-Optimal%20Transport%20%7C%20Graph%20Kernel-green)
![Conference](https://img.shields.io/badge/Reference-MICCAI%202023-orange)
![Language](https://img.shields.io/badge/Python-3.10-yellow)


---

# Overview

This repository presents an **independent algorithmic reproduction and synthetic validation study** of:

> **Positive Definite Wasserstein Graph Kernel for Brain Disease Diagnosis**
> Ma et al., MICCAI 2023

The original work introduces a novel graph kernel framework that combines:

* Graph spectral representation
* Optimal transport geometry
* Sliced Wasserstein distance
* Positive definite kernel learning
* Kernel-based classification

for the analysis of brain functional connectivity networks.

This project reconstructs the complete computational pipeline from the mathematical formulation of the paper and evaluates the behavior of the proposed Sliced Wasserstein Graph Kernel (SWG) under controlled synthetic brain network conditions.

---

# Research Objective

The main objectives of this project are:

1. **Independent reproduction of the SWG graph kernel framework**

2. **Implementation of the complete mathematical pipeline**

3. **Construction of synthetic functional brain networks with disease-specific connectivity alterations**

4. **Evaluation using kernel SVM classification**

5. **Visualization of graph-level geometric differences using Wasserstein distance matrices**

---

# Why This Repository Matters

This project is not only a software implementation.

The complete methodology was independently reconstructed from the original research formulation:

✓ Graph representation

✓ Laplacian spectral embedding

✓ Wasserstein-based graph comparison

✓ Sliced projection approximation

✓ Positive definite kernel construction

✓ Kernel SVM classification

✓ Cross-validation evaluation framework

The repository serves as a reproducible research artifact demonstrating the connection between:

* Medical AI
* Graph Machine Learning
* Optimal Transport
* Computational Neuroscience

---

# Scientific Background

Brain functional connectivity networks can be represented as weighted graphs:

* Nodes → brain regions
* Edges → functional connectivity strength

Traditional graph analysis often relies on handcrafted local features:

* Degree statistics
* Clustering coefficients
* Centrality measures

However, these descriptors may not fully capture global structural differences between brain networks.

The SWG framework addresses this challenge by representing graphs as geometric distributions in spectral space.

---

# Method Overview

```
Brain Connectivity Graph
          |
          v
Laplacian Spectral Embedding
          |
          v
Node Distribution in Latent Space
          |
          v
Sliced Wasserstein Distance
          |
          v
Positive Definite Graph Kernel
          |
          v
Kernel SVM Classification
```

---

# Implemented Methodology

## 1. Synthetic Brain Network Generation

Because the original neuroimaging datasets were unavailable in this reproduction setting, synthetic functional connectivity networks were generated.

Each graph contains:

* 90 brain regions
* 6 modular communities
* Weighted adjacency matrix
* Gaussian subject-level noise
* Disease-specific connectivity perturbations

Dataset configuration:

```
100 subjects per disease experiment

50 Healthy Controls (NC)
50 Disease Subjects
```

---

# Disease-Specific Network Models

## ADHD

Simulated alteration:

* Reduced fronto-parietal connectivity
* Weakening of inter-module communication

## ASD

Simulated alteration:

* Increased connectivity within a DMN-like module
* Reduced cross-module connectivity

## EMCI

Simulated alteration:

* Temporal-occipital network disruption
* Reduced intra-module connectivity

---

# Core Algorithm

## Laplacian Spectral Embedding

For each brain graph:

$$
L=D-A
$$

where:

* A = adjacency matrix
* D = degree matrix

The graph is embedded using the smallest non-zero eigenvectors:

$$
X_G \in R^{90 \times 10}
$$

Each graph becomes a point cloud representation in spectral space.

---

# Graph Sliced Wasserstein Distance

The embedded graphs are compared using Wasserstein geometry.

Random projection directions:

$$
\theta_1,\theta_2,...,\theta_L
$$

For each projection:

1. Project node embeddings
2. Sort projected distributions
3. Compute 1D Wasserstein distance

The final graph distance:

$$
D_{GSW}(G_1,G_2)
$$

is obtained by averaging over multiple slices.

---

# Positive Definite SWG Kernel

The graph similarity is defined as:

$$
K_{SWG}(G_1,G_2)
$$

$$
e^{-\lambda D_{GSW}(G_1,G_2)}
$$

This produces a positive definite kernel matrix suitable for SVM learning.

---

# Machine Learning Pipeline

Classifier:

```
Support Vector Machine
Kernel: Precomputed SWG Kernel
```

Evaluation:

```
Leave-One-Out Cross Validation
```

Hyperparameter search:

| Parameter | Values                |
| --------- | --------------------- |
| λ         | 0.01, 0.1, 1, 10, 100 |
| C         | 0.001 - 100           |

Metrics:

* Accuracy
* ROC-AUC
* Runtime
* Wasserstein distance visualization

---

# Experimental Results

## Synthetic Validation Performance

| Dataset    | Accuracy |   AUC | Interpretation                 |
| ---------- | -------: | ----: | ------------------------------ |
| ADHD vs NC |    67.0% | 0.726 | Subtle connectivity alteration |
| ASD vs NC  |     100% | 1.000 | Strong modular separation      |
| EMCI vs NC |     100% | 1.000 | Strong structural disruption   |

---

# Comparison With Original MICCAI 2023 Results

| Dataset | Original Paper Accuracy | Synthetic Validation |
| ------- | ----------------------: | -------------------: |
| ADHD    |                  78.83% |                67.0% |
| ASD     |                  90.54% |               100.0% |
| EMCI    |                  85.44% |               100.0% |

Important:

The results in this repository are obtained from **synthetic brain networks**, not the original fMRI datasets.

Therefore:

* These values demonstrate algorithmic sensitivity.
* They should not be interpreted as clinical diagnostic performance.
* Real-world validation requires external neuroimaging datasets.

---

# Distance Matrix Visualization

The Wasserstein distance matrices provide an interpretable view of graph-level separation.

## ADHD

![ADHD Heatmap](results/heatmap_ADHD.png)

## ASD

![ASD Heatmap](results/heatmap_ASD.png)

## EMCI

![EMCI Heatmap](results/heatmap_EMCI.png)

---

# Repository Structure

```
SWG-Kernel-Brain-Network-Diagnosis/

│
├── paper/
│   └── Positive_Definite_Wasserstein_Graph_Kernel_MICCAI2023.pdf
│
├── presentation/
│   └── Positive Definite Sliced Wasserstein Graph Kernel.pptx
│
├── report/
│   └── PDSW_Graph_Kernel_Report.pdf
│
├── results/
│   ├── heatmap_ADHD.png
│   ├── heatmap_ASD.png
│   └── heatmap_EMCI.png
│
├── src/
│   └── swg_brain_diagnosis_pipeline.py
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

# Technical Stack

## Programming

* Python 3.10
* NumPy
* SciPy
* scikit-learn
* Matplotlib

## Machine Learning

* Kernel SVM
* Leave-One-Out Cross Validation
* Hyperparameter Optimization

## Mathematical Framework

* Spectral Graph Theory
* Optimal Transport
* Wasserstein Distance
* Positive Definite Kernels

---

# Installation

```bash
git clone https://github.com/hannahfathi99/SWG-Kernel-Brain-Network-Diagnosis.git

cd SWG-Kernel-Brain-Network-Diagnosis

pip install -r requirements.txt
```

---

# Running the Experiment

```bash
python src/swg_brain_diagnosis_pipeline.py
```

The pipeline automatically:

1. Generates synthetic brain graphs
2. Computes spectral embeddings
3. Builds SWG kernel matrices
4. Performs SVM classification
5. Reports Accuracy and AUC
6. Generates Wasserstein heatmaps

---

# Reproducibility

All experiments use:

* Fixed random seed = 42
* Deterministic synthetic graph generation
* Embedding dimension:

$$
d=10
$$

* Number of Wasserstein slices:

$$
L=50
$$

The complete experiment can be reproduced from the provided source code.

---

# Research Contributions

## 1. Independent Mathematical Reimplementation

The SWG framework was reconstructed from the original mathematical description without relying on unavailable official implementation.

## 2. Integration of Optimal Transport and Graph Learning

This project combines:

* Graph spectral methods
* Wasserstein geometry
* Kernel learning
* Medical AI

## 3. Disease-Specific Brain Network Simulation

A biologically motivated synthetic framework was designed to model:

* Network disruption
* Modular connectivity changes
* Disease-related graph geometry alterations

## 4. Reproducible Research Artifact

The repository provides:

* Source implementation
* Scientific report
* Research presentation
* Original paper reference
* Experimental results
* Visualization outputs

---

# Limitations

## Synthetic Data

Synthetic networks cannot fully capture:

* Biological variability
* Scanner effects
* Individual differences
* Real fMRI complexity

## Evaluation Strategy

The current evaluation uses flat hyperparameter search.

Future studies should include:

* Nested cross-validation
* Statistical significance testing
* External dataset validation

---

# Future Research Directions

## Real Neuroimaging Validation

Potential datasets:

* ADHD-200
* ABIDE
* ADNI

## Advanced Graph Representation Learning

Future extensions:

* Graph Neural Networks
* Neural Optimal Transport
* Learnable Wasserstein projections

## Multi-scale Graph Kernels

Investigate multiple embedding dimensions:

$$
d \in {5,10,20,50}
$$

for richer structural representations.

---

# References

Ma et al.

**Positive Definite Wasserstein Graph Kernel for Brain Disease Diagnosis**

MICCAI 2023.

---

# Acknowledgement

This repository was developed as a graduate-level Artificial Intelligence research project exploring the intersection of:

* Medical AI
* Graph Machine Learning
* Optimal Transport
* Brain Network Analysis

It provides a reproducible foundation for future experiments on real neuroimaging datasets.

---

# Citation

If you use this repository in academic work, please cite:

```
Fathi, Hannah.
Positive Definite Sliced Wasserstein Graph Kernel for Brain Disease Diagnosis:
Independent Algorithmic Reproduction and Synthetic Validation Study.
Fall 2025.
```

---

# Research Timeline

**Fall 2025**

# Author

**Hannah Fathi**
M.Sc. Artificial Intelligence Student

Research Interests:

* Computer Vision
* Medical AI
* Remote Sensing
* Large Language Models (LLMs)
* Graph Machine Learning
* Optimal Transport
* Brain Network Analysis
