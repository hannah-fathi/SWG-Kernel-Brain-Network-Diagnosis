# Positive Definite Sliced Wasserstein Graph Kernel for Brain Disease Diagnosis

### Algorithmic Reproduction and Synthetic Validation Study (MICCAI 2023)

![Research Area](https://img.shields.io/badge/Domain-Medical%20AI-blue)
![Method](https://img.shields.io/badge/Method-Optimal%20Transport%20%2B%20Graph%20Kernel-green)
![Paper](https://img.shields.io/badge/Reference-MICCAI%202023-orange)
![Python](https://img.shields.io/badge/Python-3.10-yellow)


---

# Overview

This repository presents an independent implementation and experimental validation of the:

> **Positive Definite Wasserstein Graph Kernel for Brain Disease Diagnosis**

proposed by Ma et al., MICCAI 2023.

The original work introduces a novel graph kernel based on **Sliced Wasserstein Distance (SWD)** applied to **Laplacian spectral embeddings of brain functional networks**.

This project focuses on:

1. Reproducing the mathematical pipeline of the proposed Sliced Wasserstein Graph Kernel (SWG).
2. Implementing the complete graph classification framework.
3. Designing synthetic brain functional connectivity networks with disease-specific alterations.
4. Evaluating the kernel using SVM classification and Leave-One-Out Cross Validation.
5. Providing interpretable distance visualizations through GSW heatmaps.

Because the original neuroimaging datasets were not publicly available in this reproduction setting, synthetic functional connectivity graphs were generated to validate the algorithmic behavior of the proposed method.

---

# Scientific Motivation

Brain functional networks can be represented as weighted graphs where:

* Nodes represent brain regions.
* Edges represent functional connectivity strength.

Traditional graph analysis methods often rely on handcrafted local statistics:

* Degree
* Clustering coefficient
* Centrality measures

However, these features may fail to capture global geometric differences between brain networks.

The SWG framework addresses this limitation by combining:

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

Synthetic functional connectivity matrices are generated with:

* 90 brain regions
* 6 modular communities
* Weighted adjacency matrices
* Gaussian subject-level noise
* Disease-specific connectivity perturbations

Implemented disease models:

### ADHD

Simulated disruption:

* Reduced fronto-parietal connectivity
* Inter-module communication weakening

### ASD

Simulated alterations:

* Increased connectivity inside a DMN-like module
* Reduced cross-module interaction

### EMCI

Simulated degeneration pattern:

* Temporal-occipital connectivity disruption
* Reduced intra-module connectivity

Each dataset contains:

```
100 subjects

50 Healthy Controls (NC)
50 Disease Subjects
```

---

# Core Algorithm

## Laplacian Spectral Embedding

For each graph:

$$
L=D-A
$$

where:

* A = adjacency matrix
* D = degree matrix

The graph is embedded into:

$$
R^{90 \times 10}
$$

using the smallest non-zero Laplacian eigenvectors.

---

## Graph Sliced Wasserstein Distance

The embedded graphs are treated as point clouds.

Random projections:

$$
\theta_1,\theta_2,...,\theta_L
$$

are generated on the unit sphere.

For each projection:

1. Project node embeddings.
2. Sort projected distributions.
3. Compute 1D Wasserstein distance.

The final distance:

$$
D_{GSW}(G_1,G_2)
$$

is obtained by averaging over slices.

---

## Positive Definite SWG Kernel

The graph similarity is defined as:

$$
K_{SWG}(G_1,G_2)
$$


$$
exp(-\lambda D_{GSW}(G_1,G_2))
$$

This allows direct usage with kernel-based classifiers.

---

# Machine Learning Pipeline

Classifier:

```
Support Vector Machine
(kernel = precomputed)
```

Evaluation:

```
Leave-One-Out Cross Validation
```

Hyperparameter search:

| Parameter | Values            |
| --------- | ----------------- |
| λ         | 0.01,0.1,1,10,100 |
| C         | 0.001-100         |

Metrics:

* Accuracy
* ROC-AUC
* Runtime
* Distance matrix visualization

---

# Experimental Results

## Synthetic Dataset Performance

| Disease | Accuracy | AUC   |
| ------- | -------- | ----- |
| ADHD    | 67.0%    | 0.726 |
| ASD     | 100%     | 1.000 |
| EMCI    | 100%     | 1.000 |

---

# Comparison With Original Paper

Original MICCAI 2023 Results:

| Dataset | Paper Accuracy |
| ------- | -------------- |
| ADHD    | 78.83%         |
| ASD     | 90.54%         |
| EMCI    | 85.44%         |

Important:

The results in this repository are obtained on **synthetic networks**, not the original fMRI datasets.

Therefore:

* 100% accuracy does not represent clinical performance.
* Synthetic results demonstrate algorithmic sensitivity.
* Real-world validation requires public neuroimaging datasets.

---

# Visualization Results

The repository provides GSW distance heatmaps:

```
results/

├── heatmap_ADHD.png
├── heatmap_ASD.png
└── heatmap_EMCI.png
```

The heatmaps demonstrate:

* Within-group similarity
* Patient/control separation
* Disease-specific geometric changes

---

# Repository Structure

```
.
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
│   └── swg_kernel_reproduction.py
│
├── requirements.txt
├── LICENSE
└── README.md

```

---

# Installation

```bash
git clone <repository-url>

cd sliced-wasserstein-graph-kernel-reproduction

pip install -r requirements.txt
```

---

# Running the Experiment

```bash
python src/swg_kernel_reproduction.py
```

The script will:

1. Generate synthetic brain graphs.
2. Compute Laplacian embeddings.
3. Build SWG kernels.
4. Perform SVM classification.
5. Report accuracy/AUC.
6. Generate heatmaps.

---

# Research Contributions

This project demonstrates:

## 1. Independent Mathematical Reimplementation

The SWG framework was reconstructed from the original mathematical formulation rather than relying on unavailable official code.

---

## 2. Integration of Optimal Transport and Graph Learning

The project combines:

* Graph spectral theory
* Optimal transport
* Wasserstein geometry
* Kernel methods
* Medical AI

---

## 3. Disease-Specific Graph Simulation

A biologically motivated synthetic generation framework was designed to simulate:

* Altered modular connectivity
* Network disruption
* Disease-related topology changes

---

## 4. Reproducible Research Artifact

The repository contains:

* Source code
* Scientific report
* Presentation slides
* Original paper reference
* Generated experimental results

---

# Limitations

## Synthetic Data

The generated networks cannot fully reproduce:

* Biological variability
* Scanner differences
* Subject heterogeneity
* Real fMRI noise

## Validation Strategy

The current evaluation uses flat hyperparameter search with LOO-CV.

Future work should include:

* Nested cross-validation
* External validation datasets
* Statistical significance testing

---

# Future Research Directions

Potential extensions:

## Real Neuroimaging Validation

Apply the framework to:

* ADHD-200
* ABIDE
* ADNI

## Advanced Graph Representation Learning

Integrate:

* Graph Neural Networks
* Neural Optimal Transport
* Learnable Wasserstein projections

## Multi-scale Graph Kernels

Explore multiple embedding dimensions:

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

The implementation serves as a reproducible foundation for future experiments on real neuroimaging datasets.

---

## Research Timeline

**Fall 2025**

## Author

**Hannah Fathi**
M.Sc. Artificial Intelligence Student

Research Interests:

* Computer Vision
* Medical AI
* Remote Sensing
* Large Language Models (LLMs)
* Graph Machine Learning
* Optimal Transport
