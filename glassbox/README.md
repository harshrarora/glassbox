# GlassBox — Interactive Mathematical Simulation Engine

> **A Visual Mathematical Laboratory for Machine Learning & Data Science**  
> Pedagogical visual simulation engine built in the spirit of 3Blue1Brown, Bret Victor's Explorable Explanations, and Distill.pub.

GlassBox transforms textbook formulas into interactive mathematical microscopes. Rather than presenting static tables or black-box predictions, students inspect step-by-step mathematical substitutions with real numbers plugged in, observe vector projections, manipulate SVD rank-$k$ approximations, and watch Support Vector Machine margins expand and contract live.

---

## 1. Quick Start & Execution Commands

### Environment Setup
Dependencies require Python 3.10+ with `numpy`, `streamlit`, `plotly`, `pandas`, `pytest`, and `scikit-learn`:

```bash
# Navigate to project directory
cd D:/sem5/ML/project

# Install dependencies
pip install -r glassbox/requirements.txt
```

### Launch Interactive Teaching Lab
> [!NOTE]
> If Windows Application Control / AppLocker blocks `streamlit.exe`, invoke Streamlit directly through `python -m streamlit`:

```powershell
# Set PYTHONPATH and launch via Python module
cd D:\sem5\ML\project
$env:PYTHONPATH = "D:\sem5\ML\project"
python -m streamlit run glassbox/app.py
```

Alternatively, you can run the helper script:
```powershell
.\run_app.ps1
```
or double-click `run_app.bat`.

### Run Mathematical Verification Test Suite (15 Unit Tests)
```powershell
$env:PYTHONPATH = "D:\sem5\ML\project"
python -m pytest glassbox/tests -v
```

---

## 2. The Core Data: Real 64-Paper Curated arXiv Corpus

GlassBox is powered by an offline curated corpus of **64 landmark arXiv research papers** across 4 balanced categories:
- **`cs.CV` (Computer Vision):** 16 papers (*ResNet, ViT, YOLO, NeRF, U-Net, GAN, Mask R-CNN, Latent Diffusion, Segment Anything, Swin, PointNet, AlexNet, CLIP, DDPM, SimCLR, VGG*).
- **`cs.CL` (Computation & Language / NLP):** 16 papers (*Attention Is All You Need / Transformer, BERT, GPT-3, RoBERTa, Word2Vec, T5, LLaMA, LoRA, ELMo, Sentence-BERT, InstructGPT, Seq2Seq, FlashAttention, Bahdanau Attention, Chain-of-Thought, DPO*).
- **`cs.LG` (Deep Learning Theory):** 16 papers (*Adam, BatchNorm, Dropout, Double Descent, Understanding Deep Learning Generalization, Neural Tangent Kernel, LayerNorm, ResNet Ensembles, AdamW, Large-Batch Sharp Minima, Implicit Bias, VAE, Contrastive Predictive Coding, Loss Landscape, Fixup Initialization*).
- **`stat.ML` (Optimization & Statistical Learning):** 16 papers (*Bottou SGD, FISTA Proximal Gradient, PAC-Bayes, Stein Variational Gradient Descent, Computational Optimal Transport, Differential Privacy DP-SGD, Frank-Wolfe, Minimax Linear Regression, Distributionally Robust Optimization, Variational Inference, Conformal Prediction, Kernel Ridge Regression, Algorithmic Stability, Non-Convex Optimization, High-Dimensional M-Estimators, Accelerated Stochastic Search*).

---

## 3. Four-Stage Syllabus Architecture

### Stage 1: Feature Representation, 3D PCA & SVD Rank-k Simulator
- **The Token Microscope:** Abstract text where words light up with visual intensity proportional to their exact $L_2$-normalized TF-IDF coordinate.
- **Interactive Mathematical Derivation:** Shows the exact formula with real numbers plugged in:
  $$\text{IDF}(t) = \ln\left(\frac{1 + 64}{1 + \text{DF}(t)}\right) + 1 = \ln\left(\frac{65}{1 + \dots}\right) + 1 = \dots$$
  $$z(t, d) = c(t, d) \times \text{IDF}(t), \quad x(t, d) = \frac{z(t, d)}{\|z(:, d)\|_2}$$
- **2D Vector Strip Heatmap:** Interactive Plotly heatmap displaying $(d, t)$ coordinates with hover inspection.
- **3D PCA & SVD Canvas:**
  - 64 papers plotted in 3D coordinate space with domain colors.
  - Mean centroid $\boldsymbol{\mu}$, sample covariance matrix $\mathbf{\Sigma}$ rendered as an interactive heatmap, and principal component axes as 3D vectors.
  - **SVD Rank-$k$ Simulator:** Toggle rank $k \in \{1, 2, 3\}$. Watch vectors snap from a 1D line ($k=1$) to the 2D principal plane ($k=2$) and back to original coordinates ($k=3$), tracking singular values $\sigma_i$ and Frobenius reconstruction error $\|X - \hat{X}_k\|_F$.

### Stage 2: Supervised Linear Boundaries (LDA & SVM)
- **Linear Discriminant Analysis (LDA) Simulator:**
  - Interactive heatmaps for the **Within-Class Scatter Matrix ($S_W$)** and **Between-Class Scatter Matrix ($S_B$)**:
    $$S_W = \sum_{c=1}^C \sum_{x \in \mathcal{C}_c} (x - \mu_c)(x - \mu_c)^T, \quad S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T$$
  - Solves the generalized eigenvalue problem $(S_W^{-1} S_B) w = \lambda w$.
  - 2D projection scatter plot contrasting LDA with PCA to demonstrate how class-aware projection contracts within-class spread while maximizing centroid separation.
- **Support Vector Machine (SVM) Margin Explorer:**
  - Projects papers to 2D and fits a linear classifier for any category pair (e.g. Vision vs NLP).
  - Explicitly highlights **Support Vectors** with glowing rings.
  - Draws the separating hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ and margin gutters $\mathbf{w}^T \mathbf{x} + b = \pm 1$.
  - **Regularization Slider $C$:** Move $C \in [0.01, 20.0]$ live to observe the margin width $\gamma = \frac{2}{\|\mathbf{w}\|_2}$ expand (soft margin) or contract (hard margin).
  - Individual point inspector calculating functional margins $y_i (\mathbf{w}^T \mathbf{x}_i + b)$ and slack variables $\xi_i = \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$.

### Stage 3: Evidence Retrieval & Statistical Shift Laboratory
- Preserves the 10-abstract teaching fixture demonstrating how global corpus statistics alter local retrieval rankings.
- One-click teaching preset **"Hide new papers, refresh IDF"** demonstrating how diluting the rarity of a key term causes an untouched paper (**DOC-03**) to drop from Rank #5 out of the top-five shortlist.

---

## 4. Syllabus Mapping to Course Materials

| GlassBox Feature | Course Module & Material Consulted | Direct Theoretical Alignment |
| :--- | :--- | :--- |
| **Token Microscope & Vector Strips** | `UCS551 ADS Lab: 2.Handling Unstructured Textual Data/6.Lab Session 3(b)` & `UML501: 4_Data Pre-Processing I` | Term frequencies, Laplace smoothed $IDF$, and $L_2$ Euclidean normalization. |
| **3D PCA & SVD Rank-$k$** | `UCS551 ADS Theory: 6.Data Pre-Processing-IV.pdf` & `Lab: PCA_Scratch_Inbuilt.ipynb` | Sample covariance $\Sigma = \frac{1}{n-1} X_c^T X_c$, symmetric Jacobi eigensolution, and Eckart-Young-Mirsky truncated SVD approximation. |
| **Multiclass LDA Scatter** | `UCS551 ADS Theory: 8.Data Pre-Processing-VI.pdf` & `Lab: LDA_Scratch_Inbuilt.ipynb` | Within-class scatter $S_W$, between-class scatter $S_B$, and Fisher generalized eigenproblem $(S_W^{-1} S_B) w = \lambda w$. |
| **SVM Margin Gutters** | `UML501: 2025_Syllabus.pdf` & `Classification Evaluation Metrics` | Linear separating hyperplanes, support vectors on margin boundaries, and slack variable penalty $C$. |

---

## 5. Verification Suite

15 unit tests verify numerical accuracy, scikit-learn reference agreement, and mathematical invariant properties:
- `test_classifiers.py`: arXiv corpus balance (16 per category), SVD rank-$k$ monotonicity, LDA scatter matrix symmetry/eigenvalues, and SVM margin width $\gamma = 2 / \|\mathbf{w}\|$.
- `test_math.py`: Hand calculations, scikit-learn $TF$-$IDF$ agreement within $10^{-5}$, Jacobi eigenpair residuals $< 10^{-10}$, and projection orthogonality.
- `test_shift.py`: All 4 Evidence Under Shift combinations and target paper demotion.
