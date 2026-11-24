# GlassBox: Interactive Mathematical Simulation Engine

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

### Run Mathematical Verification Test Suite (21 Unit Tests)
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

## 3. The 6-Stage End-to-End Interactive Pipeline Architecture

### STAGE 1: Raw Corpus Hub
- **Purpose:** Inspect unstructured scientific literature before machine learning operations.
- **Features:**
  - Complete browsing of 64 arXiv records with category chips, word counts, and metadata.
  - Category distribution bar chart ($16 \times 4$ balanced partition).
  - Corpus summary metrics (vocabulary baseline, character counts, average document length).

### STAGE 2: Preprocessing & Linguistics Lab (Interactive Diffs)
- **Purpose:** Follow the progressive sanitization and normalization of unstructured text into clean feature tokens.
- **Features:**
  - **Living Visual Diffs:** Select any abstract and watch text dynamically transform. Strikethrough red chips show stripped LaTeX math ($...$), URLs, digits, and punctuation.
  - **Stopword Elimination:** Interactive table of filtered stopwords with their occurrence frequencies across the corpus.
  - **Porter Stemmer vs. WordNet Lemmatizer Arena:** Side-by-side morphological comparison highlighting divergent transformations (e.g. `studies` $\to$ `studi` vs. `study`, `running` $\to$ `run`, `better` $\to$ `better` vs. `good`).
  - **Vocabulary Contraction Funnel:** Dynamic stage-by-stage contraction meter tracking vocabulary reduction from Raw Words $\to$ Regex $\to$ Stopwords $\to$ Lemmatization.

### STAGE 3: Vectorization & Feature Engineering
- **Purpose:** Translate discrete linguistic tokens into numerical geometry.
- **Features:**
  - Count Matrix vs. TF-IDF toggle.
  - **Plugged-in Math Substitution Cards:** Live numerical substitution for every term:
    $$\text{IDF}(t) = \ln\left(\frac{1 + 64}{1 + \text{DF}(t)}\right) + 1 = \dots$$
  - **N-gram Explosion Inspector:** Unigrams vs. Bigrams with feature space expansion metrics.
  - **Matrix Sparsity Visualizer:** Interactive 2D heatmap showing $(d, t)$ coordinates with cell-level inspection.

### STAGE 4: Linear Algebra & Geometric Reduction (PCA, SVD, LDA, Clustering)
- **Purpose:** Project high-dimensional representations onto low-dimensional principal and discriminant subspaces, and discover latent natural clusters without supervisory labels.
- **Features:**
  - **3D PCA Projection Canvas:** 64 papers plotted in 3D coordinate space colored by category, with mean centroid $\boldsymbol{\mu}$ and principal component vectors.
  - **Scree Plot & Explained Variance:** Cumulative variance ratio $R^2$ curve.
  - **Covariance Matrix Heatmap:** $\mathbf{\Sigma} = \frac{1}{n-1} X_c^T X_c$.
  - **SVD Rank-$k$ Simulator:** Toggle rank $k \in \{1, 2, 3\}$. Watch points snap to the 1D line ($k=1$), 2D plane ($k=2$), and original coordinates ($k=3$), tracking singular values $\sigma_i$ and Frobenius reconstruction error $\|X - \hat{X}_k\|_F$.
  - **Multiclass Fisher LDA:** Side-by-side interactive heatmaps of Within-Class Scatter Matrix ($S_W$) and Between-Class Scatter Matrix ($S_B$), showing how class labels separate cluster centroids while minimizing internal variance.
  - **K-Means Centroid Migration Simulation:** Step slider $t = 0 \to T$ watching centroids $\boldsymbol{\mu}_k$ physically migrate step-by-step to cluster convergence, tracking WCSS (Inertia) and migration paths.
  - **The Elbow Method & Silhouette Curve:** Dual-axis plot of WCSS vs. Silhouette scores across $K \in \{2..8\}$, visually demonstrating the natural 4-cluster structure.
  - **Hierarchical Agglomerative Clustering (HAC) & Dendrogram:** Dendrogram of landmark research papers with Ward, Complete, and Average linkage criteria.

### STAGE 5: Supervised Classifier Arena & Evaluation Suite
- **Purpose:** Compare linear, tree-based, non-parametric, probabilistic, and neural decision boundaries on the same feature space, accompanied by rigorous error diagnostics.
- **Features:**
  - **SVM Margin Explorer:** Separating hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$, margin gutters $\mathbf{w}^T \mathbf{x} + b = \pm 1$, glowing support vectors, and a reactive regularization slider $C \in [0.01, 20.0]$ controlling margin width $\gamma = \frac{2}{\|\mathbf{w}\|_2}$.
  - **Decision Tree Split Math:** Step-by-step Gini Impurity and Entropy calculation for binary coordinate thresholds.
  - **KNN Metric Neighborhood:** Interactive metric neighborhood ball around a test paper showing Euclidean distances and majority class voting.
  - **Naive Bayes Probabilities:** Class prior $P(y)$ and likelihood products $\prod P(x_i \mid y)$ computing exact posterior distributions.
  - **2-Layer Multi-Layer Perceptron (MLP) Inspector:** Interactive network graph lighting up input nodes $(x_1, x_2) \to$ hidden units (4 ReLUs) $\to$ output nodes (Softmax probabilities for 4 domains), with step-by-step plugged-in algebraic derivations.
  - **Interactive 4x4 Confusion Matrix:** Annotated heatmap with clickable/selectable off-diagonal cells to trigger the **"Inspect Failure"** deep-dive, isolating the exact misclassified papers and their cross-disciplinary vocabulary overlap.
  - **Classification Report HUD:** Scratch NumPy calculation of per-class and macro-averaged Precision, Recall, and F1-Scores.
  - **Bias-Variance / Overfitting Slider:** Vary Decision Tree `max_depth` from 1 to 15 to watch training accuracy climb to 100% while test accuracy drops, rendering the U-shaped generalization curve.

### STAGE 6: Model MRI & Failure Debugger (Influence Functions & Evidence Shift)
- **Purpose:** Diagnose model predictions, trace errors back to influential training instances, and analyze distribution shift.
- **Features:**
  - **Koh & Liang Empirical Influence Functions:** Calculates sample influence $\mathcal{I}_{\text{up, loss}}(z)$ to identify which training papers most strongly pull or push the decision boundary.
  - **Top Culprits & Helpers:** Pinpoint the most influential training abstracts driving a misclassification.
  - **Counterfactual Retraining:** Toggle the removal of top culprit papers and watch the decision boundary snap to its new position.
  - **Evidence Under Shift (Lesson 1 Fixture):** One-click preset "Hide new papers, refresh IDF" demonstrating how diluting the rarity of a key term causes an untouched paper (`DOC-03`) to fall out of the top-five shortlist without changing its own text.

---

## 4. Live Presentation & Classroom Verification Polish

### 🎓 1-Click Professor Demo Mode
A single click on **"Start Professor Tour"** in the sidebar launches a guided 5-stop interactive walkthrough designed for classroom presentation:
1. **Stop 1 (Linguistics Lab):** Sample abstract regex stripping & Porter Stemmer vs. WordNet Lemmatizer divergence.
2. **Stop 2 (Linear Algebra):** 3D PCA plot with orthogonal projection lines, covariance matrix, and SVD rank-$k$ snapping.
3. **Stop 3 (Supervised Linear Geometry):** Fisher LDA $S_W$ and $S_B$ scatter matrices pushing classes apart.
4. **Stop 4 (Classification & Regularization):** SVM margin gutters, glowing support vectors, and reactive $C$ slider.
5. **Stop 5 (Diagnostic Failure Analysis):** Model MRI isolating the culprit training paper that caused a misclassification.

### 🔬 First-Principles Math Verification HUD
A collapsible drawer at the bottom of the application runs automated machine-precision audits comparing scratch NumPy algorithms against scikit-learn references:
- **TF-IDF Matrix Norm $L_\infty$:** Difference $\le 1.67 \times 10^{-16}$ (Threshold: $1 \times 10^{-6}$)
- **PCA Eigenvalues Spectrum:** Difference $\le 5.20 \times 10^{-18}$ (Threshold: $1 \times 10^{-5}$)
- **Sample Covariance Matrix:** Difference $\le 4.34 \times 10^{-19}$ (Threshold: $1 \times 10^{-12}$)
- **Fisher LDA Scatter Conservation ($S_W + S_B = S_{\text{Total}}$):** Difference $\le 3.89 \times 10^{-16}$ (Threshold: $1 \times 10^{-10}$)

---

## 5. Syllabus Mapping to Course Materials

| GlassBox Feature | Course Module & Material Consulted | Direct Theoretical Alignment |
| :--- | :--- | :--- |
| **Token Microscope & Vector Strips** | `UCS551 ADS Lab: 2.Handling Unstructured Textual Data/6.Lab Session 3(b)` & `UML501: 4_Data Pre-Processing I` | Term frequencies, Laplace smoothed $IDF$, and $L_2$ Euclidean normalization. |
| **3D PCA & SVD Rank-$k$** | `UCS551 ADS Theory: 6.Data Pre-Processing-IV.pdf` & `Lab: PCA_Scratch_Inbuilt.ipynb` | Sample covariance $\Sigma = \frac{1}{n-1} X_c^T X_c$, symmetric Jacobi eigensolution, and Eckart-Young-Mirsky truncated SVD approximation. |
| **Multiclass LDA Scatter** | `UCS551 ADS Theory: 8.Data Pre-Processing-VI.pdf` & `Lab: LDA_Scratch_Inbuilt.ipynb` | Within-class scatter $S_W$, between-class scatter $S_B$, and Fisher generalized eigenproblem $(S_W^{-1} S_B) w = \lambda w$. |
| **K-Means & HAC Dendrogram** | `UML501: 2025_Syllabus.pdf (Unsupervised Learning)` & `K-Means/Hierarchical Clustering` | Centroid migration, WCSS Elbow method, Silhouette coefficients, and Ward/Complete/Average hierarchical linkages. |
| **SVM Margin Gutters** | `UML501: 2025_Syllabus.pdf` & `Classification Evaluation Metrics` | Linear separating hyperplanes, support vectors on margin boundaries, and slack variable penalty $C$. |
| **Neural Network (MLP) Inspector** | `UML501: 2025_Syllabus.pdf (Artificial Neural Networks)` & `Multi-Layer Perceptrons` | 2-layer forward propagation, ReLU activations, output logits, and Softmax probability distributions. |
| **Evaluation Suite & Overfitting** | `UML501: Classification Evaluation Metrics` & `Model Selection & Bias-Variance` | $4 \times 4$ Confusion Matrix, scratch Precision/Recall/Macro-F1, and Decision Tree depth overfitting curve. |
| **Model MRI Influence Functions** | `Advanced ML Diagnostics & Data Attribution (Koh & Liang, ICML 2017)` | Empirical influence functions $\mathcal{I}_{\text{up, loss}}(z) = - \nabla_\theta L^T H^{-1} \nabla_\theta L$ and counterfactual boundary recovery. |

---

## 6. Verification Suite (27 Unit Tests)

27 unit tests verify numerical accuracy, scikit-learn reference agreement, and mathematical invariant properties:
- `test_classifiers.py` (4 tests): arXiv corpus balance (16 per category), SVD rank-$k$ monotonicity, LDA scatter matrix symmetry/eigenvalues, and SVM margin width $\gamma = 2 / \|\mathbf{w}\|$.
- `test_math.py` (8 tests): Hand calculations, scikit-learn $TF$-$IDF$ agreement within $10^{-5}$, Jacobi eigenpair residuals $< 10^{-10}$, and projection orthogonality.
- `test_pipeline.py` (12 tests): Regex cleaning, stemmer vs lemmatizer divergence, vocabulary contraction funnel, Decision Tree Gini split, KNN neighborhood ball, influence functions, K-Means centroid migration & WCSS monotonicity, HAC linkage matrix, MLP forward pass and activations, confusion matrix and macro metrics, bias-variance curve, and the automated scratch vs scikit-learn audit.
- `test_shift.py` (3 tests): All 4 Evidence Under Shift combinations and target paper demotion.
