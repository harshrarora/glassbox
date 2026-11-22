# GlassBox — From Text to Evidence

> **A Mathematical Teaching Lab for Machine Learning & Accelerated Data Science**  
> Designed for university classroom projection and interactive lecture demonstrations.

GlassBox reveals the exact, deterministic mathematical mechanics behind modern NLP, dimensionality reduction, and retrieval systems. Rather than treating vectorization or vector retrieval as black boxes, students can pause at every transformation step, inspect exact values, and understand how changes in global corpus statistics alter local retrieval rankings.

---

## 1. Quick Start & Execution Commands

### Environment Setup
Dependencies require Python 3.10+ with `numpy`, `streamlit`, `plotly`, `pandas`, `pytest`, and `scikit-learn` (used solely for reference validation in test suites):

```bash
# Navigate to project root
cd D:/sem5/ML/project

# Install dependencies
pip install -r glassbox/requirements.txt
```

### Launch Interactive Teaching Lab
```bash
# Set PYTHONPATH and launch Streamlit
$env:PYTHONPATH = "D:\sem5\ML\project"
streamlit run glassbox/app.py
```

### Run Mathematical Verification Test Suite
```bash
$env:PYTHONPATH = "D:\sem5\ML\project"
python -m pytest glassbox/tests -v
```

---

## 2. Course Syllabus & Lab Mapping

The implementation directly incorporates the notation, formulas, and scratch implementations from the course materials:

| GlassBox Screen / Step | Course Module & Source Material Consulted | Direct Pedagogical Mapping |
| :--- | :--- | :--- |
| **Step 1: Text to Matrix** | **UCS551 ADS Lab:** `2.Handling Unstructured Textual Data/6.Lab Session 3(b)-Preprocessed Corpus to Feature Matrix_Part1-Complete.ipynb`<br>**UML501 ML:** `4_Data Pre-Processing I_copy.pdf` | Implements the exact scratch term-frequency, smoothed inverse document frequency ($IDF$), unnormalized product ($Z$), and $L_2$ Euclidean row normalization matching `TfidfVectorizer(smooth_idf=True, norm='l2')`. |
| **Step 2: 3D Geometric PCA** | **UCS551 ADS Theory:** `2.data pre-processing/6.Data Pre-Processing-IV.pdf` (Slides 9–23)<br>**UCS551 ADS Lab:** `6.1 Data preprocessing -iii/PCA_Scratch_Inbuilt.ipynb`<br>**UML501 ML:** `6_Data Pre-Processing-III.pdf` | Computes sample covariance $\Sigma = \frac{1}{n-1} X_c^T X_c$, solves the eigensystem via a custom symmetric Jacobi solver ($\det(\Sigma - \lambda I) = 0$), sorts eigenpairs, projects coordinates via $\text{scores} = X_c V_2$, and traces orthogonal residual drops. |
| **Step 3: Evidence Retrieval** | **UML501 ML:** `2025_Syllabus.pdf` (Information Retrieval & Similarity Metrics)<br>**UCS551 ADS Theory:** `Structured and Unstructured Data.pdf` | Vector space model retrieval via unit-vector dot product cosine similarity $\sum_t \hat{q}(t) \hat{d}(t)$, reconciling exact per-term products to total score, and exposing high-scoring lexical traps. |
| **Step 4: Evidence Under Shift** | **UML501 ML:** `Classification Evaluation Metrics.pptx` & Data Distribution Shift concepts | Demonstrates covariate/statistical corpus shift: freezing candidate eligibility while expanding the background corpus to prove that rarity ($IDF$) is a global relative property, not an intrinsic document feature. |

*Note on Material Audit:* All course files in `D:/sem5/ML` and `D:/Conversational_AI/Accelerated_Data_Science` were inspected and preserved intact without modification. The existing project PDF (`D:/sem5/ML/project/ML_Project_Research_Report.pdf`) remains untouched.

---

## 3. Mathematical & Numerical Conventions

GlassBox adheres strictly to educational and reproducible conventions:

### Tokenization & Vocabulary
- **Token Pattern:** `(?u)\b\w+\b` (Unicode word characters bounded by word boundaries, converted to lowercase).
- **Vocabulary Ordering:** Alphabetically sorted list of distinct tokens across training corpus, producing deterministic index mappings $0, 1, \dots, V-1$.

### Vectorization Formulas
$$c(t, d) = \text{Raw integer count of term } t \text{ in document } d$$
$$f(t, d) = \frac{c(t, d)}{\sum_{t' \in d} c(t', d)} \quad (\text{Relative term frequency with named denominator})$$
$$\text{idf}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1 \quad (\text{Natural log, smoothed with Laplace constants})$$
$$z(t, d) = c(t, d) \times \text{idf}(t) \quad (\text{Unnormalized TF-IDF weight})$$
$$x(t, d) = \frac{z(t, d)}{\|z(:, d)\|_2} = \frac{z(t, d)}{\sqrt{\sum_{t'} z(t', d)^2}} \quad (\text{L2-normalized coordinate for } \|z\|_2 > 0)$$

### Sample Covariance & Jacobi Eigensolver
$$X_c = X - \mu, \quad \mu = \frac{1}{n}\sum_{i=1}^n x_i$$
$$\Sigma = \frac{1}{n - 1} X_c^T X_c \in \mathbb{R}^{3 \times 3}$$
- **Jacobi Rotation:** Iteratively targets the maximum off-diagonal element $A_{pq}$ ($p < q$), computes rotation angle $\tau = \frac{A_{qq} - A_{pp}}{2 A_{pq}}$, $t = \frac{\text{sgn}(\tau)}{|\tau| + \sqrt{1 + \tau^2}}$, $c = \frac{1}{\sqrt{1 + t^2}}$, $s = t \cdot c$, applying Givens transformations until $\max |A_{pq}| < 10^{-12}$.
- **Sign Convention:** The coordinate with the largest absolute magnitude in each eigenvector is enforced positive ($v_{k, j} \ge 0$).
- **Projection:**
  $$\text{scores} = X_c V_2 \in \mathbb{R}^{n \times 2}, \quad \tilde{X} = \text{scores} V_2^T + \mu \in \mathbb{R}^{n \times 3}$$
  $$\text{Residual (orthogonal distance)}: \quad r_i = X_i - \tilde{X}_i, \quad r_i \cdot v_1 = 0, \quad r_i \cdot v_2 = 0$$

### Full-Vector Retrieval & Score Reconciliation
$$\text{score}(q, d) = \sum_{t \in \mathcal{V}} \hat{q}(t) \times \hat{d}(t) = \sum_{t \in \mathcal{V}} \text{contribution}(t)$$
- Stable tie-breaking: Sort key is `(-score, doc_id)`.

---

## 4. Classroom Lecture Walkthrough: The "Evidence Under Shift" Preset

Here is a 5-minute teaching script a professor can follow live in class:

### Phase 1: Establish the Baseline (Step 1 & Step 3)
1. **Open Step 1:** Point out the default teaching claim:  
   *"Synaptic consolidation during non-REM sleep stabilizes associative memory."*
2. Select **DOC-01** and term **`synaptic`**. Walk students through the Mathematical Inspector on the right: show $N=7$, $\text{DF}=4$, $\text{idf}=1.4700$, and observe how $L_2$ normalization converts raw word frequencies into unit-sphere coordinates.
3. **Move to Step 3:** View the baseline ranking.
   - Note that **DOC-03** (*"Structural Dendritic Spine Remodeling Underpins Neocortical Memory Consolidation"*) ranks **#5** with similarity `0.2573` (inside the Top 5 shortlist).
   - Click **"Reveal Fixture Expert Annotation"**. Point to **DOC-04** (Rank #3 or #4): point out that DOC-04 is a **Lexical Trap**—it scores high solely because it contains the words *synaptic*, *consolidation*, and *memory*, despite proving that awake exploration *destabilizes* memory!

### Phase 2: Formulate the Hypothesis Question (Step 4)
1. **Navigate to Step 4 (Evidence Under Shift).**
2. Ask the class:  
   > *"If we keep the candidate list strictly fixed to the original 7 papers and introduce 3 new unrelated papers only into the corpus IDF statistics, can an original paper move or drop out of the top five?"*
3. Most students assume that because candidate eligibility and document texts are identical, rankings must remain invariant.

### Phase 3: Execute the Shift Preset (One-Click Reveal)
1. Click **"⭐ Apply Teaching Preset: Hide new papers, refresh IDF"**.
2. **Observe the Result:**
   - **DOC-03** drops from **Rank #5 to Rank #6**, falling completely off the shortlist!
   - **DOC-06** rises to take the #5 spot!
3. **Inspect the Demoted Focus Paper (`DOC-03`):**
   - In the baseline corpus ($N=7$), `consolidation` appeared in only 4 papers ($\text{DF}=4, \text{IDF}=1.4700$).
   - The 3 incoming papers (studying orthopedic bone consolidation and fiscal economic consolidation) increase `DF('consolidation')` to 7 ($N=10$), diluting its $\text{IDF}$ down to `1.3185` (-10.3%).
   - Because DOC-03 relies heavily on `consolidation`, its relative coordinate shrinks. Meanwhile, competitor document DOC-06 contains query terms whose rarity was preserved (`stabilizes`, `associative`, `memory`).
   - Thus, DOC-06 surpasses DOC-03 without changing a single character in either paper.

---

## 5. Architectural Guide for Subsequent Lessons

Lesson 1 establishes modular foundations that subsequent lessons will extend:
- **Lesson 2 (Feature Selection & Rank-k SVD):** Will import `TextVectorizer` to demonstrate column masking vs feature elimination, computing compact truncated SVD $U_k \Sigma_k V_k^T$ and residual Frobenius loss.
- **Lesson 3 (Linear Discriminant Analysis):** Will reuse the centered projections from `geometry.py`, computing within-class scatter $S_W = \sum (x - \mu_c)(x - \mu_c)^T$ and between-class scatter $S_B = \sum n_c (\mu_c - \mu)(\mu_c - \mu)^T$ to contrast Fisher's discriminant axis against unsupervised PCA.
- **Lesson 4 (Model MRI):** Will train a regularized linear model on claim–abstract pairs, computing parameter influence $\mathcal{I}_{\text{up, loss}}(z)$ via Hessian vector products to attribute erroneous predictions to lexical shortcuts.
