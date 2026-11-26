"""Generate publication-grade figures for the GlassBox Machine Learning Lab Project Report."""
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import dendrogram

# Add project root to path
ROOT_DIR = Path(r"D:\sem5\ML\project")
sys.path.insert(0, str(ROOT_DIR))

FIG_DIR = ROOT_DIR / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from glassbox.data.arxiv_corpus import get_arxiv_corpus
from glassbox.core.vectorizer import TextVectorizer
from glassbox.core.geometry import fit_pca_3d
from glassbox.core.classifiers import LDASimulator, SVMSimulator
from glassbox.utils.math_engine import (
    compute_hac_linkage,
    simulate_kmeans,
    compute_influence_functions_2d,
    compute_retrained_counterfactual,
)
from glassbox.core.experiments import leakage_experiment

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
})

CATEGORY_COLORS = {
    "cs.CV": "#10b981",    # Emerald
    "cs.CL": "#6366f1",    # Indigo
    "cs.LG": "#f59e0b",    # Amber
    "stat.ML": "#f43f5e",   # Rose
}

papers = get_arxiv_corpus()
texts = [p.abstract for p in papers]
labels = [p.category for p in papers]
unique_cats = sorted(list(set(labels)))

print("Generating Figure 1: Pipeline Architecture...")
# -----------------------------------------------------------------------------
# Figure 1: Pipeline Flowchart / Architecture
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.2), dpi=300)
ax.axis("off")

stages = [
    ("Stage 1\nRaw Corpus", "64 Curated\narXiv Abstracts\n4 Balanced Domains", "#e0e7ff"),
    ("Stage 2\nPreprocessing", "Regex Sanitization\nStopword Diff\nStem vs. Lemma", "#dbeafe"),
    ("Stage 3\nVectorization", "Smoothed TF-IDF\nMatrix Sparsity\nParity Correlation", "#ccfbf1"),
    ("Stage 4\nGeometry", "Jacobi 3D PCA\nFisher LDA (Sw/Sb)\nK-Means & HAC", "#fef3c7"),
    ("Stage 5\nClassifiers", "SVM, Tree, KNN,\nBayes, 2-Layer MLP\nData Leakage Lab", "#fed7aa"),
    ("Stage 6\nModel MRI", "Koh & Liang\nInfluence Attribution\nLOO Retraining", "#fecdd3"),
]

from matplotlib.patches import FancyBboxPatch

box_width, box_height = 1.35, 1.8
for i, (title, desc, color) in enumerate(stages):
    x = i * 1.6 + 0.1
    y = 0.6
    rect = FancyBboxPatch((x, y), box_width, box_height, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor="#475569", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + box_width/2, y + box_height - 0.35, title, ha="center", va="center",
            fontweight="bold", fontsize=10, color="#0f172a")
    ax.text(x + box_width/2, y + 0.55, desc, ha="center", va="center",
            fontsize=8, color="#334155", linespacing=1.3)
    
    # Arrow to next stage
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + box_width + 0.22, y + box_height/2),
                    xytext=(x + box_width + 0.03, y + box_height/2),
                    arrowprops=dict(arrowstyle="->", color="#0f172a", lw=2, mutation_scale=15))

ax.set_xlim(-0.1, len(stages) * 1.6 + 0.1)
ax.set_ylim(0.2, 2.7)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_pipeline_architecture.png")
plt.close(fig)

print("Generating Figure 2: Corpus EDA & Abstract Lengths...")
# -----------------------------------------------------------------------------
# Figure 2: Corpus EDA & Abstract Lengths
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=300)

cat_counts = [sum(1 for p in papers if p.category == c) for c in unique_cats]
bars = ax1.bar(unique_cats, cat_counts, color=[CATEGORY_COLORS[c] for c in unique_cats], edgecolor="#1e293b", linewidth=1)
ax1.set_title("(a) Category Balance Across Curated Corpus", fontweight="bold")
ax1.set_ylabel("Number of Research Papers")
ax1.set_ylim(0, 20)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")

lengths_by_cat = {c: [len(p.abstract.split()) for p in papers if p.category == c] for c in unique_cats}
bp = ax2.boxplot([lengths_by_cat[c] for c in unique_cats], tick_labels=unique_cats, patch_artist=True)
for patch, c in zip(bp["boxes"], unique_cats):
    patch.set_facecolor(CATEGORY_COLORS[c])
    patch.set_alpha(0.7)
    patch.set_edgecolor("#1e293b")
ax2.set_title("(b) Abstract Word Count Distribution", fontweight="bold")
ax2.set_ylabel("Word Count per Abstract")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_corpus_eda.png")
plt.close(fig)

print("Generating Figure 3: PCA vs LDA Projection...")
# -----------------------------------------------------------------------------
# Figure 3: PCA vs LDA 2D Projections
# -----------------------------------------------------------------------------
vec = TextVectorizer().fit(texts)
_, _, X_all = vec.transform(texts)
terms = ["image", "language", "optimization"]
terms_idx = [vec.vocabulary.get(t, 0) for t in terms]
X_features = X_all[:, terms_idx]

pca_res = fit_pca_3d(X_features, terms)
lda_res = LDASimulator.fit(X_features, labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

# Unsupervised PCA
for c in unique_cats:
    mask = np.array([p == c for p in labels])
    ax1.scatter(pca_res.scores_2d[mask, 0], pca_res.scores_2d[mask, 1],
                color=CATEGORY_COLORS[c], label=c, s=40, edgecolors="#1e293b", alpha=0.85)
ax1.set_title("(a) Unsupervised PCA: Maximizing Total Variance", fontweight="bold")
ax1.set_xlabel(f"PC1 ({pca_res.explained_variance_ratio[0]*100:.1f}% Variance)")
ax1.set_ylabel(f"PC2 ({pca_res.explained_variance_ratio[1]*100:.1f}% Variance)")
ax1.legend(frameon=True)

# Supervised Fisher LDA
for c in unique_cats:
    mask = np.array([p == c for p in labels])
    ax2.scatter(lda_res.projected_2d[mask, 0], lda_res.projected_2d[mask, 1],
                color=CATEGORY_COLORS[c], label=c, s=40, edgecolors="#1e293b", alpha=0.85)
ax2.set_title(f"(b) Supervised Fisher LDA: Maximizing $S_B / S_W$ Class Separation", fontweight="bold")
ax2.set_xlabel(rf"Discriminant Axis 1 ($\lambda_1 = {lda_res.eigenvalues[0]:.2f}$)")
ax2.set_ylabel(rf"Discriminant Axis 2 ($\lambda_2 = {lda_res.eigenvalues[1]:.2f}$)")
ax2.legend(frameon=True)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_pca_vs_lda.png")
plt.close(fig)

print("Generating Figure 4: HAC Dendrogram & K-Means Clusters...")
# -----------------------------------------------------------------------------
# Figure 4: HAC Dendrogram & K-Means Clusters
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)

# HAC Dendrogram on landmark papers
pts_2d = lda_res.projected_2d
rep_indices = [0, 1, 16, 17, 32, 33, 48, 49]
sub_pts = pts_2d[rep_indices]
sub_labels = [f"[{papers[i].category}] {papers[i].title[:16]}..." for i in rep_indices]
Z = compute_hac_linkage(sub_pts, method="ward")

dendrogram(Z, labels=sub_labels, orientation="top", leaf_rotation=45, leaf_font_size=8, ax=ax1, color_threshold=0.7*np.max(Z[:, 2]))
ax1.set_title("(a) Hierarchical Dendrogram (Ward Linkage)", fontweight="bold")
ax1.set_ylabel("Merge Dissimilarity Distance")

# K-Means Cluster Partition
km_res = simulate_kmeans(pts_2d, k=4, max_iter=20, random_state=42)
final_labels = km_res["final_labels"]
final_centroids = km_res["final_centroids"]

for k in range(4):
    k_mask = final_labels == k
    ax2.scatter(pts_2d[k_mask, 0], pts_2d[k_mask, 1], s=35, alpha=0.75, label=f"Cluster {k+1}")
ax2.scatter(final_centroids[:, 0], final_centroids[:, 1], color="#0f172a", marker="X", s=130, label=r"Centroids $\mu_k$")
ax2.set_title("(b) K-Means Converged Partitions ($K=4$)", fontweight="bold")
ax2.set_xlabel("LDA Feature Axis 1")
ax2.set_ylabel("LDA Feature Axis 2")
ax2.legend(frameon=True)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_clustering_dendrogram.png")
plt.close(fig)

print("Generating Figure 5: Classifier Performance Comparison...")
# -----------------------------------------------------------------------------
# Figure 5: Classifier Metrics & Confusion Matrix
# -----------------------------------------------------------------------------
train_idx, test_idx = train_test_split(np.arange(len(papers)), test_size=0.3, random_state=42, stratify=labels)
y_train = np.array([labels[i] for i in train_idx])
y_test = np.array([labels[i] for i in test_idx])

# Train and test on LDA 2D
X_tr = pts_2d[train_idx]
X_te = pts_2d[test_idx]

from sklearn.svm import SVC
svm_model = SVC(kernel="linear", C=1.0).fit(X_tr, y_train)
y_pred_svm = svm_model.predict(X_te)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

tree_model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_train)
y_pred_tree = tree_model.predict(X_te)

knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_train)
y_pred_knn = knn_model.predict(X_te)

nb_model = GaussianNB().fit(X_tr, y_train)
y_pred_nb = nb_model.predict(X_te)

models = ["Linear SVM", "Decision Tree", "KNN (k=5)", "Gaussian NB"]
accuracies = [
    np.mean(y_pred_svm == y_test) * 100,
    np.mean(y_pred_tree == y_test) * 100,
    np.mean(y_pred_knn == y_test) * 100,
    np.mean(y_pred_nb == y_test) * 100,
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)

bars = ax1.bar(models, accuracies, color=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"], edgecolor="#1e293b")
ax1.set_title("(a) Test Accuracy Across Syllabus Classifiers", fontweight="bold")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(0, 105)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

# Confusion matrix for SVM
cm = confusion_matrix(y_test, y_pred_svm, labels=unique_cats)
im = ax2.imshow(cm, cmap="Blues", interpolation="nearest")
ax2.set_title("(b) SVM Confusion Matrix (Test Split)", fontweight="bold")
ax2.set_xticks(np.arange(len(unique_cats)))
ax2.set_yticks(np.arange(len(unique_cats)))
ax2.set_xticklabels(unique_cats)
ax2.set_yticklabels(unique_cats)
ax2.set_xlabel("Predicted Domain")
ax2.set_ylabel("True Ground-Truth Domain")
for i in range(len(unique_cats)):
    for j in range(len(unique_cats)):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > np.max(cm)/2 else "black", fontweight="bold")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig5_classifier_evaluation.png")
plt.close(fig)

print("Generating Figure 6: First-Principles Parity Scatter...")
# -----------------------------------------------------------------------------
# Figure 6: Scratch vs Reference Parity Scatter
# -----------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf

vec_scratch = TextVectorizer().fit(texts)
_, _, X_scratch = vec_scratch.transform(texts)

vec_ref = SkTfidf(analyzer=vec_scratch.tokenizer, vocabulary=vec_scratch.vocabulary,
                  norm="l2", smooth_idf=True, dtype=np.float64).fit(texts)
X_ref = vec_ref.transform(texts).toarray()

nz_mask = (X_scratch > 0) | (X_ref > 0)
diff = np.abs(X_scratch - X_ref)
max_err = np.max(diff)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)

# Parity plot
sample_pts = np.random.RandomState(42).choice(np.flatnonzero(nz_mask), size=min(1200, np.count_nonzero(nz_mask)), replace=False)
x_p = X_ref.ravel()[sample_pts]
y_p = X_scratch.ravel()[sample_pts]

ax1.plot([0, 0.7], [0, 0.7], "--", color="#10b981", lw=2, label="Identity Line ($y = x$)")
ax1.scatter(x_p, y_p, color="#3b82f6", s=16, alpha=0.6, label="Sampled TF-IDF Coordinates")
ax1.set_title("(a) Scratch NumPy vs. scikit-learn Parity", fontweight="bold")
ax1.set_xlabel("scikit-learn Coordinate Value")
ax1.set_ylabel("Scratch NumPy Coordinate Value")
ax1.legend(frameon=True)

# Absolute error distribution
ax2.hist(diff.ravel()[sample_pts], bins=25, color="#f43f5e", edgecolor="#1e293b", log=True)
ax2.set_title(rf"(b) Numerical Residual Distribution ($\|\Delta\|_\infty \leq {max_err:.1e}$)", fontweight="bold")
ax2.set_xlabel(r"Absolute Coordinate Error $|\Delta|$")
ax2.set_ylabel("Frequency (Log Scale)")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig6_parity_verification.png")
plt.close(fig)

print("Generating Figure 7: Data Leakage Curves...")
# -----------------------------------------------------------------------------
# Figure 7: Data Leakage Investigation
# -----------------------------------------------------------------------------
res_leak = leakage_experiment(seed=42, source="noise")
depths = res_leak["depths"]
honest_test = np.array(res_leak["honest"]["test"]) * 100
leaked_test = np.array(res_leak["leaked"]["test"]) * 100

fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=300)
ax.plot(depths, honest_test, "o-", color="#10b981", lw=2.2, label="Honest Protocol (Train-Fitted Representation)")
ax.plot(depths, leaked_test, "s--", color="#e11d48", lw=2.2, label="Leaked Protocol (Full-Corpus Supervised LDA)")
ax.axhline(25, color="#64748b", linestyle=":", label="Theoretical Chance Level (25%)")
ax.set_title("Data Leakage Simulation: Random-Label Noise Control", fontweight="bold")
ax.set_xlabel("Decision Tree Max Depth")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_ylim(10, 105)
ax.legend(frameon=True)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig7_leakage_curves.png")
plt.close(fig)

print("Generating Figure 8: Model MRI Influence & Retraining...")
# -----------------------------------------------------------------------------
# Figure 8: Model MRI Influence Functions vs Retrained Boundary
# -----------------------------------------------------------------------------
cv_cl_papers = [p for p in papers if p.category in ("cs.CV", "cs.CL")]
terms_mri = ["image", "language"]
vec_mri = TextVectorizer().fit([p.abstract for p in cv_cl_papers])
_, _, X_mri_all = vec_mri.transform([p.abstract for p in cv_cl_papers])
idx_img = vec_mri.vocabulary.get("image", 0)
idx_lang = vec_mri.vocabulary.get("language", 0)
X_mri = X_mri_all[:, [idx_img, idx_lang]] * 10.0
y_mri = np.array([1 if p.category == "cs.CV" else -1 for p in cv_cl_papers])

# Held out point
test_idx = 0
x_test = X_mri[test_idx]
y_test_pt = y_mri[test_idx]
train_mask = np.arange(len(cv_cl_papers)) != test_idx
X_train_mri = X_mri[train_mask]
y_train_mri = y_mri[train_mask]

inf_res = compute_influence_functions_2d(X_train_mri, y_train_mri, x_test, y_test_pt, l2_reg=0.1)
most_influential = int(np.argmax(inf_res["influences"]))

cf_res = compute_retrained_counterfactual(X_train_mri, y_train_mri, x_test, y_test_pt, drop_idx=most_influential, l2_reg=0.1)

fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
# Training points
for lbl, name, col in [(1, "cs.CV (Vision)", "#10b981"), (-1, "cs.CL (Language)", "#6366f1")]:
    mask = y_train_mri == lbl
    ax.scatter(X_train_mri[mask, 0], X_train_mri[mask, 1], color=col, label=name, s=35, alpha=0.6)

# Test point
ax.scatter(x_test[0], x_test[1], color="#f59e0b", marker="*", s=160, label="Test Exemplar $z_{test}$", edgecolors="#1e293b")

# Removed influential point
ax.scatter(X_train_mri[most_influential, 0], X_train_mri[most_influential, 1],
           facecolors="none", edgecolors="#e11d48", marker="o", s=140, lw=2, label="Top Positive Influence")

# Decision boundaries
theta_orig = inf_res["theta"]
theta_cf = cf_res["theta_cf"]

x_line = np.linspace(-0.2, np.max(X_train_mri[:, 0]) + 0.5, 100)
y_orig = (-theta_orig[2] - theta_orig[0] * x_line) / theta_orig[1]
y_cf = (-theta_cf[2] - theta_cf[0] * x_line) / theta_cf[1]

ax.plot(x_line, y_orig, "k--", lw=1.8, label="Original Boundary ($P=0.5$)")
ax.plot(x_line, y_cf, color="#e11d48", lw=2.0, label="Counterfactual Retrained Boundary")

ax.set_title("Model MRI: Counterfactual Leave-One-Out Boundary Shift", fontweight="bold")
ax.set_xlabel("Term Feature 'image' TF-IDF $\\times 10$")
ax.set_ylabel("Term Feature 'language' TF-IDF $\\times 10$")
ax.set_ylim(-0.2, np.max(X_train_mri[:, 1]) + 0.5)
ax.legend(frameon=True, fontsize=8)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig8_model_mri_counterfactual.png")
plt.close(fig)

print("All 8 figures successfully generated in report/figures/!")
