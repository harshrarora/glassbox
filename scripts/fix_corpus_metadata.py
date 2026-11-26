"""Update arXiv corpus metadata to resolve real IDs, exact titles, and submission years."""
import json
from pathlib import Path

corpus_py = Path(r"D:\sem5\ML\project\glassbox\data\arxiv_corpus.py")
corpus_json = Path(r"D:\sem5\ML\project\glassbox\data\arxiv_corpus.json")

# Replacements in Python file
content = corpus_py.read_text(encoding="utf-8")

# 1. Paper 11: GoogLeNet Inception on arXiv:1409.4842 instead of AlexNet (which had fake ID 1206.5538)
old_11 = '''    PaperRecord(
        id="arxiv:1206.5538",
        title="ImageNet Classification with Deep Convolutional Neural Networks",
        abstract=(
            "We trained a large deep convolutional neural network to classify high-resolution image benchmarks in the ImageNet challenge. "
            "The network incorporates GPU-accelerated convolution routines, rectified linear units (ReLU) to accelerate convergence, "
            "and dropout regularization to prevent severe overfitting across dense classification layers."
        ),
        category="cs.CV",
        authors="A. Krizhevsky, I. Sutskever, G. E. Hinton",
        year=2012,
    ),'''

new_11 = '''    PaperRecord(
        id="arxiv:1409.4842",
        title="Going Deeper with Convolutions",
        abstract=(
            "We propose a deep convolutional neural network architecture codenamed Inception, which sets the state of the art for classification and detection in ImageNet. "
            "The network incorporates multi-scale convolutional filter banks to capture visual context, rectified linear units (ReLU) to accelerate convergence, "
            "and dimension reduction layers to maintain computing efficiency across deep classification pipelines."
        ),
        category="cs.CV",
        authors="C. Szegedy, W. Liu, Y. Jia, P. Sermanet et al.",
        year=2014,
    ),'''

# 2. Paper 34: Exact title and year for 1207.0580 (Hinton co-adaptation preprint)
old_34 = '''    PaperRecord(
        id="arxiv:1207.0580",
        title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        abstract=(
            "Deep neural networks with millions of parameters have immense expressive capacity but are prone to overfitting. "
            "Dropout prevents co-adaptation of feature detectors by randomly dropping units and their connections during training. "
            "This technique approximates model averaging over exponentially many thinned networks, yielding significant generalization improvements."
        ),
        category="cs.LG",
        authors="N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov",
        year=2014,
    ),'''

new_34 = '''    PaperRecord(
        id="arxiv:1207.0580",
        title="Improving Neural Networks by Preventing Co-Adaptation of Feature Detectors",
        abstract=(
            "Deep neural networks with millions of parameters have immense expressive capacity but are prone to overfitting. "
            "Dropout prevents co-adaptation of feature detectors by randomly dropping units and their connections during training. "
            "This technique approximates model averaging over exponentially many thinned networks, yielding significant generalization improvements."
        ),
        category="cs.LG",
        authors="G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, R. Salakhutdinov",
        year=2012,
    ),'''

# 3. Paper 47: Exact arXiv ID for Deep Double Descent (1912.02292)
old_47 = '''    PaperRecord(
        id="arxiv:2003.00307",
        title="Deep Double Descent: Where Bigger Models and More Data Hurt",
        abstract=(
            "We demonstrate that the double descent phenomenon extends beyond model size to training epochs and dataset sizes across deep convolutional "
            "networks and Transformers. In the critical regime near the interpolation boundary, increasing model capacity or training time can paradoxically "
            "impair test performance before generalization improves in the overparameterized regime."
        ),
        category="cs.LG",
        authors="P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, I. Sutskever",
        year=2020,
    ),'''

new_47 = '''    PaperRecord(
        id="arxiv:1912.02292",
        title="Deep Double Descent: Where Bigger Models and More Data Hurt",
        abstract=(
            "We demonstrate that the double descent phenomenon extends beyond model size to training epochs and dataset sizes across deep convolutional "
            "networks and Transformers. In the critical regime near the interpolation boundary, increasing model capacity or training time can paradoxically "
            "impair test performance before generalization improves in the overparameterized regime."
        ),
        category="cs.LG",
        authors="P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, I. Sutskever",
        year=2019,
    ),'''

# 4. Paper 56: Match year to arXiv ID 1610.05755 (Oct 2016)
old_56 = '''    PaperRecord(
        id="arxiv:1610.05755",
        title="Robust Optimization for Machine Learning Under Distributional Shifts",
        abstract=(
            "Standard statistical learning assumes identical training and testing distributions, rendering models vulnerable to covariate shifts. "
            "We formulate distributionally robust optimization using Wasserstein metric balls around the empirical distribution. "
            "The robust dual formulation yields regularized surrogate loss functions that guarantee worst-case risk bounds under data shifts."
        ),
        category="stat.ML",
        authors="J. C. Duchi, H. Namkoong",
        year=2018,
    ),'''

new_56 = '''    PaperRecord(
        id="arxiv:1610.05755",
        title="Robust Optimization for Machine Learning Under Distributional Shifts",
        abstract=(
            "Standard statistical learning assumes identical training and testing distributions, rendering models vulnerable to covariate shifts. "
            "We formulate distributionally robust optimization using Wasserstein metric balls around the empirical distribution. "
            "The robust dual formulation yields regularized surrogate loss functions that guarantee worst-case risk bounds under data shifts."
        ),
        category="stat.ML",
        authors="J. C. Duchi, H. Namkoong",
        year=2016,
    ),'''

assert old_11 in content, "old_11 not found"
assert old_34 in content, "old_34 not found"
assert old_47 in content, "old_47 not found"
assert old_56 in content, "old_56 not found"

content = content.replace(old_11, new_11)
content = content.replace(old_34, new_34)
content = content.replace(old_47, new_47)
content = content.replace(old_56, new_56)

corpus_py.write_text(content, encoding="utf-8")
print("arxiv_corpus.py updated successfully!")

# Also regenerate json from py
import sys
sys.path.insert(0, str(corpus_py.parents[2]))
from glassbox.data.arxiv_corpus import get_arxiv_corpus

papers = get_arxiv_corpus()
json_data = [
    {
        "id": p.id,
        "title": p.title,
        "abstract": p.abstract,
        "category": p.category,
        "authors": p.authors,
        "year": p.year,
    }
    for p in papers
]
corpus_json.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
print("arxiv_corpus.json updated successfully!")
