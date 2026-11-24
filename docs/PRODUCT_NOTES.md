# Playground structure and verification

## Product direction

GlassBox is a reactive pipeline with an optional puzzle layer. The primary experience is manipulating an object and looking under the hood, not completing a lesson sequence or submitting predictions.

The top navigation always exposes Corpus → Preprocess → Vectorize → Geometry → Classify → Diagnose. Stage-level lenses reveal the playground, mathematical details, implementation comparisons, or a failure spotlight. Stage 3 has a dedicated coordinate microscope. Geometry retains its PCA/SVD, Fisher LDA and clustering tabs; classifier evaluation has its own lens to avoid an excessively long combined screen.

## What travels through the pipeline

Regex cleaning, stopword removal, morphology and n-gram choices persist in a dedicated configuration, independent of Streamlit's transient widget lifecycle. The main vectorizer uses this analyzer. Geometry and classifiers use its normalized TF-IDF representation. Honest classifier evaluation fits vocabulary, IDF and LDA on training rows only, with the same preprocessing recipe.

Failure spotlights use their own declared fixtures and frozen protocols so that one intervention has an interpretable effect. Their captions explain that separation. They are not silently affected by unrelated pipeline controls.

## Under the hood

- TF-IDF: select a document and term, see the active cell highlighted, then inspect count, document frequency, IDF, vector norm and normalized coordinate with real numbers substituted.
- TF-IDF comparison: same analyzer, vocabulary, smoothing and normalization for project and scikit-learn outputs.
- PCA comparison: same active three term columns, sample covariance, eigenvalue residual and a sign-invariant projector comparison.
- K-means comparison: same PCA coordinates and starting centers, measured inertia and optimally matched center residuals, plus label-independent partition agreement. Clustering defaults to PCA; supervised LDA is disclosed when selected.
- Classifier comparison: scratch KNN and multinomial Bayes versus reference estimators on an identical held-out split. KNN distance ties can lead to different neighbor orderings.
- Diagnosis: the same logistic objective produces the influence score, boundary and counterfactual. Classical and normalization-aware approximations are distinguished.

Comparisons establish agreement on specified inputs and conventions, not universal mathematical equivalence or predictive quality.

## Visual direction

Modern sans-serif type, a white/slate surface, crisp borders, indigo interface accents and consistent emerald/indigo/amber/rose data colors. Compact headings keep the experiment on screen. Technical detail is available without imposing a wall of equations. The stage strip scrolls horizontally on narrow viewports; controls remain native keyboard-operable Streamlit widgets.

There is no hero landing page before the playground, hidden workbench hierarchy, retro paper theme, compulsory quiz, confetti or invented mastery score.

## Validation approach

Numerical tests preserve the original mathematical checks and the new invariant and finite-difference tests. UI tests now assert immediate results without predictions, immediate recomputation when controls change, persistent preprocessing through navigation, direct stage access and reference-view rendering. Tests that required quiz gates were deliberately replaced because they asserted the rejected interaction design.

Real browser inspection must complement AppTest. AppTest cannot establish pixel layout, readable chart labels, responsive behavior or working browser interaction. A running server can retain old imported modules after source changes; a fresh preview process is the reliable comparison when the rendered UI does not match files on disk.

## Portfolio framing

> Built GlassBox, a reactive ML playground tracing text through preprocessing, vectorization, geometric reduction, classification and attribution. Implemented inspectable NumPy algorithms, aligned scratch/reference comparisons, and controlled failure experiments backed by invariant, finite-difference and interaction tests.

Demonstrate a causal chain: change preprocessing, inspect the resulting vector, compare the implementation, then provoke a failure under a controlled intervention. Explain both a mathematical derivation and an implementation limitation.

## Remaining scope boundaries

The curated paper metadata and text provenance still require source verification before describing the dataset as an authoritative arXiv corpus. Interactive evaluation uses validation data, not an untouched final benchmark. The influence experiment explains a two-feature logistic classifier with a frozen representation. Scientific and educational impact require evidence beyond a functioning demonstration.
