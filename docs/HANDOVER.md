# GlassBox final handover

Release prepared on 24 November 2026. The six-stage playground is the finished product structure. This document is the operating guide, not a redesign backlog.

## Start

From the repository root, using the existing environment:

```powershell
python -m streamlit run glassbox/app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. No `PYTHONPATH` assignment is needed. The Windows launchers resolve their own directory.

For a separate Python 3.12 environment matching the validated direct dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-tested.txt
.\.venv\Scripts\python -m streamlit run glassbox/app.py
```

`requirements-tested.txt` records direct dependency versions. Transitive packages and optional NLTK resources are not locked. The interface identifies the morphology backend; exported experiment records include it. No NLTK download is required.

After replacing source files, stop the app with Ctrl+C and launch it again so imported modules are refreshed. If a port is occupied, add `--server.port 8502`. The final development preview uses 8502; the normal launcher uses Streamlit's default port.

## Verify

```powershell
python -m pytest glassbox/tests -q
python -m compileall -q glassbox
```

Final local result: **89 tests passed**, including 25 Streamlit AppTest cases. The saved [JUnit report](validation-results.xml) records the run. CI is configured for Python 3.11 and 3.12; a local pass is not a claim that hosted CI has run.

Numerical coverage includes scale-relative Jacobi eigenpairs, PCA/SVD reconstruction agreement, generalized LDA residuals, leakage isolation, influence finite differences, actual logistic retraining, singleton silhouettes, K-means final-state consistency, vertical SVM margins, MLP finite-difference gradients, and matched reference outputs. AppTest covers six-stage navigation, all classifier branches, HAC, reference views, persistent preprocessing and immediate interventions.

Browser verification complements these tests: desktop and narrow layouts, live geometry controls, clustering/reference displays, and diagnosis are checked against the running application. Screenshots are saved beside this document. AppTest alone does not establish visual quality.

The browser-downloaded [example influence record](example-influence-record.json) records a measured loss change of approximately −0.030251, versus −0.031037 classically and −0.030197 with leave-one-out normalization. It includes the exact parameters, code fingerprint and observed environment.

## A five-minute demonstration

| Time | Action | Point to explain |
| :--- | :--- | :--- |
| 0:00 | Corpus: inspect a record and its length | These are teaching summaries with explicit provenance limits. |
| 0:30 | Preprocess: toggle stopwords or morphology | The representation discards information by design. |
| 1:00 | Vectorize: select a term, then Magnify | Substitute real counts, IDF and norm into one coordinate. |
| 1:45 | Compare implementations | Shared inputs and conventions make residuals interpretable. |
| 2:15 | Geometry: rank 1 → 2 → 3 | Centered reconstruction, retained energy and error agree. |
| 3:00 | Geometry: Failure spotlight | A principal axis can move much more than the leading plane. |
| 3:45 | Diagnose: change the removed paper | A local influence estimate is a hypothesis tested by retraining. |
| 4:30 | Expand normalization check; export record | Explain the original n versus n−1 loss normalization and retain the evidence. |

For an evaluation-focused interview, replace the geometry spotlight with Classify → Failure spotlight. Random labels make leakage visible without relying on a real classification signal. Do not select only the most dramatic seed and present it as typical performance.

## Accurate portfolio copy

> Built GlassBox, a six-stage interactive ML playground and failure laboratory. Implemented inspectable NumPy algorithms for text geometry, clustering, neural computation and logistic influence; aligned reference comparisons; and tested leakage, eigenspace instability and attribution through controlled interventions. Validated numerical and UI contracts with 89 automated tests.

Live comparisons cover TF-IDF, PCA, K-means, KNN and multinomial Bayes. SVM and fitted decision trees are reference estimators. LDA uses NumPy's symmetric eigensolver; SVD uses NumPy decomposition. Influence is a first-order approximation computed with linear solves, not an exact closed-form deletion effect.

## Package

```powershell
python scripts/package_release.py
```

The script creates `release/GlassBox-v1.0.zip` and its SHA-256 sidecar. The archive contains source, fixtures, tests, dependency specifications, CI configuration, launchers and documentation. Its internal manifest hashes every included file. It excludes Git history, caches, virtual environments and prior build outputs. It does not publish or upload anything.

The original `ML_Project_Research_Report.pdf` remains in the checkout as a historical report. The current implementation contract is this handover and the root README; the old report is not included in the release bundle. No new license or ownership claim has been assigned on the author's behalf.

## Scope at handover

- The 64 texts and paper metadata are not a verified arXiv benchmark.
- Validation scores are exploratory; there is no untouched benchmark test set.
- MRI explains a frozen, two-feature, regularized logistic classifier.
- A stable leading plane is approximately stable under the constructed perturbation, not invariant in general.
- Reference agreement is measured on matched inputs and declared conventions. Empty clusters and distance ties have explicit boundaries.

These are properties of the experiment design, not unfinished product features.
