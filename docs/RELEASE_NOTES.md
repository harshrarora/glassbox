# Final scientific and interface audit

## Corrections made

| Finding | Final behavior | Evidence |
| :--- | :--- | :--- |
| K-means was called unsupervised while operating on LDA | PCA is the default; label-informed LDA is explicitly selectable and disclosed | UI interaction test and browser inspection |
| A fixed explanation asserted K=4 was optimal | The displayed peak comes from the current measured silhouette sweep | No asserted cluster count |
| A singleton received silhouette 1 | Singleton contribution is 0; invalid partitions display undefined scores | scikit-learn silhouette comparison |
| K-means final labels/inertia could describe the previous centers | Final assignments and inertia are recomputed at the returned centers | One-update-budget regression; monotone history check |
| Integer features truncated centroid means | Floating-point arithmetic preserves fractional centers | Integer-input regression |
| Rank-k SVD used raw data beside centered PCA axes | SVD uses centered data and adds the mean back | Reconstruction equality and residual-energy identity |
| Jacobi convergence depended on absolute matrix scale | Normalize scale, check the final allowed rotation, retain small eigenvalues | Tests at scales 10⁻¹⁸, 1 and 10¹⁸ |
| Vertical SVM separators were rendered as y=0 | Each boundary and margin is parameterized by the numerically stable coordinate | Level-set residuals for all three lines |
| Tree feature labels reversed horizontal/vertical boundaries | Labels match the actual plotted cut | Classifier interaction checks |
| KNN circles could be visually distorted | Equal axis scales preserve Euclidean geometry | Plot layout inspection |
| Regex diff highlighted punctuation but missed whole URLs/formulas | Removed spans are highlighted and HTML is escaped | Diff regression |
| Metrics could silently drop unknown labels or unequal lists | Invalid evaluation input raises an explicit error | Input-contract regression |

The final pass also adds live matched-start K-means comparison, labels training-only displays clearly, reports an unmatched Naive Bayes query as prior-only inference, and uses current Streamlit width arguments.

## Reference protocol

K-means uses shared initial centers, Lloyd updates, one start, a 100-update limit and zero movement tolerance. Center residuals use optimal assignment; adjusted Rand index ignores cluster numbering. Empty-cluster strategies remain different and are disclosed. Final nearest-center reassignment follows the distinction documented in [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Silhouette is meaningful only for 2 through n−1 occupied clusters; see [the reference definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html).

The default three-term teaching projection is itself a useful counterexample: the four sampled starting rows have identical coordinates, so the first assignment occupies only one cluster. Scratch ends with three occupied clusters, while the reference relocates empty centers. The observed adjusted Rand index is approximately 0.206 and inertia difference 0.0653. This is retained as a visible, documented implementation-boundary example; it is not reported as successful parity. The nondegenerate shared-initialization regression does establish agreement on its declared fixture.

The generalized LDA test checks the regularized eigenproblem against SciPy rather than claiming parity with a differently configured scikit-learn solver. The MLP test checks backpropagation against a finite-difference loss gradient away from ReLU kinks. Existing influence tests compare the analytical local derivative with actual infinitesimal upweighting, then separately test finite leave-one-out retraining.

## Verification boundary

89 tests passed locally on Python 3.12.10. See the saved JUnit report and tested direct dependencies. A passing finite test suite is not a proof for all inputs, an external scientific review, or evidence of research impact. The product is packaged as an inspectable educational laboratory with stated assumptions.
