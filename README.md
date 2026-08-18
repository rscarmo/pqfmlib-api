# PQFMLib

PQFMLib is a Python library for Projected Quantum Feature Maps (PQFMs) for
tabular machine learning. It builds quantum circuits from numerical datasets,
executes them with Qiskit/Aer or IBM Quantum Runtime, and exports projected
expectation values as classical feature matrices that can be used by standard
machine-learning pipelines.

The central research object in this repository is `XYZProjectiveQFM`: a
feature-map Hamiltonian that combines ideas from CD-Ising encodings and
Heisenberg-style multi-axis interactions. The library also includes
`CDIsingProjectiveQFM` and `HeisenbergProjectiveQFM` as comparison maps.

## What The Library Does

PQFMLib turns a numeric CSV dataset into quantum features through this workflow:

1. Load a tabular dataset whose last column is the target.
2. Estimate pairwise feature relations with a normalized mutual-information
   matrix $J$.
3. Split features into one or more encoding blocks/layers and axis.
4. Map features to qubits, optionally using QPU connectivity and edge quality.
5. Build a parametrized Hamiltonian circuit.
6. Measure one-local and two-local Pauli observables.
7. Save the resulting projected quantum features to CSV/NPY.

The library supports ideal Aer simulation, MPS simulation, fake-backend
simulation from IBM backends, resource estimation, and real IBM Quantum execution using IBM Runtime.

## Installation

```bash
pip install -e .
```

or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Data Format

Datasets are loaded from:

```text
<data_dir>/<name_file>.csv
```

All columns must be numeric and finite, with no missing values. The last column
is treated as the target $y$; all previous columns are encoded as input
features.

PQFMLib assumes that the dataset has already been preprocessed. In particular,
features should be normalized before running the quantum feature maps. A common
choice is standard-score normalization:

```math
h_i = \bar{x}_{f(i)},
\qquad
\bar{x}_f = \frac{x_f - \mu_f}{w_f},
```

where $\mu_f$ is the feature mean and $w_f$ is the feature standard deviation.

## Quick Example

```python
from pqfmlib import XYZProjectiveQFM

qfm = XYZProjectiveQFM(
    name_file="my_dataset",
    data_dir="./data",
    output_root="./results",
    ideal=True,
    q_enc=4,
    features_per_qubit=3,
    axes=("x", "y", "z"),
    encoding_mode="multi_axis",
    keep_diagonal_terms=True,
    keep_cross_terms=True,
    measure_cross_observables=True,
)

result = qfm.run()
print(result["Xq_all_raw"].shape)
print(result["csv_path"])
```

Runnable examples using the `Toxicity_preprocessed_shuffled` dataset are
available in `examples/`.

## Transformer API

All three maps also support an in-memory transformer workflow for simulations:

```python
from pqfmlib import XYZProjectiveQFM

qfm = XYZProjectiveQFM(
    name_file="cross_validation",
    output_root="./results",
    simulation=True,
    ideal=True,
    q_enc=4,
    features_per_qubit=2,
    axes=("x", "y"),
)

qfm.fit(X_train)
Xq_train = qfm.transform(X_train)
Xq_val = qfm.transform(X_val)
Xq_test = qfm.transform(X_test)
```

`fit_transform(X_train)` is equivalent to `fit(X_train)` followed by
`transform(X_train)`:

```python
Xq_train = qfm.fit_transform(X_train)
```

The transformer API follows these rules:

- `fit(X, y=None)` returns the fitted map. The optional `y` is accepted for
  scikit-learn-style composition but is not used by the quantum feature maps.
- `transform(X)` returns a NumPy feature matrix and does not save CSV or NPY
  feature files. `fit()` may still save structural artifacts such as blocks or
  physical nodes through the existing reproducibility mechanisms.
- The number of input features must match the training matrix. When training
  uses a pandas DataFrame, later DataFrames must use the same columns in the
  same order.
- Circuit-affecting configuration is frozen by `fit()`. Change the
  configuration and call `fit()` again instead of transforming with a circuit
  built for different settings.
- The transformer API currently requires `simulation=True`. Real-QPU
  submission remains available through the backward-compatible `run()` API.

For CD-Ising and XYZ, `fit()` computes mutual information, feature blocks,
feature assignment, interactions, physical nodes, and circuit structure using
only `X_train`. Validation and test matrices never recompute those quantities.
Their values are used only to construct the parameter matrix for their own
`transform()` call.

Heisenberg does not use mutual information. Its `fit()` freezes only the state
that depends on the training feature dimension, including the number of
blocks, parameter slots, interaction scale, layout, circuit, and observables.
Each `transform(X)` applies the existing scaling and padding to the values in
that specific `X`.

The legacy file-based workflow is unchanged:

```python
result = qfm.run()
```

It still loads `<data_dir>/<name_file>.csv`, treats the last column as the
target, saves feature artifacts, and preserves real-QPU submission behavior.

## Execution Modes

PQFMLib separates the execution mode from the hardware topology used to build
the feature map:

- `ideal=True`: runs an ideal Aer simulation. The encoded qubits are treated as
  fully connected when the map allows it, without enforcing a specific IBM QPU
  coupling map.
- `ideal=False` and `simulation=True`: simulates the circuit while using the
  structure of the selected `ibm_qpu`. For example, a Heron r2 backend uses its
  heavy-hex topology, so the available two-qubit terms are limited by the QPU
  coupling map even though execution is simulated. This mode is also useful for
  MPS simulation with `mps=True`.
- `ideal=False`, `simulation=True`, and `fakebackend=True`: simulates from the
  selected IBM backend model through `AerSimulator.from_backend`, including the
  backend topology and fake-backend behavior.
- `ideal=False` and `simulation=False`: submits the job to the selected IBM
  Quantum backend through Runtime.

For `ideal=False` transformer fits, the physical subgraph, logical edges,
feature assignment, and prepared circuit are selected once and reused by every
`transform()` call. In CD-Ising and XYZ, physical-node selection is a greedy
low-error step; the genetic algorithm then assigns features to logical slots
on that selected subgraph. With `use_edge_error=True`, its fitness combines
mutual information with physical-edge quality.

When `fakebackend=False`, the simulator executes the logical circuit, while its
available interaction edges and feature assignment still come from the chosen
QPU topology. With `fakebackend=True`, the prepared circuit is also transpiled
against the backend-derived Aer target, and the resulting layout is reused.

### Fixed fitted state

The existing reproducibility options can initialize transformer state:

- `use_fixed_blocks=True` loads the saved feature assignment and embedded
  interactions instead of recomputing mutual information.
- `use_fixed_phys_nodes=True` loads and validates the saved physical nodes.
- `fixed_circuit_file_name` loads a QPY circuit during `fit()` and reuses that
  same circuit for every `transform()` call.

For full reproduction of a hardware-aware fitted state, combine fixed blocks,
fixed physical nodes, and a fixed circuit. Any component not fixed explicitly
is selected again during the next `fit()`.

Fixed QPY circuits must expose exactly the parameter names expected by the map
configuration fitted from the training feature dimension. An incompatible
circuit is rejected during `fit()` rather than failing during a later
transformation. This validates the parameter interface; users remain
responsible for supplying a circuit produced for the intended Hamiltonian and
backend configuration.

## Hamiltonian Maps

### CD-Ising PQFM

`CDIsingProjectiveQFM` is a counterdiabatic-inspired Ising-glass feature map
based on the approach presented in [1]. In the present implementation, it is extended to support multi-feature
encoding per qubit by increasing the circuit depth. Each circuit block, or
layer, encodes one feature per qubit, allowing multiple features to be assigned
sequentially to the same qubit across different layers.

For each feature block, PQFMLib constructs data-dependent local fields $h_i$
and couplings $J_{ij}$ from the input values and the mutual-information matrix.
The local fields are interpreted as normalized feature values.

A useful way to view the underlying Ising problem Hamiltonian is:

```math
H_{\mathrm{Ising}}(x)
= \sum_i h_i(x) Z_i
+ \sum_{(i,j)} J_{ij} Z_i Z_j .
```

The implemented circuit applies a first-order counterdiabatic term:

```math
H_{\mathrm{CD}}(x,t) =
-2 \dot{\lambda}(t) \alpha_1(t)
[
\sum_i h_i(x)Y_i +
\sum_{i \lt j} J_{ij}(Y_i Z_j + Z_i Y_j)
].
```

The first-order CD coefficient is:

```math
\alpha_1(t) =
-\frac{\sum_i h_i^2 + \sum_{i \lt j} J_{ij}^2}{4R(t)}.
```

with:

```math
R(t) =
(1-\lambda(t))^2
(\sum_i h_i^2 + 4\sum_{i\ne j} J_{ij}^2)
+ \lambda(t)^2
(
\sum_i h_i^4 +
\sum_{i\ne j} J_{ij}^4 +
6\sum_{i\ne j} h_i^2J_{ij}^2 +
6\sum_{i \lt j \lt k}(J_{ij}^2J_{ik}^2 + J_{ij}^2J_{jk}^2 + J_{ik}^2J_{jk}^2)
).
```

In the implementation, the schedule is represented by $s(t)$ and
$\dot{s}(t)$, which play the role of $\lambda(t)$ and $\dot{\lambda}(t)$.
This map is useful as a physically motivated Ising baseline: it encodes
features locally and uses feature-feature relations to activate two-qubit CD
terms on available edges.

### Heisenberg PQFM

`HeisenbergProjectiveQFM` follows the Heisenberg-style projected quantum feature
maps used in [2,3]. It prepares one random single-qubit unitary per qubit and
then applies repeated even/odd nearest-neighbor chain layers. Each normalized
scalar feature drives an isotropic two-qubit interaction:

```math
H_{\mathrm{Heisenberg}}(x)
= \sum_{i} J_i
(
X_i X_{i+1}
+ Y_i Y_{i+1}
+ Z_i Z_{i+1}
) .
```

Here $J_i$ denotes the angle assigned to the chain edge $(i,i+1)$. Using the
normalized feature convention defined in **Data Format**, the Heisenberg map
scales each assigned feature as:

```math
J_i = 2\pi \tanh\left(\frac{\bar{x}_{f(i)}}{3}\right).
```

The default observables are one-local $Z$, $X$, and $Y$ for every qubit, with an
option to include two-local diagonal observables $ZZ$, $XX$, and $YY$.

### XYZ PQFM

`XYZProjectiveQFM` is the main map in PQFMLib. It generalizes the Ising idea
from one Pauli axis to several Pauli axes and can also include cross-axis
interactions. A compact way to write the full Hamiltonian is:

```math
H_f
= \sum_i
(
h_i^{(x)} X_i
+ h_i^{(y)} Y_i
+ h_i^{(z)} Z_i
)
+ \sum_{i \lt j}
[J_{ij}^{(xx)} X_i X_j
+ J_{ij}^{(yy)} Y_i Y_j
+ J_{ij}^{(zz)} Z_i Z_j
+ J_{ij}^{(xy)}(X_iY_j + Y_iX_j)
+ J_{ij}^{(xz)}(X_iZ_j + Z_iX_j)
+ J_{ij}^{(yz)}(Y_iZ_j + Z_iY_j)].
```

The local fields $h_i^{(x)}$, $h_i^{(y)}$, and $h_i^{(z)}$ are normalized
axis-encoded features:

```math
h_i^{(a)} = \bar{x}_{f_a(i)},
\qquad
\bar{x}_{f_a(i)}
= \frac{x_{f_a(i)} - \mu_{f_a(i)}}{w_{f_a(i)}} ,
\qquad
a \in \{x,y,z\}.
```

The pairwise couplings between qubits $i$ and $j$ form a full axis-correlation
matrix:

```math
J_{ij}
=
\begin{pmatrix}
J_{ij}^{(xx)} & J_{ij}^{(xy)} & J_{ij}^{(xz)} \\
J_{ij}^{(yx)} & J_{ij}^{(yy)} & J_{ij}^{(yz)} \\
J_{ij}^{(zx)} & J_{ij}^{(zy)} & J_{ij}^{(zz)}
\end{pmatrix}.
```

With `keep_diagonal_terms=True`, the map includes the three Ising-like channels
$XX$, $YY$, and $ZZ$. With `keep_cross_terms=True`, it also includes cross-axis
correlations such as $XY$, $XZ$, and $YZ$. In feature-map terms, each qubit has
a local feature vector:

```math
\mathbf{x}_i
=
(
x_i^{(x)},
x_i^{(y)},
x_i^{(z)}
).
```

and each pair of qubits can carry a full correlation matrix between their
axis-encoded features.

## Axis Encoding Modes

`XYZProjectiveQFM` has two encoding modes.

### Multi-Axis Encoding

Use:

```python
encoding_mode="multi_axis"
features_per_qubit=3
axes=("x", "y", "z")
```

In multi-axis mode, each `(qubit, axis)` slot can receive a different tabular
feature:

```math
\begin{aligned}
\text{qubit } i,\ X\text{ axis} &\mapsto f_X(i), \\
\text{qubit } i,\ Y\text{ axis} &\mapsto f_Y(i), \\
\text{qubit } i,\ Z\text{ axis} &\mapsto f_Z(i).
\end{aligned}
```

This means the capacity per layer is:

```math
C_{\text{multi-axis}} = q_{\mathrm{enc}} \, |\mathrm{axes}| .
```

For example, $q_{\mathrm{enc}} = 10$ and `axes=("x", "y", "z")` can encode up to 30
features per layer. This mode is useful when the goal is to compress many
features into a small qubit register while preserving axis-specific structure.

You can also choose only two axes. For example:

```python
encoding_mode="multi_axis"
features_per_qubit=2
axes=("x", "y")
```

In this case, each qubit encodes two different features per layer, one on the
$X$ axis and one on the $Y$ axis.

### Shared-Feature Encoding

Use:

```python
encoding_mode="shared_feature"
features_per_qubit=1
axes=("x", "y", "z")
```

In shared-feature mode, each qubit receives one feature and reuses that same
feature across all selected axes:

```math
\begin{aligned}
\text{qubit } i,\ X\text{ axis} &\mapsto f(i), \\
\text{qubit } i,\ Y\text{ axis} &\mapsto f(i), \\
\text{qubit } i,\ Z\text{ axis} &\mapsto f(i).
\end{aligned}
```

The capacity per layer is:

```math
C_{\text{shared-feature}} = q_{\mathrm{enc}} .
```

This mode is useful when the experiment should compare or combine projections
of the same variable through different Pauli axes.

Shared-feature mode can also use two axes:

```python
encoding_mode="shared_feature"
features_per_qubit=1
axes=("x", "y")
```

Here each qubit still carries one feature, but that same feature is encoded on
both the $X$ and $Y$ axes. Thus, the feature is shared across two axes instead
of three.

## Layer Encoding

PQFMLib can encode more features than fit in one layer. It builds blocks from
the mutual-information matrix and stacks them as Hamiltonian layers. If a
dataset has $n_{\mathrm{features}}$ and one layer has capacity $C$, the number of blocks is
approximately:

```math
n_{\mathrm{blocks}}
= \lceil \frac{n_{\mathrm{features}}}{C} \rceil .
```

The layer mechanism can be mixed with axis encoding:

- `multi_axis` + layers: many different features per qubit per layer.
- `shared_feature` + layers: the same feature is reused across selected axes
  inside each layer, and additional features appear in later layers.
- diagonal + cross terms + layers: each layer can contain the $XX$, $YY$, and
  $ZZ$ Ising channels plus cross-axis couplings.

The physical repetition parameter $m$ controls how many Trotter steps are used
per block. Internally, PQFMLib builds a total circuit depth proportional to:

```math
m_{\mathrm{total}} = m \, n_{\mathrm{blocks}} .
```

## Important XYZ Options

```python
XYZProjectiveQFM(
    features_per_qubit=3,
    axes=("x", "y", "z"),
    encoding_mode="multi_axis",
    keep_diagonal_terms=True,
    keep_cross_terms=True,
    n_keep_terms=None,
    measure_all_zz=False,
    measure_cross_observables=False,
    use_edge_error=True,
    rho_thr=0.0,
)
```

Key options:

- `axes`: choose any unique subset of `("x", "y", "z")`.
- `encoding_mode`: choose `"multi_axis"` or `"shared_feature"`.
- `keep_diagonal_terms`: include same-axis terms such as $XX$, $YY$, $ZZ$.
- `keep_cross_terms`: include mixed-axis terms such as $XY$, $XZ$, $YZ$.
- `n_keep_terms`: keep only the strongest interaction terms in a layer.
- `measure_cross_observables`: measure cross-axis observables in addition to
  default one-local and diagonal observables.
- `use_edge_error`: when running on IBM backends, include edge quality in the
  feature-to-qubit assignment heuristic.
- `use_fixed_blocks` and `use_fixed_phys_nodes`: reproduce a previously saved
  feature assignment or physical layout.

## IBM Quantum Runtime

For real IBM Quantum execution, set a token before running:

```bash
export QISKIT_IBM_TOKEN="your-token"
```

On Windows PowerShell:

```powershell
$env:QISKIT_IBM_TOKEN="your-token"
```

Then set:

```python
ideal=False
simulation=False
ibm_qpu="ibm_kingston"
```

Generated `job_meta.json` files can be retrieved with the scripts in
`scripts/`:

- `get_job_xyz.py`
- `get_job_cd_ising.py`
- `get_job_heisenberg.py`

## References

[1] Anton Simen et al., "Digitized Counterdiabatic Quantum Feature Extraction,"
arXiv:2510.13807, 2025. https://arxiv.org/abs/2510.13807

[2] Axel Ciceri et al., "Enhanced fill probability estimates in institutional
algorithmic bond trading using statistical learning algorithms with quantum
computers," arXiv:2509.17715, 2025. https://arxiv.org/abs/2509.17715

[3] Andras Ferenczi et al., "Credit Default Prediction with Projected Quantum
Feature Models and Ensembles," arXiv:2510.01129, 2025.
https://arxiv.org/abs/2510.01129

## Repository Structure

```text
pqfmlib/
|-- examples/
|-- pqfmlib/
|   |-- core/
|   |-- hardware/
|   |-- maps/
|   |-- runners/
|   `-- utils/
|-- scripts/
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```
