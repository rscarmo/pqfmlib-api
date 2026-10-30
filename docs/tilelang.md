# TileLang simulation

TileLang executes the shared Qiskit circuits for CD-Ising, XYZ, Heisenberg and
superexchange Heisenberg on an NVIDIA CUDA GPU. It does not replace the maps'
Hamiltonians, parameter preprocessing, feature assignment or observable order.

## Scope of this implementation

| Capability | Support |
|---|---|
| FP32/FP64 statevector simulation | Yes, `precision="single"` / `"double"`, `mps=False` |
| Deterministic expectations | `expectation_method="exact"` or `"statevector"` |
| Finite-shot expectations | `expectation_method="shots"`, positive integer `shots`, local seeded RNG |
| Native connectivity | `ideal=False`, `topology="native"`, `fakebackend=False` |
| Offline native graph | `topology_file`, with optional fixed physical nodes |
| XYZ encodings/axes and diagonal/cross terms | Yes |
| CD-Ising Z products, including `k_max > 2` | Yes |
| Heisenberg local, diagonal/cross, and all-pairs readout | Yes |
| Superexchange, `J0`, `ell` | Yes |
| Compatible logical/physical QPY | Yes, within full-state GPU memory limits |
| MPS, noise, fake-backend and QPU execution | Excluded |

Defaults remain unchanged: CD-Ising/XYZ use full connectivity with `ideal=True`;
Heisenberg/superexchange default to a chain. Native connectivity requires
`ideal=False`. Selecting TileLang does not silently change these settings or
fall back to another engine. The shared default `"shots"` is supported; choose
`"exact"` or `"statevector"` explicitly for expectations without sampling.

## Isolated installation

Use a separate environment to avoid changing the GPU environment used by other
projects. The optional Python dependencies are `torch` and `tilelang`; a
compatible CUDA compiler is also required. The driver is managed by the host,
not installed or upgraded by PQFMLib.

The tested toolchain is CUDA 13.0: NVCC/NVVM/CRT 13.0.88, CCCL 13.0.85 and
runtime 13.0.96. Constraints are supplied to prevent pip from mixing compiler
components from different releases:

```bash
python3 -m venv .venv_tilelang
.venv_tilelang/bin/python -m pip install -c constraints/tilelang-cuda13.txt '.[tilelang,dev]' 'tilelang[nvcc]==0.1.15'
```

When supplying your own CUDA toolkit instead, install `.[tilelang]` and configure
that toolkit according to the [TileLang installation guide](https://tilelang.com/get_started/Installation.html).
Do not apply the compiler constraints indiscriminately to an environment used
by other applications. During development, `.venv_tilelang` was independently
copied from the existing environment and only its compiler components were
aligned; the original `.venv_gpu` was not changed by that alignment.

## Usage

```python
from pqfmlib import XYZProjectiveQFM

model = XYZProjectiveQFM(
    name_file="native_xyz",
    framework="tilelang",
    expectation_method="exact",
    precision="double",
    ideal=False,
    simulation=True,
    fakebackend=False,
    topology="native",
    topology_file="topology.json",
    q_enc=3,
    features_per_qubit=1,
    axes=("x", "y", "z"),
    encoding_mode="shared_feature",
    measure_cross_observables=True,
)
model.fit(X_train)
features = model.transform(X_test)
```

The same framework settings apply to `CDIsingProjectiveQFM`,
`HeisenbergProjectiveQFM`, and `SuperexchangeHeisenbergPQFM`. The latter still
requires one input feature per qubit. Existing `fit/transform/run`, exports,
clone, refit checks, and optional fixed-block/node files retain their semantics.

The [native example](../examples/tilelang_native.py) follows the other examples:
configure a map and call `run()` on the dataset. It simulates XYZ on the GPU
using IBM Kingston connectivity. Access to that topology uses the configured
IBM account; execution remains a local simulation. To use a saved graph
offline, add `topology_file="topology.json"` to the constructor.

```bash
.venv_tilelang/bin/python examples/tilelang_native.py
```

Set `fixed_circuit_file_name` to load a compatible QPY. Parameters are matched
by name and bound using the loaded circuit's own Parameter objects. Saved
physical circuits retain their layout, and expectations use the laid-out
operators. The automatic cone planner can exclude idle physical qubits and
unrelated gates, including ancillas outside observable support. If backward
support reaches most of the device, the required state can still exceed memory.
The executor estimates buffer memory before allocating states and limits batch
size using available memory. The diagnostic reference path always uses the
complete physical width.

## Correctness and performance

The reference executor lowers unitary circuits to one-/two-qubit gates, applies
their matrices using TileLang CUDA kernels and evaluates arbitrary Hermitian
Pauli sums. Contribution kernels support X/Y/Z products on any subset of sites;
PyTorch performs FP64 GPU reductions. Global phase has no effect on expectations.
Nonunitary instructions, classical control, unsupported modes, and malformed
parameter/observable inputs raise errors.

CD-Ising now includes higher-weight Z products in its metadata and CSV exports
when `measure_all_zz=True, k_max>2`. Existing one-/two-qubit column names are
preserved; higher terms use names such as `q3_z_0_1_2`. This also repairs the
shared Qiskit export path for that configuration.

## Exact optimizations

The default executor fuses consecutive gates when their combined support has
at most two sites. Gate order and little-endian operands are preserved. Affine
angles and fused matrices are evaluated on GPU in FP64, then stored in the
chosen precision. Fixed sparse angle/instruction metadata is uploaded once per
prepared circuit; subsequent batches transfer parameters instead of full
matrices. General nonlinear expressions retain the exact CPU Qiskit binding
path, and only the affected fused matrices use that fallback. Distinct matrices
are shared across gates/cones. Local gate positions and GPU observable masks
are also reused across batches/calls and invalidated when the prepared job changes.

Backward cones start from every site of each Pauli term, including cross-axis
and higher-weight terms. Shared/nested cones are merged by unioning selected
gates in original circuit order. The planner chooses reduced states or a full
state by estimated amplitude/gate work; this is a cost heuristic, not a change
to the feature map. Observables are reduced in groups on GPU, followed by one
host transfer per group rather than per sample/observable.

Up to 128 samples are processed per batch by default, with a memory cap. For
diagnostics, replace a fitted map's executor without changing its circuit:

```python
from pqfmlib.core.tilelang_execution import TileLangExecutor

model.backend = TileLangExecutor(precision=model.precision, seed=model.seed,
                                 strategy="auto", batch_size=32)
features = model.transform(X_test)
```

Strategies are `auto`, `full` (fused/batched full state), `cones` (forced reduced
states), and `reference` (preserved phase-2 per-row/per-observable execution).
The reference path supports independent optimization comparisons and may
require much more memory for wide physical QPYs. These strategies remain exact
computations, subject to the selected floating-point precision. The simulator
does not truncate states. All strategies also support finite-shot expectations.

## Evolution and observable kernels

The automatic path reuses compiled launchers by shape. For sufficiently large
states/batches it loads each group of two/four amplitudes once and computes all
gate outputs from that group. Smaller workloads retain the original gate
kernel: fewer threads per amplitude group can hurt GPU occupancy on tiny batches.
The current gate heuristic requires at least 4096 amplitudes per state and
65536 amplitudes across the batch; it is not a universal hardware optimum.

For states with at least 1024 amplitudes, exact Pauli reduction uses CUDA bit
population counts and 1024-element reduction tiles, accumulating in FP64.
Terms with X/Y are reduced over one representative of each pair `i, i XOR x`,
then doubled. Their real contributions are equal for Hermitian Pauli words,
including words with Y; this introduces no approximation. Diagonal Z terms
retain the full basis sum. Observable order and weighted sums are preserved.

`TileLangExecutor(kernel_backend="baseline")` retains the previous kernels and
dispatch behavior for diagnostics. `"optimized"` forces the new kernels for
testing small states, while the default `"auto"` applies the workload thresholds.
The per-row `strategy="reference"` remains independent. `last_execution_stats`
includes `kernel_backend`, `grouped_gate_calls`, and `fast_pauli_calls`.
These choices do not change the public map constructors or the simple example.
See the [kernel measurements](benchmarks/tilelang_kernels.md) for measured gains,
cases without gains, and compilation limitations.

## Precision and finite shots

`precision="single"` uses FP32 state amplitudes and gate matrices, halving their
storage relative to FP64. Matrix construction uses FP64 on the GPU for affine
angles and on the CPU for general expressions; expectation reductions also
accumulate in FP64. Returned features are
NumPy float64 in either mode. This does not undo FP32 evolution rounding.
Small-circuit comparisons against Qiskit FP64 use `atol=rtol=2e-5` for FP32 and
`1e-10` for FP64; depth and circuit size can increase errors beyond those values.
Changing a fitted map's precision requires refitting.

For sampled expectations, use for example:

```python
model.set_params(precision="single", expectation_method="shots", shots=8192, seed=42)
sampled_features = model.fit_transform(X_train)
```

Pauli terms are grouped into qubit-wise compatible measurement bases using the
same stable greedy grouping as CUDA-Q. Each group uses joint bitstrings drawn
from its rotated state's Born probabilities. Compatible terms share bitstrings,
preserving correlations and the variance of their weighted sums. X and Y bases
are rotated explicitly; identity terms are exact. Cones cover the complete joint
measurement support. Probabilities accumulate in FP64 and are normalized before
sampling; this corrects norm drift, not other evolution error.

`shots` is the number of bitstrings **per input row and measurement group**, not
a total shared over all groups. Sampling uses GPU CDFs, a local seeded Torch
generator, and chunks of at most 4096 shots. It does not change Torch's global
RNG state. Repeated calls with the same inputs, seed, strategy, batch size and
environment reproduce results. Different backends, precisions, strategies or
batch sizes need not produce identical bitstrings. For a single Pauli mean,
sampling standard deviation is `sqrt((1 - expectation**2) / shots)`; weighted
sums also depend on covariance. MPS, noise and hardware remain excluded.

The FP64 exact speedups in the phase-3 report do not measure FP32 or shots.

`last_execution_stats` reports selected width, cone count, batch size, fused
gate/matrix counts, kernel calls, buffer estimates, host matrix time and the
selected matrix backend (`gpu`, `cpu`, or `hybrid`). For diagnostic comparisons,
`TileLangExecutor(matrix_backend="cpu")` retains CPU matrix construction;
the default `"auto"` uses the GPU for affine matrices. Set
`profile=True` on the executor for synchronized evolution, reduction and
transfer stage times, plus `matrix_device_seconds` for GPU matrix preparation.
Profiling adds synchronization overhead; measure normal
execution separately. No optimized strategy is guaranteed to be faster for
every graph, depth, observable set or batch size.

## Native-graph benchmarks

Run benchmarks separately from other GPU work:

```bash
.venv_tilelang/bin/python scripts/benchmark_tilelang.py --qubits 8 12 16 --steps 1 2 --rows 16 --repeats 3 --output /tmp/tilelang-benchmark.json
```

The script compares the reference path, fused full states, automatic cones,
CUDA-Q exact GPU and Aer statevector GPU using the same prepared circuit,
parameters and observables. Every result is checked against the first engine;
the first two rows are independently checked with Qiskit CPU. It reports common
fit time, parameter preparation, backend setup, cold execution, warmed medians,
parameter-plus-execution medians and a separate profiling run. Torch peak
allocation is reported only for TileLang; it is not a whole-device memory peak.

Default graphs are synthetic heavy-hex connectivity, not live IBM calibration
snapshots. Use `--graphs chain heavy_hex` for both synthetic shapes, or
`--topology-file topology.json` for a saved device graph. `--blocks` controls
input feature count (`q_enc * blocks`); the report records actual fitted block
counts, which also depend on encoding capacity. `--encoding`, `--axes`,
`--no-cross-terms` and `--no-cross-observables` vary XYZ settings. Raw JSON/CSV
include data/topology/source hashes and all timing repetitions. A speedup of
the complete execution path must not be interpreted as a raw compiler/kernel
comparison or a guarantee for another dataset or topology.

Measured results, plots, raw data and limitations are available in the
[phase-3 native-graph report](benchmarks/tilelang_phase3.md). Its
`cold_execution_seconds` field is a first call within a shared process with
shared compiler caches, not an independently cold process measurement.

The subsequent [matrix-preparation report](benchmarks/tilelang_matrices.md)
compares the pre-change executor with GPU matrix preparation using identical
inputs, circuits and FP64 precision. It also compares the preserved CPU matrix
path within the new executor. New kernels add compilation cost on first use;
warmed execution gains do not imply a faster first call.

## Tests on an actual GPU

```bash
PQFMLIB_TEST_TILELANG_GPU=1 .venv_tilelang/bin/python -m pytest -q tests/test_tilelang_execution.py tests/test_tilelang_sampling.py
PQFMLIB_TEST_TILELANG_GPU=1 PQFMLIB_TEST_CUDAQ_GPU=1 .venv_tilelang/bin/python -m pytest -q tests/test_tilelang_execution.py tests/test_native_simulation.py tests/test_cudaq_execution.py tests/test_precision.py tests/test_superexchange_backends.py
```

Explicit opt-in makes missing CUDA access/dependencies an error, rather than a
skip. Tests compare against Qiskit, enumerate every three-qubit Pauli word,
exercise nonlinear angles, both encodings, native graphs, cross/multilocal
observables, and logical/routed physical QPYs. They also check repeated
transforms and `run()` exports. FP64 comparison tolerances are `1e-10` or tighter
for these small circuits; this is not a universal error bound.
The sampling suite checks FP32 on every map, native logical/physical QPYs,
Bell-state covariance, weighted sums, seed isolation, shot chunk remainders,
more than 64 terms, wide physical cones, and the expected shot mean/variance.
