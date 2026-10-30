# TileLang extension plan

Scope: CD-Ising, XYZ, Heisenberg, and superexchange Heisenberg simulation,
including native connectivity. MPS, noise, fake-backend execution, and QPU
execution are excluded from the TileLang extension. Existing execution paths
remain supported. Each phase requires approval before the next phase starts.

## Phase 1: topology and execution contract

Integrated on top of `b3ca41a`, including the upstream superexchange and
all-pairs readout implementation. The earlier phase-1 stash is preserved.

User-confirmed semantics:

- CD-Ising/XYZ: `ideal=True` implies complete connectivity.
- Heisenberg/superexchange: preserve the default chain; `full` remains explicit.
- Native connectivity requires `ideal=False`.
- Native noiseless simulation uses `simulation=True, fakebackend=False`.

Added keyword-only topology selection to CD-Ising/XYZ and `topology_file` to all
maps. JSON snapshots supply ordered physical edges and assignment costs without
credentials. Existing physical-subgraph selection and feature mapping algorithms
are reused. Snapshot execution is restricted to noiseless simulation. The
existing CUDA-Q adapter accepts explicit native topology without relaxing its
noise/QPU restrictions; its legacy default validation remains unchanged.

Tests cover offline/backend equivalence of blocks, logical edges, observables,
and numerical features, native `fit/transform/run`, clone/refit behavior,
configuration rejection, superexchange and saved logical QPY equivalence.
CUDA-Q native preparation is checked with a mocked executor; this does not
certify execution on an actual GPU.

## Phase 2: general exact FP64 TileLang executor

Implemented as a new general executor, separate from the specialized
Heisenberg proposal. It preserves the phase-1 topology contract.

- Support all map circuits and requested Pauli observables, including cross-axis
  and higher-weight terms. Preserve column order, metadata, and layouts.
- **Required QPY fix:** bind parameters from the loaded circuit by name;
  matching names do not imply matching Qiskit parameter identities. Test logical
  and supported physical QPYs for each map. This is implemented in
  `pqfmlib/core/tilelang_execution.py` with GPU roundtrip tests for all maps.
- Declare optional dependencies and installation instructions, with approval
  before adding/installing dependencies. Keep imports optional.
- Validate unsupported configurations early and never silently switch engines.
- Compare against Qiskit and test actual compiled GPU kernels. CPU emulation
  alone cannot validate generated GPU code.

## Phase 3: optimization and measurement

Batch samples, reuse compilation/matrices, fuse valid gate sequences, and build
exact causal cones from each observable's complete support. Preserve a full-state
path when cones are not beneficial. Benchmark equal circuits, precision, rows,
and observables over native graph sizes, blocks, depths, and axes. Measure fit,
parameter preparation, compilation/warmup, evolution, reduction, transfers,
memory, and end-to-end time separately. The specialized Heisenberg speedup is
not a guarantee for other configurations.

## Phase 4: remaining compatibility

Add FP32 with explicit tolerances and shot-based expectations with seeded,
statistically appropriate tests. Publish the tested capability matrix.

## Phase 1 operational log

Changes: topology snapshot loading/export, explicit native selection for
CD-Ising/XYZ, offline native preparation for all four public map classes, and
validation of native CUDA-Q simulation. The preparation algorithms, coupling
formulas, default connectivity, and QPY encoding checks remain unchanged.
Snapshots describe metadata only; they cannot request noisy or QPU execution.

Files modified/added: `.gitignore`, `README.md`, `pqfmlib/core/base.py`,
`pqfmlib/core/framework.py`, `pqfmlib/hardware/topology.py`,
`pqfmlib/maps/cd_ising.py`, `pqfmlib/maps/xyz.py`,
`pqfmlib/maps/heisenberg.py`, `tests/test_native_simulation.py`, and this file.
The ignore exception makes the new test eligible for version control.

Validation used the existing `.venv_gpu` environment. Initial focused checks:

```bash
.venv_gpu/bin/python -m pytest -q --disable-warnings --tb=short tests/test_native_simulation.py tests/test_framework_selection.py
```

Result: 23 passed. Then all 16 versioned test modules plus the new native
simulation module were run with `-q --disable-warnings --tb=short`, obtaining
**184 passed, 40 skipped**. This includes superexchange, all-pairs readout,
fixed QPY circuits, framework selection, precision, and existing CUDA-Q CPU
translation checks. `git diff --check` passed. Existing Qiskit/Aer deprecation
warnings were reported by the test suite. Native CUDA-Q preparation uses a
mocked executor; no new native GPU execution or performance claim is validated.

No new dependencies were added or installed, and no jobs were submitted to IBM.
No commits, pushes, or remote branch changes were made. The original stash is
retained as a backup; it is not the current implementation because it used the
superseded `ideal=True` native semantics.

Phase 1 is complete. Phase 2 remains pending approval: the general TileLang
executor, its QPY parameter-identity correction, optional dependencies, and
real GPU validation. No TileLang code was introduced in phase 1.

## Phase 2 operational log

Phase 2 was subsequently approved and is complete. Added a general FP64 CUDA
TileLang executor and kernels, framework dispatch, optional imports/dependencies,
early validation, and installation constraints. The executor binds loaded QPY
parameters by name using the loaded/lowered circuit's own Parameter objects.
Arbitrary supported unitary circuits are lowered to one-/two-qubit gates;
Hermitian Pauli sums include cross-axis and higher-weight observables. The
existing phase-1 connectivity semantics remain unchanged.

Also repaired an existing CD-Ising issue required for multilocal coverage:
`k_max>2` produced additional observables without corresponding metadata/CSV
columns. Those terms now have metadata and `q3_z_...`/`q4_z_...` column names;
one-/two-qubit names remain unchanged.

Phase-2 files modified/added: `.gitignore`, `pyproject.toml`, `README.md`,
`constraints/tilelang-cuda13.txt`, `docs/tilelang.md`, this plan,
`examples/tilelang_native.py`, `pqfmlib/core/framework.py`,
`pqfmlib/core/base.py`, `pqfmlib/core/execution.py`,
`pqfmlib/core/tilelang_execution.py`, `pqfmlib/core/_tilelang_kernels.py`,
`pqfmlib/maps/heisenberg.py`, `pqfmlib/maps/cd_ising.py`, and
`tests/test_tilelang_execution.py`.

### Environment and validation

The sandbox hid CUDA access. Outside it, the RTX 5070 was accessible and
112 existing CUDA-Q/Aer/superexchange tests passed with GPU opt-in enabled.
The initial pip compiler installation mixed NVCC 13.0 with NVVM/CRT 13.4 and
failed TileLang compilation. After the user's compatibility review, copied
the existing environment independently to `.venv_tilelang` and aligned only
that copy: NVCC/NVVM/CRT 13.0.88, CCCL 13.0.85, runtime 13.0.96, TileLang 0.1.15,
PyTorch 2.14.0, CUDA-Q 0.16.0.post1. No host driver/system CUDA changes occurred.

The original `.venv_gpu` retains the optional packages installed before
isolation, including its unaligned compiler components. It received no further
package-version changes after isolation (verified against the package snapshot).
Use `.venv_tilelang` for TileLang; the original environment is not claimed as a
validated TileLang environment.

Tests/commands:

```bash
PQFMLIB_TEST_TILELANG_GPU=1 .venv_tilelang/bin/python -m pytest -q --disable-warnings --tb=short tests/test_tilelang_execution.py
.venv_tilelang/bin/python -m pytest -q --disable-warnings --tb=short tests/test_in_memory_api.py tests/test_sklearn_integration.py
.venv_tilelang/bin/python examples/tilelang_native.py /tmp/pqfmlib-example-topology.json --map xyz
.venv_tilelang/bin/python examples/tilelang_native.py /tmp/pqfmlib-example-topology.json --map cd-ising
git diff --check
```

After the CD-Ising export correction, the final full run collected the 16
versioned test modules plus `test_native_simulation.py` and
`test_tilelang_execution.py`, with both `PQFMLIB_TEST_TILELANG_GPU=1` and
`PQFMLIB_TEST_CUDAQ_GPU=1`. **317 passed, none skipped**, including actual
TileLang, CUDA-Q, Aer GPU/MPS, precision, all four maps, native connectivity,
logical/routed QPY roundtrips, higher-weight readout and exports. Another
**7 in-memory/scikit-learn integration tests passed**. The two examples matched
Qiskit within max absolute differences of 1.44e-15 (XYZ) and 8.88e-16 (CD-Ising).
Diff checks passed. Existing Qiskit/Aer deprecation warnings remain.

### Remaining work

This executor is the correctness reference, not the optimized Heisenberg path
from the reviewed proposal. It processes full states one sample at a time;
large physical QPY widths can exceed GPU memory. No speedup is claimed.
Phase 3 proposes batching/fusion, reusable matrices, exact general causal
cones, and measured comparisons over native graphs. FP32 and sampling remain
phase 4; MPS/noise/fake-backend/QPU execution remain excluded from TileLang.
No commits or pushes were made. Stop here pending approval for phase 3.

## Phase 3 operational log

Phase 3 was approved and is complete. The default TileLang path now fuses
consecutive operations whose union has at most two sites, evaluates affine
parameters in batches (with exact Qiskit binding for nonlinear expressions),
shares matrices across gates/cones, and caches constant CPU matrices.
FP64 CUDA kernels apply gates to memory-bounded sample batches and reduce
groups of Pauli expectations. Backward cones start from every measured site;
nested supports share a union of original gates in original order. An estimated
work heuristic selects cones or fused full states. No state truncation or
physics/readout change is used. The phase-2 per-row/per-observable executor
is preserved as `strategy="reference"`; `full`, `cones`, and `auto` allow direct
comparisons. Wide physical QPYs with idle ancillas can now use small cones.

Files changed/added in this phase:

- `pqfmlib/core/tilelang_plan.py`
- `pqfmlib/core/tilelang_execution.py`
- `pqfmlib/core/_tilelang_kernels.py`
- `tests/test_tilelang_execution.py`
- `scripts/benchmark_tilelang.py`
- `README.md`, `docs/tilelang.md`, and this log
- `docs/benchmarks/tilelang_phase3*`: report, JSON/CSV data, PNG/SVG plot,
  single-row, 128-row, two-block and multi-axis measurements

Validation: 103 TileLang CPU/GPU tests passed after the initial optimization.
The final full suite, including all versioned tests, native/TileLang tests and
the existing in-memory/scikit-learn tests, passed **334 tests, none skipped**
with both GPU opt-in flags enabled. The extra cases check fusion orientation,
nonlinear parameters, partial batches, every diagnostic strategy, complete
multilocal support and a 127-position circuit whose live cone has two qubits.
`git diff --check` and compilation checks passed.

The full-test command used the 16 versioned test modules plus
`test_native_simulation.py`, `test_tilelang_execution.py`,
`test_in_memory_api.py`, and `test_sklearn_integration.py`, with
`PQFMLIB_TEST_TILELANG_GPU=1 PQFMLIB_TEST_CUDAQ_GPU=1` and
`.venv_tilelang/bin/python -m pytest -q --disable-warnings --tb=short`.

Benchmarks ran separately from GPU tests. The runner uses the same circuit,
theta and observables for each backend and checks every result, including
independent Qiskit CPU checks on up to two rows. It separates common fit,
parameter preparation, setup, first execution, warmed medians and synchronized
profiling. Torch peak allocation is explicitly not a whole-device memory peak.
First execution can benefit from shared compiler caches and is not an
independently cold process. Commands and all repetitions are in the report.

On RTX 5070, the 12-case primary grid uses synthetic heavy-hex graphs,
8/12/16 qubits, m=1/2 and 16 evaluation rows. Warmed execution speedups versus
CUDA-Q were 9.1–18.7x (CD-Ising) and 4.8–17.8x (XYZ). Extra configurations
include one row (about 1.5x/1.2x), 128 rows at 8 qubits (91x/68x), two fitted
blocks (about 11–12x), and XYZ multi-axis (about 16x). These ranges are
configuration-specific, not general framework/compiler guarantees. All 95
engine/configuration measurements passed; max CPU comparison error was
1.51e-14. The CD-Ising 16-qubit m=1 case selected two cones of max width 15,
using 16.96 MiB Torch peak allocation versus 34.09 MiB for fused full states.
Most primary cases selected full batched states. Fit and CPU parameter/matrix
preparation remain relevant to total pipeline cost.

No dependencies, driver, system CUDA or existing GPU-environment packages were
changed in this phase. No QPU jobs, commits, pushes or remote branch changes
were made. Phase 4 (FP32 and shot-based expectations) remains pending approval.

## Phase 4 operational log

Phase 4 was subsequently approved. Added real FP32 state/gate storage and
evolution, with FP64 expectation accumulation. Both precisions accept exact,
statevector and finite-shot expectations through the existing public maps,
including native connectivity and compatible logical/routed physical QPYs.
The shared default `expectation_method="shots"` now works with TileLang.

Sampling groups qubit-wise compatible Pauli terms, rotates X/Y measurement
bases, and draws joint bitstrings from normalized Born CDFs on GPU. Shared
bitstrings preserve compatible terms' covariance, including weighted Pauli
sums. Cones use the whole measurement group's support. Shots are processed
in chunks of at most 4096; integer parity reductions accumulate in int64.
The local seeded Torch generator preserves global Torch RNG state. Bitwise
repeatability requires the same strategy, batching and environment; it is not
promised across engines or precisions. `shots` is a budget per row and basis.
No FP32/shots speedup is inferred from the historical FP64 exact benchmarks.

Files modified/added in phase 4: `pqfmlib/core/base.py`, `framework.py`,
`execution.py`, `observables.py`, `cudaq_execution.py`, `tilelang_execution.py`,
`tilelang_plan.py`, `_tilelang_kernels.py`, and new `tilelang_sampling.py`;
`tests/test_tilelang_execution.py`, new `tests/test_tilelang_sampling.py`,
`.gitignore`, `README.md`, `docs/tilelang.md`, this log, and the phase-3 report's
source-provenance note/archive. CUDA-Q's existing greedy grouping was moved
to a shared helper without changing its algorithm. All five historical JSON
reports' four source hashes match `tilelang_phase3_source.tar.gz`.

New tests cover every map/precision/method's configuration and clone behavior,
invalid shots/seeds, FP32 full/native numerical comparisons and refitting,
actual FP32 tensors, logical/physical QPYs, CSV/NPY exports, all four strategies,
Bell covariance, seed isolation, expected shot mean/variance, chunk remainders,
more than 64 grouped terms, and a 127-position physical circuit with a live
two-qubit joint cone. FP32 comparisons use `atol=rtol=2e-5` on the tested small
circuits; sampling tests use analytical statistics, not equality to exact means.

Validation used `.venv_tilelang` outside the sandbox on the RTX 5070. The final
full-suite command used the same 16 versioned modules as phase 3 plus
`test_native_simulation.py`, `test_tilelang_execution.py`,
`test_tilelang_sampling.py`, `test_in_memory_api.py`, and
`test_sklearn_integration.py`, with both GPU opt-in flags enabled:

```bash
PQFMLIB_TEST_TILELANG_GPU=1 PQFMLIB_TEST_CUDAQ_GPU=1 .venv_tilelang/bin/python -m pytest -q --disable-warnings --tb=short \
  tests/test_base_lifecycle.py tests/test_cd_ising_transformer.py tests/test_cudaq_execution.py \
  tests/test_execution.py tests/test_framework_selection.py tests/test_heisenberg_all_pairs.py \
  tests/test_heisenberg_fixed_circuits.py tests/test_heisenberg_topology.py tests/test_heisenberg_transformer.py \
  tests/test_precision.py tests/test_run_compatibility.py tests/test_statevector_execution.py \
  tests/test_superexchange.py tests/test_superexchange_backends.py tests/test_xyz_shap_notebook.py \
  tests/test_xyz_transformer.py tests/test_native_simulation.py tests/test_tilelang_execution.py \
  tests/test_tilelang_sampling.py tests/test_in_memory_api.py tests/test_sklearn_integration.py
```

Result: **429 passed, none skipped**, in 174.97 seconds. The original
Qiskit/Aer deprecation warnings remain. Log:
`/tmp/pqfmlib-phase4-full-tests.log`. Compilation and `git diff --check` passed.
The original `.venv_gpu` package versions still exactly match the post-isolation
snapshot; no dependencies, driver or system CUDA were changed in phase 4.

An earlier focused run, placing TileLang before the CUDA-Q/Aer regressions,
aborted natively in CUDA-Q MPS during `test_precision.py`; it did not complete.
The full-suite pass above is a separate run and does not explain that abort.
An isolated ordering recheck, without another test suite running concurrently,
then passed **134 tests** in 42.72 seconds:

```bash
PQFMLIB_TEST_TILELANG_GPU=1 PQFMLIB_TEST_CUDAQ_GPU=1 .venv_tilelang/bin/python -m pytest -q -s --disable-warnings --tb=short tests/test_tilelang_sampling.py tests/test_precision.py
```

Log: `/tmp/pqfmlib-phase4-runtime-order.log`. The abort was not reproduced; its
root cause is unconfirmed, so a universal claim about native-runtime stability
is not warranted. The primary 429-test run and this ordering recheck both passed.

Phase 4 is complete. Remaining limits: FP32 rounding depends on size/depth;
shots introduce statistical error; state/CDF memory still grows exponentially
with the selected live width; and FP32/shots performance was not benchmarked.
MPS, noise, fake-backend and QPU execution remain excluded from TileLang.
No QPU jobs, commits, pushes or remote changes were made. All four approved
phases are integrated locally. No further implementation phase is started;
any additional scope requires a new plan and approval.

## Additional performance phase 1: GPU matrix preparation

The user approved only the first additional optimization phase. This phase is
complete: affine fused matrices are now prepared on GPU, using FP64 arithmetic
and storage in the selected precision. Sparse angle coefficients, fused-gate
instructions, local gate positions and GPU observable masks are reused until
the prepared job changes. General nonlinear QPY expressions retain exact CPU
binding for the affected fused matrices. No Hamiltonian, superexchange coupling,
topology, readout or public map/example configuration was changed.

Added `matrix_backend="auto"/"cpu"/"gpu"` to the internal executor for diagnostic
comparisons; maps automatically use the optimized path. The preserved CPU
matrix path and phase-2 reference executor remain available. New diagnostic
stats separate host and GPU matrix preparation. Memory estimates include
parameter and angle buffers. No dependencies or CUDA/driver versions changed.

Files changed in this phase:

- `pqfmlib/core/tilelang_matrices.py` (new sparse-plan packer and matrix builder)
- `pqfmlib/core/_tilelang_kernels.py` (angle and fused-matrix kernels)
- `pqfmlib/core/tilelang_execution.py` (integration, metadata caching/invalidation)
- `tests/test_tilelang_execution.py` (matrix equivalence/layout/fallback/cache tests)
- `scripts/benchmark_tilelang.py` (CPU-matrix diagnostic and source provenance)
- `README.md`, `docs/tilelang.md`, this log
- `docs/benchmarks/tilelang_matrix*` (report, raw JSON/CSV, before/after source archives)

Initial GPU smoke: 5 passed. The first focused regression found 11 failures:
the new GPU parameter transfer preserved Fortran layout from superexchange and
rejected NumPy views with negative strides. Normalizing parameter layout before
transfer corrected these implementation issues; no mathematical change was
needed. Tests explicitly cover those layouts and reordered inputs.

Final validation used the same 21-module full-suite command recorded in phase 4,
with both GPU opt-in flags, after expanding `test_tilelang_execution.py`:
**437 passed, none skipped, in 143.45 seconds**. This includes actual GPU
execution for all maps, full/native topology, logical/physical QPYs, FP32/FP64,
shots, exports, CUDA-Q and Aer. Log: `/tmp/pqfmlib-matrix-full-tests.log`.
Existing Qiskit/Aer warnings remain. Compilation and `git diff --check` passed.

Benchmarks ran separately from tests. The pre-change implementation was archived
and measured before editing. The final measured source archive matches hashes
in all three after/diagnostic reports; the baseline archive matches the before
report. Main before/after input, training, theta and topology hashes match.
Commands and all repetitions are in [the report](benchmarks/tilelang_matrices.md).

On RTX 5070, 12 FP64 exact cases (CD-Ising/XYZ, synthetic heavy-hex,
8/12/16 qubits, m=1/2, 16 rows, five warmed repetitions) showed additional
execution speedups of 1.23–3.09x. All 24 final main-grid engine/configuration
measurements passed; maximum independent CPU error was 1.11e-14. Extra
one-row/128-row diagnostics compare GPU/CPU matrix construction within the new
executor; they must not be presented as the archived before/after comparison.

Limits: new kernels add first-use compilation cost; warmed speedups exclude fit,
preprocessing and compilation. GPU clocks/scheduling, width, depth, parameter
expressions and batching affect gains. No universal gain, bitwise cross-engine
identity, or new speedup claim for FP32, shots or superexchange is made. FP32
rounding and finite-shot error retain their existing meaning. MPS/noise/fake
backend/QPU remain excluded. No QPU jobs, commits or pushes occurred.

Next proposed phase: optimize evolution and observable-reduction kernels using
the remaining measured bottlenecks. It has not been started and requires the
user's approval.

## Additional performance phase 2: evolution and Pauli reduction

The user requested further improvement after phase 1, authorizing the previously
proposed second phase. This phase is complete. Added grouped-amplitude gate
application, larger Pauli reduction tiles with CUDA bit population counts, and
exact paired summation for terms containing X/Y. Real contributions at `i` and
`i XOR x` are equal for Hermitian Pauli words, including Y phases; one is summed
and doubled. Z-only terms retain the complete basis sum. No physical model,
Hamiltonian, topology, observable definition, precision mode or map/example
configuration changed.

Automatic selection retains the old arithmetic kernels on small states. Grouped
gates require at least 4096 amplitudes per state and 65536 across the batch;
fast Pauli reduction requires at least 1024 amplitudes. These are heuristics,
not optimal choices for every GPU. Compiled launchers are cached by signature.
`kernel_backend="baseline"` preserves the old kernels/dispatch for diagnostics;
`"optimized"` forces the new kernels. The default remains `"auto"` internally.
Memory estimates account for the changed reduction tile size. Pauli partitions
and kernel caches preserve term indices and are keyed appropriately when the
diagnostic backend changes.

Files changed/added in this phase:

- `pqfmlib/core/_tilelang_kernels.py`
- `pqfmlib/core/tilelang_execution.py`
- `pqfmlib/core/tilelang_matrices.py`
- `tests/test_tilelang_execution.py`, `tests/test_tilelang_sampling.py`
- `scripts/benchmark_tilelang.py`
- `README.md`, `docs/tilelang.md`, this log
- `docs/benchmarks/tilelang_kernel*`: report, JSON/CSV and source archives

Initial validation found a NumPy singleton slice with a negative stride that
still reported C contiguity; that layout is now copied before Torch transfer.
It also found that signed `T.popcount` did not lower with the installed compiler;
the kernel now calls the CUDA unsigned 64-bit intrinsic `__popcll` explicitly.
These were implementation/compiler-interface fixes. No new dependencies,
compiler version, host CUDA or driver changes were needed.

Validation on the RTX 5070, outside the sandbox:

- Initial existing smoke: 5 passed.
- New cross-backend GPU tests after fixes: 8 passed; after adding paired
  reduction, the same 8 passed again.
- Full 21-module regression command from phase 4 with both GPU opt-in flags:
  **454 passed, none skipped, in 228.12 seconds**. Log:
  `/tmp/pqfmlib-kernel-full-tests.log`.
- Then expanded the existing exhaustive Pauli/general-unitary test to auto and
  forced optimized modes. Both cases passed (**2 passed**, one additional unique
  case beyond the full run):

```bash
PQFMLIB_TEST_TILELANG_GPU=1 .venv_tilelang/bin/python -m pytest -q --disable-warnings --tb=short tests/test_tilelang_execution.py -k real_kernels_all_pauli_words
```

This includes all four native maps with forced new kernels in both precisions,
exact/shots, all three-qubit Pauli words, nonlinear angles, arbitrary unitary
lowering, reversed gate operands, partial/singleton batches, weighted sums,
QPY and exports. Existing Qiskit/Aer warnings remain. Compilation and
`git diff --check` passed. No QPU jobs, commits or pushes occurred.

Benchmarks were separate from tests. Archived/measured the pre-change source;
all before/after input, training, theta and topology hashes match. Current source
hashes match both final after/confirmation reports and the saved archive.
Main after run: 36 engine/configuration results, five warmed repetitions.
Confirmation: 24 results, fifteen repetitions, reversed engine order. Every
result passed numerical checks; maximum independent CPU error was 1.11e-14.
Commands, all timings, source archives and negative cases are in
[the kernel report](benchmarks/tilelang_kernels.md).

Confirmation on synthetic heavy-hex graphs: XYZ 16 qubits, m=1 fell from
31.40 to 22.74 ms (1.38x), m=2 from 49.74 to 41.10 ms (1.21x). The measured
large-case gain mainly comes from Pauli reduction, not a consistent gate
evolution speedup. Main-run Torch peak allocation for XYZ fell from 36.65 to
32.90 MiB (m=1), and 37.24 to 33.49 MiB (m=2); this is not whole-device memory.

Remaining limits: improvements are workload-dependent. CD-Ising 12-qubit m=1
was about 5% slower in confirmation; some small cases were much slower in the
separate before/after run and did not reproduce that slowdown in confirmation.
Both tables and raw repetitions are retained. No universal gain, bitwise
identity, or new FP32/shots/superexchange performance claim is made. Compilation
cost, CPU preprocessing, FP64 gate arithmetic and exponential live-state memory
remain relevant. MPS/noise/fake-backend/QPU stay excluded.

Next possible phase: investigate launch/capture overhead or further fusion
based on the remaining profiles, with fresh measurements and a separate plan.
It has not been started; any further phase requires user approval.

## Commit preparation and final validation

The user requested one local commit containing the TileLang implementations.
Reviewed and selected the executor/kernels, topology integration, QPY support,
FP32/shots, optimizations, optional dependency constraints, tests, simple example,
documentation and reproducible benchmark artifacts. The five source archives
were checked against all 12 corresponding JSON reports' source hashes. Local
documentation links were verified. Generated SVG trailing whitespace was
normalized; parsed geometry, attributes and text remain equivalent.

The historical 21-module full-suite commands above include two pre-existing,
ignored local modules: `test_in_memory_api.py` and `test_sklearn_integration.py`.
They are not part of this commit. Final validation instead ran all 19 modules
included in Git, so it can be repeated from the committed checkout:

```bash
PQFMLIB_TEST_TILELANG_GPU=1 PQFMLIB_TEST_CUDAQ_GPU=1 .venv_tilelang/bin/python -m pytest -q --disable-warnings --tb=short \
  tests/test_base_lifecycle.py tests/test_cd_ising_transformer.py tests/test_cudaq_execution.py \
  tests/test_execution.py tests/test_framework_selection.py tests/test_heisenberg_all_pairs.py \
  tests/test_heisenberg_fixed_circuits.py tests/test_heisenberg_topology.py tests/test_heisenberg_transformer.py \
  tests/test_precision.py tests/test_run_compatibility.py tests/test_statevector_execution.py \
  tests/test_superexchange.py tests/test_superexchange_backends.py tests/test_xyz_shap_notebook.py \
  tests/test_xyz_transformer.py tests/test_native_simulation.py tests/test_tilelang_execution.py \
  tests/test_tilelang_sampling.py
```

Result: **448 passed, none skipped, in 171.05 seconds** on the RTX 5070, with
the final exhaustive Pauli checks included. Existing Qiskit/Aer warnings remain.
Log: `/tmp/pqfmlib-tilelang-commit-tests.log`. The different count reflects
omitting those seven local integration tests, not removing committed coverage.
No GPU benchmarks were rerun or changed. Dependency environments and credentials
are excluded. No push, pull request or QPU job is part of this action.
