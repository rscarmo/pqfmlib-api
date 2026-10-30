# Additional performance phase 2: evolution and Pauli kernels

The default TileLang executor now reuses compiled launchers and selects grouped-amplitude gates and paired Pauli reductions for larger workloads. Feature-map Hamiltonians, topology, precision settings and readout remain unchanged.

## Method

- RTX 5070; FP64 exact expectations; synthetic heavy-hex graphs; CD-Ising/XYZ; 8/12/16 active qubits; m=1/2; 64 fitting rows, 16 evaluation rows, one fitted block. XYZ uses shared-feature X/Y/Z and cross interactions/readout.
- A pre-change executor was archived and measured first. The main after run compares `auto`, `baseline_kernels` (preserved old kernels/dispatch) and `optimized_kernels` (forced new kernels). Medians use five warmed repetitions.
- Confirmation: fifteen warmed repetitions per engine, reversing the main run order to `baseline_kernels`, then `auto`. This is a sequential within-process comparison, not randomized/interleaved or a confidence-interval study. It helps expose timing variation but does not eliminate scheduling/clock effects.
- All before/after/confirmation input, training, theta and topology hashes match. Every execution is numerically checked, with independent Qiskit CPU checks on up to two rows. Benchmarks ran without GPU tests concurrently.
- Times exclude fit, CPU parameter preparation and first-use compilation. Raw files also include parameter-plus-execution time, repetitions and separate synchronized profiles. First calls share persistent compiler caches and are not isolated cold measurements.

## Confirmation: new versus preserved kernels

| Map | Qubits | m | Preserved kernels (ms) | Auto (ms) | Ratio |
|---|---:|---:|---:|---:|---:|
| cd-ising | 8 | 1 | 0.815 | 0.757 | 1.08x |
| xyz | 8 | 1 | 1.716 | 1.699 | 1.01x |
| cd-ising | 8 | 2 | 1.022 | 0.886 | 1.15x |
| xyz | 8 | 2 | 3.479 | 3.188 | 1.09x |
| cd-ising | 12 | 1 | 1.014 | 1.063 | 0.95x |
| xyz | 12 | 1 | 3.852 | 3.192 | 1.21x |
| cd-ising | 12 | 2 | 1.344 | 1.260 | 1.07x |
| xyz | 12 | 2 | 4.863 | 4.480 | 1.09x |
| cd-ising | 16 | 1 | 2.906 | 2.602 | 1.12x |
| xyz | 16 | 1 | 31.402 | 22.745 | 1.38x |
| cd-ising | 16 | 2 | 7.528 | 6.603 | 1.14x |
| xyz | 16 | 2 | 49.737 | 41.096 | 1.21x |

The strongest confirmed improvement is XYZ at 16 qubits, m=1: about 31.4 to 22.7 ms (1.38x throughput, about 28% less execution time). At m=2: about 49.7 to 41.1 ms (1.21x). CD-Ising gains are smaller; the 12-qubit m=1 confirmation is about 5% slower. No universal gain is claimed.

## Archived executor: before/after main run

| Map | Qubits | m | Before (ms) | After (ms) | Ratio |
|---|---:|---:|---:|---:|---:|
| cd-ising | 8 | 1 | 0.787 | 0.782 | 1.01x |
| xyz | 8 | 1 | 1.837 | 2.559 | 0.72x |
| cd-ising | 8 | 2 | 0.887 | 1.276 | 0.69x |
| xyz | 8 | 2 | 3.208 | 3.031 | 1.06x |
| cd-ising | 12 | 1 | 0.973 | 0.933 | 1.04x |
| xyz | 12 | 1 | 3.231 | 3.117 | 1.04x |
| cd-ising | 12 | 2 | 1.648 | 1.280 | 1.29x |
| xyz | 12 | 2 | 7.218 | 5.238 | 1.38x |
| cd-ising | 16 | 1 | 3.373 | 3.507 | 0.96x |
| xyz | 16 | 1 | 31.483 | 23.085 | 1.36x |
| cd-ising | 16 | 2 | 7.334 | 6.524 | 1.12x |
| xyz | 16 | 2 | 49.736 | 41.195 | 1.21x |

Small cases varied substantially: XYZ 8-qubit m=1 and CD-Ising 8-qubit m=2 were slower in the separate main before/after run; those regressions were not reproduced in the longer within-process confirmation. Auto retains the original arithmetic kernels at 8 qubits, but launcher caching/dispatch changes and run conditions can still affect wall time. The full table and repetitions are provided rather than presenting only favorable cases.

## What changed

- A grouped gate kernel reads a two-/four-amplitude group once and computes all outputs. Auto uses it at widths of at least 4096 amplitudes and at least 65536 amplitudes across the batch; smaller workloads keep the original gate kernel.
- Exact Pauli reduction uses CUDA `__popcll` parity and 1024-element tiles for states with at least 1024 amplitudes. Terms with X/Y use one representative from each `i, i XOR x` pair and double its contribution; the real contributions are equal for Hermitian Pauli words, including Y phases. Pure Z terms retain a full-basis sum.
- Compiled launchers are cached by shape/signature; the previous path remains under `kernel_backend="baseline"`. `"optimized"` forces the new kernels for diagnostics. No new public map options are required.
- A new test exposed NumPy singleton slices that retained a negative stride despite their contiguous flag. The parameter transfer now copies that special case. This is a memory-layout fix, not a superexchange/theory change.

Grouped gates did not show a consistent evolution-time improvement in the measured FP64 grid; the clear large-case gain is the Pauli reduction. For XYZ 16 qubits m=1, its separately profiled reduction fell from roughly 12 ms to roughly 3.4 ms. Torch peak allocation in the main run fell from 36.65 to 32.90 MiB (m=1) and 37.24 to 33.49 MiB (m=2). This measures Torch allocations, not whole-device memory.

## Correctness and limits

All 36 final main-run engine/configuration results and 24 confirmation results passed numerical checks; maximum independent CPU error: 1.11e-14.

The full regression passed **454 tests, none skipped**, including actual GPU execution for all four maps, native connectivity, QPY, FP32/FP64, shots and exports. Eight new cross-backend tests cover one/two/seven/twelve-qubit states, reversed gate operands, Pauli Y phases, weighted sums, partial batches and shots. Eight further cases force the new kernels on all four native maps in both precisions, for exact/shots. After parameterizing the existing exhaustive three-qubit Pauli test, both auto/optimized cases passed separately (**2 passed**); this adds one unique test beyond the full run.

All algorithms retain floating-point rounding; reduction order changes, so cross-engine bitwise equality is not promised. Shots retain sampling error. No new performance claim is made for FP32, shots, Heisenberg or superexchange; correctness tests do cover them. New kernels add compilation/caching cost, and thresholds are heuristics tuned on this GPU. MPS/noise/fake-backend/QPU remain excluded. No dependency, driver or CUDA changes were made.

## Reproduce

Use the [baseline source archive](tilelang_kernel_baseline_source.tar.gz) for the first command and the [new source archive](tilelang_kernel_source.tar.gz) for the others. Their hashes match the corresponding raw reports.

```bash
.venv_tilelang/bin/python scripts/benchmark_tilelang.py --qubits 8 12 16 --steps 1 2 --rows 16 --repeats 5 --engines auto --output /tmp/kernel-before.json
.venv_tilelang/bin/python scripts/benchmark_tilelang.py --qubits 8 12 16 --steps 1 2 --rows 16 --repeats 5 --engines auto baseline_kernels optimized_kernels --output /tmp/kernel-after.json
.venv_tilelang/bin/python scripts/benchmark_tilelang.py --qubits 8 12 16 --steps 1 2 --rows 16 --repeats 15 --engines baseline_kernels auto --output /tmp/kernel-confirmation.json
```

[Before JSON](tilelang_kernel_before.json), [CSV](tilelang_kernel_before.csv); [after JSON](tilelang_kernel_after.json), [CSV](tilelang_kernel_after.csv); [confirmation JSON](tilelang_kernel_confirmation.json), [CSV](tilelang_kernel_confirmation.csv).
