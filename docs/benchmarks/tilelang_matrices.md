# GPU matrix preparation: additional performance phase 1

The default TileLang executor now prepares affine fused matrices on GPU and reuses fixed metadata. This report compares against the actual executor saved before this change, using identical inputs. It does not change feature-map physics or precision.

## Protocol

- RTX 5070; FP64; exact expectations; synthetic heavy-hex graphs; 8/12/16 active qubits; m=1/2; 64 fitting rows and 16 evaluation rows; one fitted block. XYZ includes shared-feature X/Y/Z interactions and cross readout.
- Before and after were measured separately on the same GPU, without another test/benchmark suite running. Input, training, theta and topology hashes match for all 12 pairs. Every execution is checked; up to two rows use independent Qiskit CPU expectations.
- Table: medians of five warmed executions, excluding fit, parameter preprocessing and first-use compilation. Raw files also include parameter-plus-execution time, first-call time and separate synchronized profiling. GPU scheduling/clocks and run order can affect results; gains are not universal.
- The new runner also times `cpu_matrices` in the same process: the new executor with CPU matrix construction, retaining metadata caching. It isolates matrix preparation more closely but is not identical to the pre-change executor.
- First-call times are within a process with persistent compiler caches, not a clean compiler-cache benchmark. Initial trials had first calls around 3 seconds including compilation of the new matrix kernels; warmed gains do not imply a faster first call.

## Before/after warmed execution

| Map | Qubits | m | Before (ms) | After (ms) | Additional speedup |
|---|---:|---:|---:|---:|---:|
| cd-ising | 8 | 1 | 1.581 | 1.061 | 1.49x |
| xyz | 8 | 1 | 5.112 | 1.696 | 3.01x |
| cd-ising | 8 | 2 | 2.471 | 1.066 | 2.32x |
| xyz | 8 | 2 | 9.533 | 3.756 | 2.54x |
| cd-ising | 12 | 1 | 2.291 | 1.152 | 1.99x |
| xyz | 12 | 1 | 8.931 | 3.706 | 2.41x |
| cd-ising | 12 | 2 | 3.667 | 1.329 | 2.76x |
| xyz | 12 | 2 | 15.782 | 5.102 | 3.09x |
| cd-ising | 16 | 1 | 4.766 | 3.065 | 1.55x |
| xyz | 16 | 1 | 38.845 | 31.502 | 1.23x |
| cd-ising | 16 | 2 | 9.940 | 6.970 | 1.43x |
| xyz | 16 | 2 | 64.708 | 49.648 | 1.30x |

## Correctness and limits

Maximum independent CPU error over the final main-grid engines: 1.11e-14. All 24 final engine/configuration results passed the numerical checks. The older 12 baseline results also passed.

The full regression suite passed **437 tests, none skipped**, including all four maps, full/native connectivity, logical/routed physical QPYs, FP32/FP64, finite-shot statistics, exports and CUDA-Q/Aer regressions. New tests compare GPU fused matrices directly with CPU matrices, cover reversed operands, affine combinations, Fortran-layout/negative-stride inputs, nonlinear fallbacks, partial batches and cache invalidation.

Matrix construction remains FP64, with the result stored in FP32 or FP64 as selected. Arbitrary nonlinear QPY expressions use the preserved CPU binder for their affected fused matrices. Floating-point operation order can change rounding; no bitwise agreement across engines is promised. Sampled features retain statistical error. The speed table does not benchmark FP32, shots, Heisenberg or superexchange performance.

State evolution and observable-reduction kernels were not optimized in this phase. Fit, graph assignment and preprocessing remain on CPU. MPS, noise, fake-backend and QPU execution remain outside TileLang scope. No dependencies, driver or CUDA versions were changed.

## Reproduce

Before-change sources: [baseline archive](tilelang_matrix_baseline_source.tar.gz). Final measured sources: [current archive](tilelang_matrix_source.tar.gz). Their relevant SHA-256 hashes match the corresponding JSON reports. Use the baseline sources to reproduce the before command; running it on current sources measures the new executor.

```bash
.venv_tilelang/bin/python scripts/benchmark_tilelang.py --qubits 8 12 16 --steps 1 2 --rows 16 --repeats 5 --engines auto --output /tmp/matrix-before.json
.venv_tilelang/bin/python scripts/benchmark_tilelang.py --qubits 8 12 16 --steps 1 2 --rows 16 --repeats 5 --engines auto cpu_matrices --output /tmp/matrix-after.json
```

[Before JSON](tilelang_matrix_before.json), [before CSV](tilelang_matrix_before.csv), [after JSON](tilelang_matrix_after.json), [after CSV](tilelang_matrix_after.csv).

## Batch-size diagnostics

These compare GPU versus CPU matrix construction in the **new executor**, not the archived pre-change version. Both keep the metadata cache; seven warmed repetitions, 8 qubits, m=1.

| Map | Rows | GPU matrices (ms) | CPU matrices (ms) | CPU/GPU ratio |
|---|---:|---:|---:|---:|
| cd-ising | 1 | 0.702 | 1.669 | 2.38x |
| xyz | 1 | 2.272 | 5.409 | 2.38x |
| cd-ising | 128 | 0.984 | 3.207 | 3.26x |
| xyz | 128 | 2.088 | 9.987 | 4.78x |

```bash
.venv_tilelang/bin/python scripts/benchmark_tilelang.py --qubits 8 --steps 1 --rows 1 --repeats 7 --engines auto cpu_matrices --output /tmp/matrix-single.json
.venv_tilelang/bin/python scripts/benchmark_tilelang.py --qubits 8 --steps 1 --rows 128 --repeats 7 --engines auto cpu_matrices --output /tmp/matrix-batch128.json
```

[Single-row JSON](tilelang_matrix_single_row.json), [CSV](tilelang_matrix_single_row.csv), [128-row JSON](tilelang_matrix_batch128.json), [CSV](tilelang_matrix_batch128.csv).
