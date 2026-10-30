"""Opt-in, offline native-graph comparisons with identical circuits and rows.

Times execution separately from parameter preparation, fit and synchronized
profiling. Generated heavy-hex graphs are synthetic, not device calibrations.
Nothing runs on import, and no QPU jobs or network services are used.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from pqfmlib import CDIsingProjectiveQFM, XYZProjectiveQFM
from pqfmlib.core.execution import execute_statevector_features
from pqfmlib.core.tilelang_execution import TileLangExecutor
from pqfmlib.maps.cd_ising import make_theta_matrix_all_blocks_hw, s_curve, ds_curve, makeAlpha1
from pqfmlib.maps.xyz import make_theta_matrix_full_cross_blocks


def parameter_matrix(model, X):
    if isinstance(model, CDIsingProjectiveQFM):
        return make_theta_matrix_all_blocks_hw(X, model.blocks, model.edges_log, model.tau, model.m,
                                              s_curve, ds_curve, makeAlpha1)[0]
    return make_theta_matrix_full_cross_blocks(
        X, model.blocks, model.edges_log, features_per_qubit=len(model.axes), axes=model.axes,
        encoding_mode=model.encoding_mode, tau=model.tau, m_phys=model.m,
        keep_diagonal_terms=model.keep_diagonal_terms, keep_cross_terms=model.keep_cross_terms)[0]


def graph_edges(kind, qubits):
    if kind == "chain":
        return [[i, i + 1] for i in range(qubits - 1)]
    if kind != "heavy_hex":
        raise ValueError("Synthetic graph must be 'chain' or 'heavy_hex'.")
    distance = 3
    while CouplingMap.from_heavy_hex(distance).size() < qubits:
        distance += 2
    return [list(edge) for edge in CouplingMap.from_heavy_hex(distance).get_edges()]


def _aer(device, qubits):
    backend = AerSimulator(method="statevector", device=device, precision="double", seed_simulator=42)
    backend.set_max_qubits(qubits)
    return backend


def benchmark_case(model, X, *, repeats=3, engines=("reference", "full", "auto", "cudaq", "aer_gpu"),
                   batch_size=128):
    """All engines execute the same prepared job; every result is validated."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    import torch
    job = model._prepared_execution
    parameter_timings = []
    for _ in range(repeats):
        before = perf_counter()
        theta = parameter_matrix(model, X)
        parameter_timings.append(perf_counter() - before)
    # Independent Qiskit validation, outside performance measurements.
    cpu = _aer("CPU", job.qc_t.num_qubits)
    expected_cpu = execute_statevector_features(job, theta[:2], cpu)[0]
    rows, reference = [], None
    for label in engines:
        before = perf_counter()
        if label in ("reference", "full", "auto", "cones", "cpu_matrices", "baseline_kernels", "optimized_kernels"):
            executor = TileLangExecutor(strategy="auto" if label in ("cpu_matrices", "baseline_kernels", "optimized_kernels") else label,
                                       batch_size=batch_size,
                                       matrix_backend="cpu" if label == "cpu_matrices" else "auto",
                                       kernel_backend={"baseline_kernels": "baseline", "optimized_kernels": "optimized"}.get(label, "auto"))
            execute = lambda values: executor.execute(job, values)[0]
        elif label == "cudaq":
            from pqfmlib.core.cudaq_execution import CudaqExecutor
            executor = CudaqExecutor(precision="double", mps=False)
            execute = lambda values: executor.execute(job, values, expectation_method="exact")[0]
        elif label in ("aer_gpu", "aer_cpu"):
            executor = _aer("GPU" if label == "aer_gpu" else "CPU", job.qc_t.num_qubits)
            execute = lambda values: execute_statevector_features(job, values, executor)[0]
        else:
            raise ValueError(f"Unknown benchmark engine: {label}")
        setup_seconds = perf_counter() - before
        torch.cuda.synchronize()
        before = perf_counter()
        actual = execute(theta)
        torch.cuda.synchronize()
        cold_seconds = perf_counter() - before
        np.testing.assert_allclose(actual[:2], expected_cpu, atol=1e-9, rtol=1e-9)
        if reference is None:
            reference = actual.copy()
        np.testing.assert_allclose(actual, reference, atol=1e-9, rtol=1e-9)
        timings, parameter_and_execution = [], []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            torch.cuda.synchronize()
            before = perf_counter()
            actual = execute(theta)
            torch.cuda.synchronize()
            timings.append(perf_counter() - before)
            np.testing.assert_allclose(actual, reference, atol=1e-9, rtol=1e-9)
            before = perf_counter()
            combined = execute(parameter_matrix(model, X))
            torch.cuda.synchronize()
            parameter_and_execution.append(perf_counter() - before)
            np.testing.assert_allclose(combined, reference, atol=1e-9, rtol=1e-9)
        peak = torch.cuda.max_memory_allocated() if isinstance(executor, TileLangExecutor) else None
        stats = getattr(executor, "last_execution_stats", None)
        if isinstance(executor, TileLangExecutor) and label != "reference":
            # Separate profiling run: synchronized stage times do not affect medians.
            executor.profile = True
            profiled = execute(theta)
            np.testing.assert_allclose(profiled, reference, atol=1e-9, rtol=1e-9)
            stats = executor.last_execution_stats
        median = statistics.median(timings)
        row = dict(engine=label, n_samples=len(X), n_parameters=theta.shape[1],
                   n_observables=actual.shape[1], repeats=repeats,
                   parameter_seconds=statistics.median(parameter_timings),
                   backend_setup_seconds=setup_seconds, cold_execution_seconds=cold_seconds,
                   execution_seconds=median, execution_samples_seconds=timings,
                   parameter_and_execution_seconds=statistics.median(parameter_and_execution),
                   rows_per_second=len(X) / median, torch_peak_allocated_bytes=peak,
                   max_abs_error_vs_first_engine=float(np.max(np.abs(actual - reference))),
                   max_abs_error_vs_cpu_first_two=float(np.max(np.abs(actual[:2] - expected_cpu))),
                   parameter_sha256=hashlib.sha256(theta.tobytes()).hexdigest(),
                   stats=stats)
        rows.append(row)
        print(f"  {label}: {median:.6f} s; error={row['max_abs_error_vs_first_engine']:.3g}", flush=True)
        del executor
    by_engine = {row["engine"]: row for row in rows}
    for row in rows:
        for base in ("reference", "cudaq", "aer_gpu"):
            row[f"speedup_vs_{base}"] = (by_engine[base]["execution_seconds"] / row["execution_seconds"]
                                            if base in by_engine else None)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--graphs", nargs="+", choices=("chain", "heavy_hex"), default=["heavy_hex"])
    parser.add_argument("--topology-file", type=Path, help="Use an actual saved graph instead of synthetic graphs")
    parser.add_argument("--maps", nargs="+", choices=("cd-ising", "xyz"), default=["cd-ising", "xyz"])
    parser.add_argument("--engines", nargs="+", choices=("reference", "full", "auto", "cones", "cpu_matrices", "baseline_kernels", "optimized_kernels", "cudaq", "aer_gpu", "aer_cpu"),
                        default=["reference", "full", "auto", "cudaq", "aer_gpu"])
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--train-rows", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=1, help="Input feature count is qubits * blocks")
    parser.add_argument("--encoding", choices=("shared_feature", "multi_axis"), default="shared_feature")
    parser.add_argument("--axes", default="xyz")
    parser.add_argument("--cross-terms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cross-observables", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.rows < 1 or args.train_rows < 4 or args.blocks < 1 or args.repeats < 1
            or min(args.qubits) < 2 or min(args.steps) < 1 or not 1 <= args.batch_size <= 65535):
        parser.error("Require positive rows/blocks/repeats/steps, at least 2 qubits and 4 training rows.")
    import torch
    results = []
    with tempfile.TemporaryDirectory(prefix="pqfmlib-tilelang-benchmark-") as root:
        for n in args.qubits:
            for graph in (["snapshot"] if args.topology_file else args.graphs):
                path = args.topology_file or Path(root) / "topology.json"
                if not args.topology_file:
                    path.write_text(json.dumps(dict(schema_version=1, coupling_edges=graph_edges(graph, n))))
                rng = np.random.default_rng(42 + n)
                features = n * args.blocks
                # Correlated data avoids degenerate MI graphs; each engine sees identical rows.
                train = rng.normal(size=(args.train_rows, 1)) + .15 * rng.normal(size=(args.train_rows, features))
                X = rng.normal(size=(args.rows, 1)) + .15 * rng.normal(size=(args.rows, features))
                for steps in args.steps:
                    for name in args.maps:
                        options = dict(name_file="benchmark", q_enc=n, ideal=False, simulation=True,
                                       fakebackend=False, topology="native", topology_file=str(path),
                                       framework="tilelang", expectation_method="exact", precision="double", m=steps)
                        if name == "xyz":
                            axes = tuple(args.axes)
                            options.update(features_per_qubit=1 if args.encoding == "shared_feature" else len(axes),
                                           encoding_mode=args.encoding, axes=axes,
                                           keep_cross_terms=args.cross_terms,
                                           measure_cross_observables=args.cross_observables)
                        before = perf_counter()
                        model = (CDIsingProjectiveQFM if name == "cd-ising" else XYZProjectiveQFM)(**options).fit(train)
                        fit_seconds = perf_counter() - before
                        print(f"{name}: {graph}, {n} qubits, m={steps}, rows={len(X)}, blocks={len(model.blocks)}", flush=True)
                        rows = benchmark_case(model, X, repeats=args.repeats, engines=tuple(args.engines), batch_size=args.batch_size)
                        for row in rows:
                            row.update(map=name, graph=graph, qubits=n, steps=steps, fit_seconds=fit_seconds,
                                       n_blocks=len(model.blocks), encoding=args.encoding if name == "xyz" else None,
                                       axes=args.axes if name == "xyz" else "z", n_edges=len(model.edges_act),
                                       phys_nodes=model.phys_nodes, logical_edges=model.edges_act,
                                       input_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                                       train_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                                       topology_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        results.extend(rows)
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), root / "pqfmlib/core/tilelang_execution.py",
                    root / "pqfmlib/core/tilelang_plan.py", root / "pqfmlib/core/_tilelang_kernels.py",
                    root / "pqfmlib/core/tilelang_matrices.py", root / "pqfmlib/core/tilelang_sampling.py"]
    versions = {}
    for name in ("torch", "tilelang", "cuda-quantum-cu12", "qiskit-aer-gpu", "qiskit-aer"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = dict(gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  cuda=torch.version.cuda,
                  versions=versions,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  options={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    flat = [{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in results]
    pd.DataFrame(flat).to_csv(args.output.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
