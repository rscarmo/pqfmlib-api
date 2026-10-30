"""Precision and physical finite-shot tests; GPU opt-in must fail if unavailable."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import SparsePauliOp
from sklearn.base import clone

from pqfmlib import CDIsingProjectiveQFM, HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM, XYZProjectiveQFM
from pqfmlib.core.execution import run_projected_feature_job
from pqfmlib.core.tilelang_execution import TileLangExecutor, _compile_job
from test_tilelang_execution import MAPS, prepared, exact, gpu, native_graph


@pytest.mark.parametrize("map_type", MAPS)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("method", ["shots", "exact", "statevector"])
def test_supported_precision_method_and_clone_without_runtime(map_type, precision, method):
    with patch("pqfmlib.core.tilelang_execution._load_runtime") as runtime:
        model = map_type("config", framework="tilelang", precision=precision, expectation_method=method)
        assert clone(model).get_params() == model.get_params()
    runtime.assert_not_called()


@pytest.mark.parametrize("map_type", MAPS)
@pytest.mark.parametrize("shots", [0, -1, 1.5, True, None, float("nan"), float("inf")])
def test_invalid_shots_rejected_before_runtime(map_type, shots):
    with patch("pqfmlib.core.tilelang_execution._load_runtime") as runtime:
        with pytest.raises(ValueError, match="shots"):
            map_type("invalid", framework="tilelang", expectation_method="shots", shots=shots)
    runtime.assert_not_called()


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**64])
def test_invalid_seed_rejected_before_runtime(seed):
    with patch("pqfmlib.core.tilelang_execution._load_runtime") as runtime:
        with pytest.raises(ValueError, match="seed"):
            TileLangExecutor(seed=seed)
    runtime.assert_not_called()


def test_low_level_invalid_shots_rejected_before_preparation():
    with patch("pqfmlib.core.execution.prepare_projected_feature_job") as prepare:
        with pytest.raises(ValueError, match="shots"):
            run_projected_feature_job(None, [], np.empty((1, 0)), None, None, None, [], [],
                                      framework="tilelang", simulation=True, expectation_method="shots", shots=0)
    prepare.assert_not_called()


def test_measurement_groups_use_complete_support_and_match_cudaq_grouping():
    from pqfmlib.core.cudaq_execution import _group_paulis
    from pqfmlib.core.observables import group_paulis_qubitwise
    labels = ["ZI", "IZ", "ZZ", "XX", "YY", "XI"]
    expected = [("ZZ", (0, 1, 2)), ("XX", (3, 5)), ("YY", (4,))]
    assert group_paulis_qubitwise(labels) == _group_paulis(labels) == expected
    job = _compile_job(prepared(QuantumCircuit(2), obs=[SparsePauliOp(label) for label in labels]))
    assert [g.members for g in job.sampling_groups] == [x[1] for x in expected]
    assert all(len(c.sites) == 2 for c in job.sampling_cones)


def options(map_type, native, graph):
    path, nodes, _backend = graph
    kwargs = dict(name_file="precision_sampling", q_enc=3, seed=17)
    X = np.random.default_rng(29).normal(size=(6, 4))
    if map_type is XYZProjectiveQFM:
        kwargs.update(features_per_qubit=1, axes=("z", "x", "y"), encoding_mode="shared_feature",
                      measure_cross_observables=True, m=2)
    elif map_type is CDIsingProjectiveQFM:
        kwargs.update(measure_all_zz=True, k_max=3, m=2)
    else:
        kwargs.update(measure_all_pairs=True, R=3, alpha=.9)
    if map_type is SuperexchangeHeisenbergPQFM:
        X = X[:, :3]
        kwargs.update(J0=1.3, ell=.6)
    if native:
        kwargs.update(ideal=False, topology="native", topology_file=str(path),
                      use_fixed_phys_nodes=True, fixed_phys_nodes_file=str(nodes))
    return kwargs, X


@gpu
@pytest.mark.parametrize("map_type", MAPS)
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("strategy", ["auto", "reference"])
def test_fp32_all_maps_qpy_and_refit(map_type, native, strategy, native_graph, tmp_path):
    kwargs, X = options(map_type, native, native_graph)
    expected = map_type(**kwargs, expectation_method="statevector").fit_transform(X)
    model = map_type(**kwargs, framework="tilelang", precision="single", expectation_method="exact").fit(X)
    model.backend = TileLangExecutor(precision="single", strategy=strategy)
    actual = model.transform(X)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert model.backend.last_execution_stats["state_dtype"] == "float32"
    np.testing.assert_array_equal(model.transform(X), actual)
    path = tmp_path / "fp32.qpy"
    with path.open("wb") as f:
        qpy.dump(model._prepared_execution.qc_t, f)
    restored = map_type(**kwargs, framework="tilelang", precision="single", expectation_method="statevector",
                        fixed_circuit_file_name=str(path))
    np.testing.assert_allclose(restored.fit_transform(X), expected, atol=2e-5, rtol=2e-5)
    model.set_params(precision="double")
    with pytest.raises(RuntimeError, match="configuration changed"):
        model.transform(X)
    np.testing.assert_allclose(model.fit_transform(X), expected, atol=1e-10, rtol=1e-10)


@gpu
def test_single_precision_really_uses_float32_state_and_matrices():
    import torch
    qc = QuantumCircuit(3)
    qc.ry(.7, 0)
    qc.cx(0, 2)
    executor = TileLangExecutor(precision="single")
    original = executor._kernels.apply_batched_gate
    seen = []
    def inspect(*args):
        kernel = original(*args)
        def run(state, matrix, *remaining):
            seen.append((state.dtype, matrix.dtype))
            return kernel(state, matrix, *remaining)
        return run
    with patch.object(executor._kernels, "apply_batched_gate", side_effect=inspect):
        executor.execute(prepared(qc), np.empty((1, 0)))
    assert seen and all(a == b == torch.float32 for a, b in seen)


@gpu
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("strategy", ["auto", "full", "cones", "reference"])
def test_joint_bell_measurements_preserve_covariance_and_seed(precision, strategy):
    import torch
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = [SparsePauliOp(label) for label in ("ZI", "IZ", "ZZ", "XX", "YY")]
    obs.append(SparsePauliOp.from_list([("ZI", .5), ("IZ", -.5), ("II", .25)]))
    job = prepared(qc, obs=obs)
    rows = np.empty((7, 0))
    executor = TileLangExecutor(precision=precision, seed=71, strategy=strategy, batch_size=3)
    first, _ = executor.execute(job, rows, expectation_method="shots", shots=4097)
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    again, _ = executor.execute(job, rows, expectation_method="shots", shots=4097)
    np.testing.assert_array_equal(first, again)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    np.testing.assert_array_equal(first[:, 0], first[:, 1])
    np.testing.assert_allclose(first[:, 2:5], np.tile([1., 1., -1.], (len(rows), 1)), atol=1e-12)
    np.testing.assert_allclose(first[:, 5], .25, atol=1e-15)
    assert np.max(np.abs(first[:, :2])) < 5 / np.sqrt(4097)
    assert executor.last_execution_stats["n_measurement_groups"] == 3
    other = TileLangExecutor(precision=precision, seed=72, strategy=strategy, batch_size=3)
    assert not np.array_equal(first[:, 0], other.execute(job, rows, expectation_method="shots", shots=4097)[0][:, 0])


@gpu
@pytest.mark.parametrize("precision", ["single", "double"])
def test_shot_statistics_have_correct_mean_variance_and_count(precision):
    qc = QuantumCircuit(1)
    qc.ry(np.arccos(.3), 0)
    job = prepared(qc, obs=[SparsePauliOp("Z")])
    executor = TileLangExecutor(precision=precision, seed=2026)
    variances = []
    for shots in (32, 512):
        values = executor.execute(job, np.empty((256, 0)), expectation_method="shots", shots=shots)[0][:, 0]
        theory = (1 - .3**2) / shots
        assert abs(values.mean() - .3) < 5 * np.sqrt(theory / len(values))
        variance = values.var(ddof=1)
        assert .45 * theory < variance < 1.65 * theory
        np.testing.assert_allclose((values + 1) * shots / 2, np.round((values + 1) * shots / 2), atol=1e-12)
        assert executor.last_execution_stats["sampled_shots"] == len(values) * shots
        variances.append(variance)
    assert variances[0] > 4 * variances[1]


@gpu
def test_sampling_more_than_64_terms_and_identity():
    qc = QuantumCircuit(7)
    qc.h(0)
    for q in range(1, 7):
        qc.cx(0, q)
    labels = ["".join("Z" if (mask >> q) & 1 else "I" for q in range(7)) for mask in range(128)]
    executor = TileLangExecutor(seed=91)
    values = executor.execute(prepared(qc, obs=[SparsePauliOp(x) for x in labels]),
                              np.empty((3, 0)), expectation_method="shots", shots=4097)[0]
    for mask in range(128):
        if mask.bit_count() % 2 == 0:
            np.testing.assert_array_equal(values[:, mask], 1.)
        else:
            np.testing.assert_array_equal(values[:, mask], values[:, 1])
    assert executor.last_execution_stats["n_measurement_groups"] == 1


@gpu
def test_sampling_wide_physical_support_uses_small_joint_cone():
    qc = QuantumCircuit(127)
    qc.h(0)
    qc.cx(0, 126)
    labels = ["Z" + "I" * 126, "I" * 126 + "Z", "Z" + "I" * 125 + "Z"]
    executor = TileLangExecutor(precision="single")
    values = executor.execute(prepared(qc, obs=[SparsePauliOp(x) for x in labels]),
                              np.empty((3, 0)), expectation_method="shots", shots=4097)[0]
    np.testing.assert_array_equal(values[:, 0], values[:, 1])
    np.testing.assert_array_equal(values[:, 2], 1.)
    assert executor.last_execution_stats["max_cone_qubits"] == 2


@gpu
@pytest.mark.parametrize("map_type", MAPS)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_shots_native_physical_qpy_and_exports(map_type, precision, native_graph, tmp_path):
    kwargs, X = options(map_type, True, native_graph)
    _path, _nodes, backend = native_graph
    reference = map_type(**kwargs, expectation_method="statevector").fit(X)
    expected = reference.transform(X)
    logical = (reference._build_circuit_and_observables()[0] if isinstance(reference, HeisenbergProjectiveQFM)
               else reference._prepared_execution.qc_t)
    physical = transpile(logical, backend=backend, initial_layout=[2, 1, 3], optimization_level=1, seed_transpiler=42)
    path = tmp_path / "physical_shots.qpy"
    with path.open("wb") as f:
        qpy.dump(physical, f)
    model = map_type(**kwargs, framework="tilelang", expectation_method="shots", precision=precision,
                     shots=8193, fixed_circuit_file_name=str(path), output_root=str(tmp_path))
    actual = model.fit_transform(X)
    np.testing.assert_allclose(actual, expected, atol=6 / np.sqrt(model.shots), rtol=0)
    np.testing.assert_array_equal(model.transform(X), actual)
    model.load_data = lambda: None
    model.X_q_all, model.y = X, np.arange(len(X)) % 2
    result = model.run()
    np.testing.assert_array_equal(result["Xq_all_raw"], actual)
    np.testing.assert_allclose(pd.read_csv(result["csv_path"]).drop(columns="y").to_numpy(), actual, atol=1e-12)
    np.testing.assert_array_equal(np.load(result["npy_path"]), actual)


@gpu
@pytest.mark.parametrize("map_type", MAPS)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_optimized_kernels_all_maps_native_exact_and_shots(map_type, precision, native_graph):
    kwargs, X = options(map_type, True, native_graph)
    expected = map_type(**kwargs, expectation_method="statevector").fit_transform(X)
    model = map_type(**kwargs, framework="tilelang", precision=precision, expectation_method="exact").fit(X)
    model.backend.kernel_backend = "optimized"
    tolerance = 2e-5 if precision == "single" else 1e-10
    np.testing.assert_allclose(model.transform(X), expected, atol=tolerance, rtol=tolerance)
    assert model.backend.last_execution_stats["grouped_gate_calls"] > 0
    assert model.backend.last_execution_stats["fast_pauli_calls"] > 0
    model.set_params(expectation_method="shots", shots=4097).fit(X)
    model.backend.kernel_backend = "optimized"
    sampled = model.transform(X)
    np.testing.assert_allclose(sampled, expected, atol=6 / np.sqrt(4097), rtol=0)
    np.testing.assert_array_equal(model.transform(X), sampled)
