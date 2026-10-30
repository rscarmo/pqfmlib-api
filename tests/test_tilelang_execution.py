"""CPU contracts and real GPU tests (PQFMLIB_TEST_TILELANG_GPU=1)."""

import builtins
import itertools
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector, random_unitary
from sklearn.base import clone

from pqfmlib import CDIsingProjectiveQFM, HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM, XYZProjectiveQFM
from pqfmlib.core.execution import PreparedProjectedFeatureJob, execute_prepared_projected_feature_job, run_projected_feature_job
from pqfmlib.core.tilelang_execution import TileLangExecutor, _compile_job, _pauli_terms
from pqfmlib.hardware.topology import save_topology_snapshot


MAPS = (CDIsingProjectiveQFM, XYZProjectiveQFM, HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM)
gpu = pytest.mark.skipif(os.environ.get("PQFMLIB_TEST_TILELANG_GPU") != "1",
                         reason="Set PQFMLIB_TEST_TILELANG_GPU=1 for actual CUDA kernel tests.")


@pytest.mark.parametrize("map_type", MAPS)
@pytest.mark.parametrize("overrides", [dict(expectation_method="unsupported"), dict(mps=True),
    dict(precision="half"), dict(statevector_device="CPU"), dict(simulation=False),
    dict(fakebackend=True), dict(ideal=False), dict(resource_estimation=True)])
def test_invalid_modes_fail_before_runtime_import(map_type, overrides):
    with patch("pqfmlib.core.tilelang_execution._load_runtime") as load:
        with pytest.raises(ValueError):
            map_type("invalid", **(dict(framework="tilelang", expectation_method="exact") | overrides))
    load.assert_not_called()


@pytest.mark.parametrize("map_type", MAPS)
def test_clone_and_revalidation_preserve_optional_imports(map_type):
    qfm = map_type("clone", framework="tilelang", expectation_method="exact")
    assert clone(qfm).get_params() == qfm.get_params()
    qfm.set_params(precision="half")
    with patch("pqfmlib.core.tilelang_execution._load_runtime") as load:
        with pytest.raises(ValueError, match="precision"):
            qfm.setup_backend_and_estimator()
    load.assert_not_called()


def test_missing_optional_dependencies_never_fall_back():
    qfm = HeisenbergProjectiveQFM("missing", framework="tilelang", expectation_method="exact")
    with patch("pqfmlib.core.tilelang_execution.import_module", side_effect=ImportError):
        with patch.object(qfm, "_make_aer_backend") as aer:
            with pytest.raises(ImportError, match=r"pqfmlib\[tilelang\]"):
                qfm.setup_backend_and_estimator()
    aer.assert_not_called()


def test_missing_gpu_never_falls_back():
    torch, kernels = Mock(), Mock()
    torch.cuda.is_available.return_value = False
    with patch("pqfmlib.core.tilelang_execution.import_module", side_effect=[torch, kernels]):
        with pytest.raises(RuntimeError, match="accessible NVIDIA CUDA GPU"):
            TileLangExecutor()


def test_qiskit_does_not_import_tilelang_or_torch():
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("tilelang", "torch"):
            raise AssertionError("optional import on Qiskit path")
        return original(name, *args, **kwargs)
    with patch("builtins.__import__", side_effect=guarded):
        HeisenbergProjectiveQFM("optional", q_enc=2, expectation_method="statevector").fit_transform([[.2]])


@pytest.mark.parametrize("instruction", ["reset", "measure", "initialize", "delay"])
def test_invalid_qpy_instructions_are_rejected(instruction):
    qc = QuantumCircuit(2)
    if instruction == "measure":
        qc.measure_all()
    elif instruction == "initialize":
        qc.initialize([1, 0], 0)
    elif instruction == "delay":
        qc.delay(10, 0)
    else:
        qc.reset(0)
    with pytest.raises(ValueError, match="TileLang"):
        _compile_job(PreparedProjectedFeatureJob(qc, (), [], []))


def test_qpy_parameters_are_reconciled_by_name(tmp_path):
    p = Parameter("angle")
    qc = QuantumCircuit(2)
    qc.rx(p, 0)
    qc.cx(0, 1)
    path = tmp_path / "saved.qpy"
    with path.open("wb") as f:
        qpy.dump(qc, f)
    with path.open("rb") as f:
        loaded = qpy.load(f)[0]
    fresh = Parameter("angle")
    assert fresh != next(iter(loaded.parameters))
    compiled = _compile_job(PreparedProjectedFeatureJob(loaded, (fresh,), [], []))
    assert compiled.columns == (0,)
    bound = compiled.circuit.assign_parameters({next(iter(compiled.circuit.parameters)): .7})
    np.testing.assert_allclose(Statevector(bound).probabilities(), Statevector(qc.assign_parameters({p: .7})).probabilities())


def test_pauli_parser_preserves_endianness_y_phase_and_coefficients():
    assert _pauli_terms(SparsePauliOp.from_list([("YZX", -.3)]), 3) == ((5, 6, 1, -.3),)
    assert _pauli_terms(SparsePauliOp.from_list([("YYY", 2.)]), 3) == ((7, 7, 3, 2.),)
    with pytest.raises(ValueError, match="Hermitian"):
        _pauli_terms(SparsePauliOp.from_list([("X", 1j)]), 1)
    with pytest.raises(ValueError, match="width"):
        _pauli_terms(SparsePauliOp("XX"), 1)


def test_dispatch_rejects_unsupported_execution_before_preparation():
    with pytest.raises(ValueError):
        execute_prepared_projected_feature_job(None, np.empty((1, 0)), None, None, framework="tilelang")
    with patch("pqfmlib.core.execution.prepare_projected_feature_job") as prepare:
        with pytest.raises(ValueError):
            run_projected_feature_job(None, [], np.empty((1, 0)), None, None, None, [], [], framework="tilelang")
    prepare.assert_not_called()


@pytest.mark.parametrize("k_max,n_columns", [(2, 10), (3, 14), (4, 15)])
def test_cd_ising_multilocal_metadata_and_csv(k_max, n_columns, tmp_path):
    X = np.random.default_rng(29).normal(size=(6, 4))
    model = CDIsingProjectiveQFM("multilocal", q_enc=4, measure_all_zz=True, k_max=k_max,
                               expectation_method="statevector", output_root=str(tmp_path))
    expected = model.fit_transform(X)
    assert expected.shape == (len(X), n_columns)
    assert len(model.obs_metadata) == n_columns
    model.load_data = lambda: None
    model.X_q_all, model.y = X, np.arange(len(X)) % 2
    result = model.run()
    table = pd.read_csv(result["csv_path"])
    assert table.columns[:4].tolist() == [f"q1_z_{i}" for i in range(4)]
    assert table.columns[4] == "q2_z_0_1"
    if k_max >= 3:
        assert "q3_z_0_1_2" in table.columns
    if k_max == 4:
        assert "q4_z_0_1_2_3" in table.columns
    np.testing.assert_allclose(table.drop(columns="y").to_numpy(), expected, atol=1e-12)


def prepared(circuit, params=None, obs=None):
    if obs is None:
        obs = [SparsePauliOp.from_list([("".join(word), 1.)])
               for word in itertools.product("IXYZ", repeat=circuit.num_qubits)]
    return PreparedProjectedFeatureJob(circuit, tuple(circuit.parameters if params is None else params),
                                      [[o] for o in obs], [o.to_list() for o in obs])


def exact(job, rows):
    positions = {p.name: i for i, p in enumerate(job.param_order)}
    return np.array([[Statevector(job.qc_t.assign_parameters({p: row[positions[p.name]] for p in job.qc_t.parameters}))
                      .expectation_value(group[0]).real for group in job.obs_isa_broadcast] for row in rows])


@gpu
@pytest.mark.parametrize("kernel_backend", ["auto", "optimized"])
def test_real_kernels_all_pauli_words_and_general_gate_matrices(kernel_backend):
    qc = QuantumCircuit(3)
    qc.unitary(random_unitary(8, seed=13), [2, 0, 1])
    qc.cx(2, 0)
    qc.swap(1, 0)
    p = ParameterVector("p", 12)
    qc.ry(p[10] * p[2] + .17, 2)
    qc.rxx(p[2], 2, 0)
    job = prepared(qc, (p[10], p[2]))
    obs = SparsePauliOp.from_list([("III", .3), ("XYZ", -.7), ("YYY", .2)])
    job = replace(job, obs_isa_broadcast=job.obs_isa_broadcast + [[obs]])
    rows = np.array([[.3, -.4], [1.7, .9]])
    executor = TileLangExecutor(kernel_backend=kernel_backend)
    actual, meta = executor.execute(job, rows)
    np.testing.assert_allclose(actual, exact(job, rows), atol=1e-11, rtol=1e-11)
    assert meta is job.obs_metadata
    cached = executor._compiled
    np.testing.assert_allclose(executor.execute(job, rows[:1])[0], actual[:1], atol=1e-12)
    assert executor._compiled is cached
    assert executor.last_execution_stats["pauli_calls"] > 0


@pytest.fixture
def native_graph(tmp_path):
    backend = GenericBackendV2(5, seed=27, basis_gates=["id", "rz", "sx", "x", "cx"],
                              coupling_map=[[i, j] for i in range(5) for j in range(5) if abs(i-j) == 1])
    path = tmp_path / "topology.json"
    save_topology_snapshot(backend, path)
    nodes = tmp_path / "nodes.json"
    nodes.write_text("[2, 1, 3]")
    return path, nodes, backend


@gpu
@pytest.mark.parametrize("map_type", MAPS)
@pytest.mark.parametrize("native", [False, True])
def test_all_maps_exact_native_and_qpy_roundtrip(map_type, native, native_graph, tmp_path):
    path, nodes, _backend = native_graph
    kwargs = dict(name_file="tilelang", q_enc=3, seed=11)
    X = np.random.default_rng(29).normal(size=(6, 4))
    if map_type is XYZProjectiveQFM:
        kwargs.update(features_per_qubit=1, axes=("z", "x", "y"), encoding_mode="shared_feature",
                      measure_cross_observables=True, m=2)
    elif map_type is CDIsingProjectiveQFM:
        kwargs.update(measure_all_zz=True, k_max=3, m=2)
    else:
        kwargs.update(measure_all_pairs=True, alpha=.9, R=3)
    if map_type is SuperexchangeHeisenbergPQFM:
        X = X[:, :3]
        kwargs.update(J0=1.3, ell=.6)
    if native:
        kwargs.update(ideal=False, topology="native", topology_file=str(path),
                      use_fixed_phys_nodes=True, fixed_phys_nodes_file=str(nodes))
    reference = map_type(**kwargs, expectation_method="statevector")
    expected = reference.fit_transform(X)
    actual_map = map_type(**kwargs, framework="tilelang", expectation_method="exact")
    actual = actual_map.fit_transform(X)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert actual_map.obs_metadata == reference.obs_metadata
    np.testing.assert_allclose(actual_map.transform(X[:1]), expected[:1], atol=1e-10)
    qpy_path = tmp_path / "circuit.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(actual_map._prepared_execution.qc_t, f)
    restored = map_type(**kwargs, framework="tilelang", expectation_method="exact", fixed_circuit_file_name=str(qpy_path))
    np.testing.assert_allclose(restored.fit_transform(X), expected, atol=1e-10, rtol=1e-10)
    assert restored._prepared_execution.fixed_circuit_loaded
    actual_map.set_params(precision="single")
    with pytest.raises(RuntimeError, match="configuration changed"):
        actual_map.transform(X)


@gpu
@pytest.mark.parametrize("map_type", MAPS)
def test_routed_physical_qpy_and_exports(map_type, native_graph, tmp_path):
    graph_path, nodes, backend = native_graph
    X = np.random.default_rng(29).normal(size=(6, 4))
    if map_type is SuperexchangeHeisenbergPQFM:
        X = X[:, :3]
    kwargs = dict(name_file="physical", q_enc=3, ideal=False, topology="native", topology_file=str(graph_path),
                  use_fixed_phys_nodes=True, fixed_phys_nodes_file=str(nodes), output_root=str(tmp_path))
    if map_type is XYZProjectiveQFM:
        kwargs.update(features_per_qubit=1, axes=("x", "z"), encoding_mode="shared_feature")
    elif map_type is CDIsingProjectiveQFM:
        kwargs.update(measure_all_zz=True, k_max=3)
    reference = map_type(**kwargs, expectation_method="statevector").fit(X)
    expected = reference.transform(X)
    if isinstance(reference, HeisenbergProjectiveQFM):
        circuit = reference._build_circuit_and_observables()[0]
    else:
        circuit = reference._prepared_execution.qc_t
    physical = transpile(circuit, backend=backend, initial_layout=[2, 1, 3], optimization_level=1, seed_transpiler=42)
    file = tmp_path / "physical.qpy"
    with file.open("wb") as f:
        qpy.dump(physical, f)
    loaded = map_type(**kwargs, framework="tilelang", expectation_method="statevector", fixed_circuit_file_name=str(file))
    np.testing.assert_allclose(loaded.fit_transform(X), expected, atol=1e-10, rtol=1e-10)
    assert loaded._prepared_execution.qc_t.num_qubits == 5
    loaded.load_data = lambda: None
    loaded.X_q_all, loaded.y = X, np.arange(len(X)) % 2
    result = loaded.run()
    np.testing.assert_allclose(result["Xq_all_raw"], expected, atol=1e-10)
    assert Path(result["csv_path"]).exists()
    np.testing.assert_allclose(np.load(result["npy_path"]), expected, atol=1e-10)


@gpu
@pytest.mark.parametrize("mode", ["shared_feature", "multi_axis"])
@pytest.mark.parametrize("axes", [("x",), ("y", "z"), ("z", "x", "y")])
@pytest.mark.parametrize("diagonal,cross", [(True, False), (False, True), (True, True)])
def test_xyz_native_encodings_axes_and_interaction_terms(mode, axes, diagonal, cross, native_graph):
    path, nodes, _backend = native_graph
    X = np.random.default_rng(31).normal(size=(6, 4))
    kwargs = dict(name_file="xyz_options", q_enc=3, ideal=False, topology="native", topology_file=str(path),
                  use_fixed_phys_nodes=True, fixed_phys_nodes_file=str(nodes), m=2, tau=.3,
                  features_per_qubit=1 if mode == "shared_feature" else len(axes),
                  encoding_mode=mode, axes=axes, keep_diagonal_terms=diagonal,
                  keep_cross_terms=cross, measure_cross_observables=True)
    J = np.full((4, 4), .3)
    np.fill_diagonal(J, 1.)
    with patch("pqfmlib.maps.xyz.mutual_information_matrix", return_value=J):
        reference = XYZProjectiveQFM(**kwargs, expectation_method="statevector").fit_transform(X)
        actual = XYZProjectiveQFM(**kwargs, framework="tilelang", expectation_method="exact").fit_transform(X)
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)


@gpu
@pytest.mark.parametrize("mode", ["shared_feature", "multi_axis"])
@pytest.mark.parametrize("axes", [("x",), ("y", "z"), ("z", "x", "y")])
@pytest.mark.parametrize("topology", ["chain", "full"])
def test_heisenberg_axes_multiblock_scaling_and_observables(mode, axes, topology):
    X = np.random.default_rng(19).normal(size=(2, 7))
    kwargs = dict(name_file="heis_options", q_enc=3, topology=topology, interaction_axes=axes,
                  encoding_mode=mode, R=3, alpha=-.4, add_barriers=False, use_tanh_scaling=False,
                  measure_2local_diagonal=True, measure_cross_observables=True)
    reference = HeisenbergProjectiveQFM(**kwargs, expectation_method="statevector").fit_transform(X)
    actual = HeisenbergProjectiveQFM(**kwargs, framework="tilelang", expectation_method="exact").fit_transform(X)
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)


@gpu
def test_empty_batches_invalid_shapes_and_memory_guard():
    job = prepared(QuantumCircuit(2))
    executor = TileLangExecutor()
    assert executor.execute(job, np.empty((0, 0)))[0].shape == (0, 16)
    for bad in (np.empty((1, 1)), np.empty((1,)), np.array([[np.nan]])):
        with pytest.raises(ValueError):
            executor.execute(job, bad)
    wide = QuantumCircuit(40)
    np.testing.assert_allclose(executor.execute(
        prepared(wide, obs=[SparsePauliOp("Z" + "I" * 39)]), np.empty((1, 0)))[0], [[1.]])
    with pytest.raises(MemoryError, match="available GPU memory"):
        TileLangExecutor(strategy="reference").execute(
            prepared(wide, obs=[SparsePauliOp("Z" + "I" * 39)]), np.empty((1, 0)))


def test_fused_matrix_batches_preserve_order_endianness_and_nonlinear_angles():
    from qiskit.quantum_info import Operator
    from pqfmlib.core.tilelang_plan import matrix_batch
    p = ParameterVector("p", 12)
    qc = QuantumCircuit(4)
    qc.h(3)
    qc.cx(3, 0)
    qc.ry(p[10] * p[2], 1)
    qc.rzz(.7 * p[2] - .3 * p[10] + .2, 0, 1)
    qc.rxx(.1, 0, 1)
    qc.ryy(.4, 1, 0)
    qc.cx(2, 1)
    job = prepared(qc, (p[10], p[2]), obs=[SparsePauliOp("ZYXI")])
    compiled = _compile_job(job)
    rows = np.array([[.1, .7], [-.8, 1.2], [.2, -.5]])
    constants = {}
    matrices = matrix_batch(compiled.matrix_keys, rows, constants)
    for index, values in enumerate(rows):
        actual = QuantumCircuit(4)
        for gate in compiled.gates:
            d = 1 << len(gate.sites)
            actual.unitary(matrices[gate.matrix_id, index, :d, :d], list(gate.sites))
        assert Operator(actual).equiv(Operator(qc.assign_parameters(dict(zip(job.param_order, values)))), atol=1e-12)
    again = matrix_batch(compiled.matrix_keys, rows[::-1], constants)
    np.testing.assert_allclose(again, matrices[:, ::-1], atol=1e-12)
    assert len(compiled.gates) < len(compiled.circuit.data)


@pytest.mark.parametrize("options", [dict(batch_size=0), dict(batch_size=True), dict(batch_size=1.2),
                                    dict(strategy="unknown"), dict(matrix_backend="unknown"),
                                    dict(kernel_backend="unknown")])
def test_invalid_optimization_options_do_not_import_runtime(options):
    with patch("pqfmlib.core.tilelang_execution._load_runtime") as load:
        with pytest.raises(ValueError):
            TileLangExecutor(**options)
    load.assert_not_called()


@gpu
@pytest.mark.parametrize("strategy", ["auto", "full", "cones", "reference"])
def test_optimization_strategies_share_exact_results(strategy):
    p = Parameter("theta")
    qc = QuantumCircuit(8)
    for a, b in ((0, 1), (3, 2), (5, 7)):
        qc.h(a)
        qc.rxx(p, a, b)
        qc.ry(p * p, b)
    operators = [SparsePauliOp(s) for s in ("IIIIIIIZ", "IIIIZIXX", "YIXIIIII", "IZIIIIIZ")]
    job = prepared(qc, obs=operators)
    rows = np.linspace(-.7, .8, 7).reshape(-1, 1)
    executor = TileLangExecutor(strategy=strategy, batch_size=3, profile=True)
    actual, _ = executor.execute(job, rows)
    np.testing.assert_allclose(actual, exact(job, rows), atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(executor.execute(job, rows[::-1])[0], actual[::-1], atol=1e-11)
    if strategy != "reference":
        assert executor.last_execution_stats["n_batches"] == 3
        assert executor.last_execution_stats["reduction_seconds"] >= 0
        assert executor.last_execution_stats["gate_calls"] < 7 * len(qc.data)


@gpu
def test_cones_ignore_idle_physical_qubits_and_use_complete_multilocal_support():
    qc = QuantumCircuit(127)
    qc.h(0)
    qc.cx(0, 126)
    obs = []
    for a, b in (("X", "X"), ("Y", "Y"), ("Z", "Z"), ("X", "Z")):
        label = ["I"] * 127
        label[0], label[-1] = a, b
        obs.append(SparsePauliOp("".join(label)))
    executor = TileLangExecutor()
    actual, _ = executor.execute(prepared(qc, obs=obs), np.empty((2, 0)))
    np.testing.assert_allclose(actual, [[1., -1., 1., 0.]] * 2, atol=1e-12)
    assert executor.last_execution_stats["max_cone_qubits"] == 2
    assert executor.last_execution_stats["source_qubits"] == 127


@gpu
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("batch", [1, 7])
def test_gpu_matrix_plan_matches_cpu_for_ordered_affine_fusions(precision, batch):
    from pqfmlib.core.tilelang_matrices import MatrixBuilder
    from pqfmlib.core.tilelang_plan import matrix_batch
    p = ParameterVector("angles", 3)
    qc = QuantumCircuit(4)
    for repeat in range(4):
        for a, b in ((0, 1), (3, 2), (1, 2), (2, 0)):
            qc.u(.3 + p[0] * 1.7 - p[2] * .2, p[1], -.7, b)
            qc.cx(b, a)
            qc.rxx(p[0] - p[1], a, b)
            qc.ryy(p[2] + repeat * .1, b, a)
            qc.rzz(-p[2] * 2, a, b)
            qc.u(p[2], -.9, p[1] * .2, a)
    job = _compile_job(prepared(qc, tuple(reversed(p))))
    rows = np.random.default_rng(102).uniform(-3, 3, (batch, 3))
    if batch > 1:
        rows = np.asfortranarray(rows)
    executor = TileLangExecutor(precision=precision)
    builder = MatrixBuilder(executor._torch, executor._kernels, job.matrix_keys)
    expected = matrix_batch(job.matrix_keys, rows, {})
    with patch("pqfmlib.core.tilelang_matrices.matrix_batch", side_effect=AssertionError("affine CPU fallback")):
        actual = builder.build(rows, executor._dtype, {}, executor._stage)
        metadata = builder.device
        reverse = builder.build(rows[::-1].copy(), executor._dtype, {}, executor._stage)
    assert builder.device is metadata
    actual = executor._torch.view_as_complex(actual).cpu().numpy()
    reverse = executor._torch.view_as_complex(reverse).cpu().numpy()
    tolerance = 1e-12 if precision == "double" else 2e-7
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(reverse, expected[:, ::-1], atol=tolerance, rtol=tolerance)


@gpu
@pytest.mark.parametrize("matrix_backend", ["cpu", "gpu", "auto"])
def test_matrix_backends_nonlinear_qpy_and_metadata_invalidation(matrix_backend):
    p = Parameter("angle")
    qc = QuantumCircuit(4)
    qc.u(p + .1, p / 3, -.2, 0)
    qc.cx(0, 1)
    qc.ry(p * p, 3)
    qc.cx(3, 2)
    job = prepared(qc)
    rows = np.linspace(-1, 1, 5).reshape(-1, 1)
    executor = TileLangExecutor(matrix_backend=matrix_backend, batch_size=3)
    np.testing.assert_allclose(executor.execute(job, rows)[0], exact(job, rows), atol=1e-11)
    assert executor.last_execution_stats["matrix_backend"] == ("cpu" if matrix_backend == "cpu" else "hybrid")
    first_builder = executor._matrix_builder
    first_metadata = executor._cone_metadata
    # Exercise an identical cone under alternating exact and sampled semantics.
    executor.execute(job, rows, expectation_method="shots", shots=513)
    np.testing.assert_allclose(executor.execute(job, rows)[0], exact(job, rows), atol=1e-11)
    np.testing.assert_allclose(executor.execute(job, rows[::-1])[0], exact(job, rows[::-1]), atol=1e-11)
    other = QuantumCircuit(4)
    other.rx(p, 1)
    other.cx(1, 3)
    other_job = prepared(other)
    np.testing.assert_allclose(executor.execute(other_job, rows)[0], exact(other_job, rows), atol=1e-11)
    assert executor._cone_metadata is not first_metadata
    if matrix_backend != "cpu":
        assert executor._matrix_builder is not first_builder


@gpu
@pytest.mark.parametrize("qubits", [1, 2, 7, 12])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_grouped_gate_and_popcount_reduction_match_qiskit(qubits, precision):
    p = Parameter("angle")
    qc = QuantumCircuit(qubits)
    for q in range(qubits):
        qc.u(p * (q + 1), -.3, .2, q)
    if qubits > 1:
        for a, b in ((0, qubits - 1), (qubits - 1, 0)):
            qc.rxx(p * .2, a, b)
            qc.ryy(.8, b, a)
            qc.cx(a, b)
        for q in range(1, qubits):
            qc.rzz(p * .4, q, q - 1)
    rng = np.random.default_rng(93)
    labels = ["I" * qubits, "Y" * qubits, "Z" * qubits, "X" * qubits]
    labels += ["".join(rng.choice(list("IXYZ"), qubits)) for _ in range(5)]
    obs = [SparsePauliOp(label) for label in labels]
    obs.append(SparsePauliOp.from_list([(labels[1], -.3), (labels[-1], .4), (labels[0], .2)]))
    job = prepared(qc, obs=obs)
    rows = np.linspace(-.9, 1.1, 5).reshape(-1, 1)
    expected = exact(job, rows)
    tolerance = 2e-5 if precision == "single" else 1e-11
    for backend in ("baseline", "optimized", "auto"):
        executor = TileLangExecutor(precision=precision, kernel_backend=backend, batch_size=2)
        result = executor.execute(job, rows)[0]
        np.testing.assert_allclose(result, expected, atol=tolerance, rtol=tolerance)
        np.testing.assert_allclose(executor.execute(job, rows[::-1])[0], expected[::-1], atol=tolerance, rtol=tolerance)
        if backend == "optimized":
            assert executor.last_execution_stats["grouped_gate_calls"] > 0
            assert executor.last_execution_stats["fast_pauli_calls"] > 0
        sampled = executor.execute(job, rows, expectation_method="shots", shots=1025)[0]
        np.testing.assert_allclose(sampled, expected, atol=6 / np.sqrt(1025), rtol=0)
