"""Native connectivity without noise, credentials, or quantum hardware jobs."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from qiskit import qpy
from qiskit.providers.fake_provider import GenericBackendV2
from sklearn.base import clone

from pqfmlib import CDIsingProjectiveQFM, HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM, XYZProjectiveQFM
from pqfmlib.hardware.coupling import get_coupling_edges_and_costs
from pqfmlib.hardware.topology import load_topology_snapshot, save_topology_snapshot


MAP_TYPES = (CDIsingProjectiveQFM, XYZProjectiveQFM, HeisenbergProjectiveQFM)


class NativeSimulationTests(unittest.TestCase):
    X = np.array([[.1, .4, .2, .8, .5, .9], [.2, .1, .5, .7, .8, .3],
                  [.8, .6, .4, .2, .9, .1], [.7, .9, .3, .1, .2, .6]])
    specs = (
        (CDIsingProjectiveQFM, {}),
        (XYZProjectiveQFM, dict(features_per_qubit=1, axes=("x", "z"), encoding_mode="shared_feature")),
        (XYZProjectiveQFM, dict(features_per_qubit=2, axes=("x", "y"), encoding_mode="multi_axis")),
        (HeisenbergProjectiveQFM, dict(measure_2local_diagonal=True, measure_cross_observables=True)),
    )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = GenericBackendV2(
            num_qubits=5, basis_gates=["id", "rz", "sx", "x", "cx"], seed=23,
            coupling_map=[[4, 1], [1, 4], [1, 3], [3, 1], [3, 0], [0, 3], [0, 2], [2, 0]],
        )
        self.snapshot = self.root / "topology.json"
        save_topology_snapshot(self.backend, self.snapshot)
        self.nodes = self.root / "nodes.json"
        self.nodes.write_text(json.dumps([1, 4, 3]), encoding="utf-8")
        # Use a deterministic interaction matrix; this test concerns topology,
        # rather than finite-sample mutual-information estimation.
        self.J = np.full((6, 6), .3)
        np.fill_diagonal(self.J, 1.)
        stack = ExitStack()
        self.addCleanup(stack.close)
        for module in ("cd_ising", "xyz"):
            stack.enter_context(patch(f"pqfmlib.maps.{module}.mutual_information_matrix", return_value=self.J))

    def options(self):
        return dict(name_file="native", q_enc=3, ideal=False, expectation_method="statevector",
                    use_fixed_phys_nodes=True, fixed_phys_nodes_file=str(self.nodes),
                    output_root=str(self.root / "unused"), use_gpu_statevector=False)

    def test_snapshot_roundtrip_preserves_order_and_costs(self):
        self.assertEqual(load_topology_snapshot(self.snapshot), get_coupling_edges_and_costs(self.backend))

    def test_offline_native_matches_legacy_blocks_observables_and_features(self):
        for map_type, extra in self.specs:
            with self.subTest(map=map_type.__name__, extra=extra):
                options = self.options() | extra
                legacy_options = {}
                if map_type is HeisenbergProjectiveQFM:
                    legacy_options["topology"] = "native"
                legacy = map_type(**options, **legacy_options)
                legacy.real_backend = self.backend
                expected = legacy.fit_transform(self.X)
                offline = map_type(**options, topology="native", topology_file=str(self.snapshot))
                with patch.object(offline, "_load_real_backend", side_effect=AssertionError("network access")):
                    actual = offline.fit_transform(self.X)
                    np.testing.assert_allclose(offline.transform(self.X[:1]), expected[:1], atol=1e-12, rtol=1e-12)
                self.assertEqual(offline.phys_nodes, [1, 4, 3])
                self.assertEqual(offline.edges_act, [(0, 1), (0, 2)])
                self.assertEqual(offline.obs_metadata, legacy.obs_metadata)
                if map_type is not HeisenbergProjectiveQFM:
                    self.assertEqual(offline.blocks, legacy.blocks)
                    self.assertEqual(offline.edges_log, legacy.edges_log)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
                self.assertEqual(offline._prepared_execution.qc_t.num_qubits, 3)
                self.assertIsNone(offline.real_backend)
                self.assertIsNone(offline.backend.options.noise_model)
                self.assertFalse((self.root / "unused").exists())

    def test_live_metadata_and_snapshot_choose_identical_automatic_subgraph(self):
        for map_type in MAP_TYPES:
            with self.subTest(map=map_type.__name__):
                options = self.options() | dict(use_fixed_phys_nodes=False, topology="native", use_edge_error=True)
                if map_type is HeisenbergProjectiveQFM:
                    options.pop("use_edge_error")
                live = map_type(**options)
                live.real_backend = self.backend
                expected = live.fit_transform(self.X)
                offline = map_type(**options, topology_file=str(self.snapshot))
                with patch.object(offline, "_load_real_backend", side_effect=AssertionError("network access")):
                    actual = offline.fit_transform(self.X)
                self.assertEqual(offline.phys_nodes, live.phys_nodes)
                self.assertEqual(offline.edges_act, live.edges_act)
                self.assertEqual(offline.obs_metadata, live.obs_metadata)
                if map_type is not HeisenbergProjectiveQFM:
                    self.assertEqual(offline.blocks, live.blocks)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_native_observable_scope_respects_explicit_all_pairs(self):
        for map_type in (CDIsingProjectiveQFM, XYZProjectiveQFM):
            for all_pairs in (False, True):
                with self.subTest(map=map_type.__name__, all_pairs=all_pairs):
                    qfm = map_type(**self.options(), topology="native", topology_file=str(self.snapshot),
                                   measure_all_zz=all_pairs)
                    qfm.fit(self.X)
                    pairs = {tuple(item[1:]) for item in qfm.obs_metadata if len(item) == 3}
                    self.assertEqual(pairs, {(0, 1), (0, 2), (1, 2)} if all_pairs else {(0, 1), (0, 2)})

    def test_cudaq_prepares_native_circuit_without_aer_or_backend_lookup(self):
        for map_type in MAP_TYPES:
            with self.subTest(map=map_type.__name__):
                qfm = map_type(**(self.options() | dict(framework="cudaq", expectation_method="exact")),
                               topology="native", topology_file=str(self.snapshot))
                with patch("pqfmlib.core.cudaq_execution.CudaqExecutor") as executor:
                    with patch.object(qfm, "_make_aer_backend", side_effect=AssertionError("Aer fallback")):
                        with patch.object(qfm, "_load_real_backend", side_effect=AssertionError("network access")):
                            qfm.fit(self.X)
                executor.assert_called_once()
                self.assertIs(qfm.backend, executor.return_value)
                self.assertIsNone(qfm.estimator)
                self.assertEqual(qfm.edges_act, [(0, 1), (0, 2)])
                self.assertEqual(qfm._prepared_execution.qc_t.num_qubits, 3)

    def test_run_matches_transform_without_contacting_backend(self):
        for map_type in MAP_TYPES:
            with self.subTest(map=map_type.__name__):
                qfm = map_type(**(self.options() | dict(output_root=str(self.root))), topology="native",
                               topology_file=str(self.snapshot))
                with patch.object(qfm, "_load_real_backend", side_effect=AssertionError("network access")):
                    expected = qfm.fit_transform(self.X)
                    qfm.load_data = Mock()
                    qfm.X_q_all, qfm.y = self.X, np.array([0, 1, 0, 1])
                    result = qfm.run()
                np.testing.assert_allclose(result["Xq_all_raw"], expected, atol=1e-12, rtol=1e-12)
                self.assertIn("native", Path(result["base_folder"]).name)
                self.assertTrue(Path(result["csv_path"]).exists())

    def test_clone_and_fitted_configuration_include_topology(self):
        for map_type in MAP_TYPES:
            for param, value in (("topology", "full"), ("topology_file", "other.json")):
                with self.subTest(map=map_type.__name__, param=param):
                    qfm = map_type(**self.options(), topology="native", topology_file=str(self.snapshot))
                    self.assertEqual(clone(qfm).get_params(), qfm.get_params())
                    qfm.fit(self.X).set_params(**{param: value})
                    with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                        qfm.transform(self.X)

    def test_topology_modes_are_validated_before_backend_creation(self):
        for map_type in MAP_TYPES:
            for overrides in (dict(topology="other"), dict(topology="full", ideal=False),
                              dict(topology_file=str(self.snapshot)),
                              dict(topology="native"),
                              dict(topology="native", topology_file=str(self.snapshot), ideal=True),
                              dict(topology="native", topology_file=str(self.snapshot), ideal=False, simulation=False),
                              dict(topology="native", topology_file=str(self.snapshot), ideal=False, fakebackend=True)):
                with self.subTest(map=map_type.__name__, overrides=overrides), self.assertRaises(ValueError):
                    map_type(name_file="invalid", **overrides)
            qfm = map_type(name_file="invalid").set_params(topology="other")
            with patch.object(qfm, "_load_real_backend") as load_backend:
                with self.assertRaises(ValueError):
                    qfm.setup_backend_and_estimator()
                load_backend.assert_not_called()

    def test_invalid_fixed_nodes_fail_without_backend_lookup(self):
        for nodes in ([1, 1, 3], [4, 0, 2], [1, 3, 9], [1, 3]):
            self.nodes.write_text(json.dumps(nodes), encoding="utf-8")
            for map_type in MAP_TYPES:
                with self.subTest(map=map_type.__name__, nodes=nodes):
                    qfm = map_type(**self.options(), topology="native", topology_file=str(self.snapshot))
                    with patch.object(qfm, "_load_real_backend", side_effect=AssertionError("network access")):
                        with self.assertRaises(ValueError):
                            qfm.fit(self.X)

    def test_default_ideal_connectivity_and_heisenberg_chain_are_preserved(self):
        for map_type in MAP_TYPES + (SuperexchangeHeisenbergPQFM,):
            with self.subTest(map=map_type.__name__):
                qfm = map_type(name_file="default", q_enc=3, expectation_method="statevector")
                X = self.X[:, :3] if map_type is SuperexchangeHeisenbergPQFM else self.X
                qfm.fit(X)
                expected = [(0, 1), (1, 2)] if isinstance(qfm, HeisenbergProjectiveQFM) else [(0, 1), (0, 2), (1, 2)]
                self.assertEqual(qfm.edges_act, expected)
                with self.assertRaises(ValueError):
                    map_type(name_file="invalid", ideal=True, topology="native")

    def test_superexchange_native_snapshot_and_logical_qpy_match_backend_metadata(self):
        X = self.X[:, :3]
        options = self.options() | dict(topology="native", measure_all_pairs=True, J0=1.3, ell=.6)
        legacy = SuperexchangeHeisenbergPQFM(**options)
        legacy.real_backend = self.backend
        expected = legacy.fit_transform(X)
        offline = SuperexchangeHeisenbergPQFM(**options, topology_file=str(self.snapshot))
        with patch.object(offline, "_load_real_backend", side_effect=AssertionError("network access")):
            actual = offline.fit_transform(X)
        self.assertEqual(offline.phys_nodes, legacy.phys_nodes)
        self.assertEqual(offline.edges_act, legacy.edges_act)
        self.assertEqual(offline.obs_metadata, legacy.obs_metadata)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        circuit, _, _, _ = offline._build_circuit_and_observables()
        path = self.root / "native_superexchange.qpy"
        with path.open("wb") as stream:
            qpy.dump(circuit, stream)
        loaded = SuperexchangeHeisenbergPQFM(**options, topology_file=str(self.snapshot),
                                            fixed_circuit_file_name=str(path))
        with patch.object(loaded, "_load_real_backend", side_effect=AssertionError("network access")):
            np.testing.assert_allclose(loaded.fit_transform(X), expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(loaded._prepared_execution.fixed_circuit_loaded)

    def test_cudaq_native_rejects_noise_hardware_and_resource_estimation_before_loading(self):
        for map_type in MAP_TYPES + (SuperexchangeHeisenbergPQFM,):
            for overrides in (dict(fakebackend=True), dict(simulation=False), dict(resource_estimation=True)):
                with self.subTest(map=map_type.__name__, overrides=overrides):
                    with patch("pqfmlib.core.cudaq_execution._load_cudaq") as load_cudaq:
                        with self.assertRaises(ValueError):
                            map_type("invalid", ideal=False, topology="native", framework="cudaq",
                                     expectation_method="exact", **overrides)
                    load_cudaq.assert_not_called()


class TopologySnapshotValidationTests(unittest.TestCase):
    def test_minimal_graph_without_costs(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "graph.json"
            path.write_text(json.dumps(dict(schema_version=1, coupling_edges=[[9, 2], [2, 5]])))
            self.assertEqual(load_topology_snapshot(path), ([(9, 2), (2, 5)], {(9, 2): 1., (2, 5): 1.}))

    def test_invalid_snapshot_is_rejected(self):
        valid = dict(schema_version=1, coupling_edges=[[0, 1]])
        invalid = [[], {}, valid | dict(schema_version=True), valid | dict(schema_version=2),
                   valid | dict(coupling_edges=[]), valid | dict(coupling_edges=[[0, 0]]),
                   valid | dict(coupling_edges=[[True, 1]]), valid | dict(coupling_edges=[[-1, 1]]),
                   valid | dict(coupling_edges=[[0, 1], [0, 1]]),
                   valid | dict(edge_costs={}), valid | dict(edge_costs=[[0, 2, .1]]),
                   valid | dict(edge_costs=[[0, 1, -.1]]), valid | dict(edge_costs=[[0, 1, 1.1]]),
                   valid | dict(edge_costs=[[0, 1, float("nan")]]),
                   valid | dict(edge_costs=[[0, 1, True]]),
                   valid | dict(edge_costs=[[0, 1, .1], [0, 1, .2]])]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "graph.json"
            for data in invalid:
                with self.subTest(data=data):
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        load_topology_snapshot(path)


if __name__ == "__main__":
    unittest.main()
