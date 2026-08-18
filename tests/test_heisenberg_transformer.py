import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

import numpy as np
from qiskit import qpy

from pqfmlib import HeisenbergProjectiveQFM
from pqfmlib.maps.heisenberg import build_heisenberg_chain_feature_circuit


class HeisenbergTransformerTests(unittest.TestCase):
    @staticmethod
    def _mock_backend_setup(qfm):
        def setup():
            qfm.backend = sentinel.backend
            qfm.estimator = sentinel.estimator

        qfm.setup_backend_and_estimator = Mock(side_effect=setup)

    @staticmethod
    def _execute_side_effect(_prepared, theta_values, *_args, **_kwargs):
        theta_values = np.asarray(theta_values, dtype=float)
        first_column = theta_values[:, :1]
        return np.hstack([first_column, first_column + 3.0]), [("z", 0)]

    def test_fit_prepares_only_dimensional_state_and_transform_reuses_it(self):
        X_train = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        X_val = np.array([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
        X_test = np.array([[1.6, 1.7, 1.8]])

        with tempfile.TemporaryDirectory() as output_root:
            qfm = HeisenbergProjectiveQFM(
                name_file="in_memory",
                output_root=output_root,
                ideal=True,
                simulation=True,
                q_enc=3,
            )
            self._mock_backend_setup(qfm)
            with patch(
                "pqfmlib.core.data.mutual_information_matrix",
            ) as mutual_information:
                with patch.object(
                    qfm,
                    "_transpile_or_load_circuit",
                    side_effect=lambda circuit: circuit,
                ) as prepare_circuit:
                    with patch(
                        "pqfmlib.maps.heisenberg.execute_prepared_projected_feature_job",
                        side_effect=self._execute_side_effect,
                    ) as execute:
                        fitted = qfm.fit(X_train)
                        fitted_theta_info = copy.deepcopy(qfm.theta_info)
                        fitted_circuit_info = copy.deepcopy(qfm.circuit_info)
                        fitted_metadata = copy.deepcopy(qfm.obs_metadata)
                        prepared_circuit = qfm._prepared_execution.qc_t

                        Xq_val = qfm.transform(X_val)
                        Xq_test = qfm.transform(X_test)

        self.assertIs(fitted, qfm)
        self.assertTrue(qfm._is_fitted)
        self.assertEqual(qfm.n_features_in_, 3)
        self.assertIsNone(qfm.X_q_all)
        mutual_information.assert_not_called()
        prepare_circuit.assert_called_once()
        self.assertEqual(execute.call_count, 2)
        self.assertTrue(all(call.args[0] is qfm._prepared_execution for call in execute.call_args_list))
        self.assertIs(qfm._prepared_execution.qc_t, prepared_circuit)
        self.assertEqual(qfm.theta_info, fitted_theta_info)
        self.assertEqual(qfm.circuit_info, fitted_circuit_info)
        self.assertEqual(qfm.obs_metadata, fitted_metadata)
        self.assertEqual(qfm.theta_info["features_per_block"], 2)
        self.assertEqual(qfm.theta_info["num_blocks"], 2)
        self.assertEqual(qfm.theta_info["total_slots"], 4)
        self.assertAlmostEqual(qfm.circuit_info["phi_scale"], qfm.alpha / 4.0)
        expected_val = 2.0 * np.pi * np.tanh(X_val[:, :1] / 3.0)
        expected_test = 2.0 * np.pi * np.tanh(X_test[:, :1] / 3.0)
        np.testing.assert_allclose(Xq_val, np.hstack([expected_val, expected_val + 3.0]))
        np.testing.assert_allclose(Xq_test, np.hstack([expected_test, expected_test + 3.0]))
        val_theta = execute.call_args_list[0].args[1]
        self.assertEqual(val_theta.shape, (2, 4))
        np.testing.assert_array_equal(val_theta[:, -1], [0.0, 0.0])

    def test_fit_transform_matches_fit_followed_by_transform(self):
        X = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        with tempfile.TemporaryDirectory() as output_root:
            direct = HeisenbergProjectiveQFM(name_file="direct", output_root=output_root, q_enc=3)
            separate = HeisenbergProjectiveQFM(name_file="separate", output_root=output_root, q_enc=3)
            self._mock_backend_setup(direct)
            self._mock_backend_setup(separate)
            with patch.object(direct, "_transpile_or_load_circuit", side_effect=lambda circuit: circuit):
                with patch.object(separate, "_transpile_or_load_circuit", side_effect=lambda circuit: circuit):
                    with patch(
                        "pqfmlib.maps.heisenberg.execute_prepared_projected_feature_job",
                        side_effect=self._execute_side_effect,
                    ):
                        Xq_direct = direct.fit_transform(X)
                        separate.fit(X)
                        Xq_separate = separate.transform(X)

        np.testing.assert_array_equal(Xq_direct, Xq_separate)

    def test_hardware_aware_simulation_reuses_fixed_physical_nodes_and_circuit(self):
        X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        X_val = np.array([[0.7, 0.8, 0.9]])

        with tempfile.TemporaryDirectory() as folder:
            phys_nodes_file = Path(folder) / "phys_nodes.json"
            phys_nodes_file.write_text(json.dumps([5, 6, 7]), encoding="utf-8")
            qfm = HeisenbergProjectiveQFM(
                name_file="hardware_aware",
                output_root=folder,
                ideal=False,
                simulation=True,
                fakebackend=False,
                q_enc=3,
                use_fixed_phys_nodes=True,
                fixed_phys_nodes_file=str(phys_nodes_file),
            )
            self._mock_backend_setup(qfm)
            qfm.real_backend = sentinel.real_backend
            with patch(
                "pqfmlib.maps.heisenberg.get_coupling_edges_and_costs",
                return_value=([(5, 6), (6, 7)], {(5, 6): 0.01, (6, 7): 0.02}),
            ):
                with patch.object(
                    qfm,
                    "_transpile_or_load_circuit",
                    side_effect=lambda circuit: circuit,
                ) as prepare_circuit:
                    with patch(
                        "pqfmlib.maps.heisenberg.execute_prepared_projected_feature_job",
                        side_effect=self._execute_side_effect,
                    ) as execute:
                        qfm.fit(X_train)
                        qfm.transform(X_val)
                        qfm.transform(X_val)

        prepare_circuit.assert_called_once()
        self.assertEqual(qfm.phys_nodes, [5, 6, 7])
        self.assertEqual(execute.call_count, 2)
        self.assertTrue(all(call.args[0] is qfm._prepared_execution for call in execute.call_args_list))

    def test_public_transformer_api_rejects_real_qpu_mode(self):
        qfm = HeisenbergProjectiveQFM(
            name_file="real_qpu",
            ideal=False,
            simulation=False,
            q_enc=3,
        )

        with self.assertRaisesRegex(NotImplementedError, "simulation=True"):
            qfm.fit([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    def test_fakebackend_fit_extracts_and_freezes_transpiler_layout(self):
        X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        transpiled_circuit = Mock(layout=sentinel.layout)
        observable = Mock()
        observable.apply_layout.return_value = sentinel.laid_out_observable

        with tempfile.TemporaryDirectory() as output_root:
            qfm = HeisenbergProjectiveQFM(
                name_file="fakebackend",
                output_root=output_root,
                ideal=False,
                simulation=True,
                fakebackend=True,
                q_enc=3,
            )
            self._mock_backend_setup(qfm)
            qfm.real_backend = sentinel.real_backend
            with patch.object(
                qfm,
                "_transpile_or_load_circuit",
                return_value=transpiled_circuit,
            ) as prepare_circuit:
                with patch("pqfmlib.maps.heisenberg.validate_circuit_parameter_compatibility") as validate_circuit:
                    with patch(
                        "pqfmlib.maps.heisenberg._extract_initial_phys_nodes",
                        return_value=[5, 6, 7],
                    ) as extract_nodes:
                        with patch(
                            "pqfmlib.maps.heisenberg.heisenberg_chain_observables",
                            return_value=([observable], [("z", 0)]),
                        ):
                            qfm.fit(X_train)

        prepare_circuit.assert_called_once()
        validate_circuit.assert_called_once()
        extract_nodes.assert_called_once()
        observable.apply_layout.assert_called_once_with(sentinel.layout)
        self.assertEqual(qfm.phys_nodes, [5, 6, 7])
        self.assertEqual(qfm._prepared_execution.obs_isa_broadcast, [[sentinel.laid_out_observable]])

    def test_fixed_qpy_circuit_is_loaded_and_validated_during_fit(self):
        X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        fixed_circuit, _param_order, _circuit_info = build_heisenberg_chain_feature_circuit(
            3,
            num_blocks=2,
            features_per_block=2,
            R=2,
            alpha=0.1,
            seed=42,
            add_barriers=True,
        )

        with tempfile.TemporaryDirectory() as folder:
            fixed_path = Path(folder) / "heisenberg_fixed.qpy"
            with fixed_path.open("wb") as stream:
                qpy.dump(fixed_circuit, stream)
            qfm = HeisenbergProjectiveQFM(
                name_file="fixed_circuit",
                output_root=folder,
                ideal=True,
                simulation=True,
                q_enc=3,
                fixed_circuit_file_name=str(fixed_path),
            )
            self._mock_backend_setup(qfm)
            qfm.fit(X_train)

        self.assertTrue(qfm._prepared_execution.fixed_circuit_loaded)
        self.assertEqual(
            {parameter.name for parameter in qfm._prepared_execution.qc_t.parameters},
            {f"theta[{index}]" for index in range(4)},
        )


if __name__ == "__main__":
    unittest.main()
