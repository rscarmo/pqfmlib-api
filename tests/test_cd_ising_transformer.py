import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

import numpy as np

from pqfmlib import CDIsingProjectiveQFM


class CDIsingTransformerTests(unittest.TestCase):
    @staticmethod
    def _interaction_matrix():
        return np.array(
            [
                [0.0, 0.8, 0.4],
                [0.8, 0.0, 0.2],
                [0.4, 0.2, 0.0],
            ]
        )

    @staticmethod
    def _mock_backend_setup(qfm):
        def setup():
            qfm.backend = sentinel.backend
            qfm.estimator = sentinel.estimator

        qfm.setup_backend_and_estimator = Mock(side_effect=setup)

    @staticmethod
    def _theta_side_effect(X, blocks, *_args):
        values = np.asarray(X, dtype=float)
        return values[:, :1].copy(), len(blocks)

    @staticmethod
    def _execute_side_effect(_prepared, theta_values, *_args, **_kwargs):
        theta_values = np.asarray(theta_values, dtype=float)
        return np.hstack([theta_values, theta_values + 1.0]), [("z", 0)]

    def test_fit_learns_from_train_and_transforms_reuse_frozen_state(self):
        X_train = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ]
        )
        X_val = np.array([[2.0, 2.1, 2.2], [2.3, 2.4, 2.5]])
        X_test = np.array([[3.0, 3.1, 3.2]])

        with tempfile.TemporaryDirectory() as output_root:
            qfm = CDIsingProjectiveQFM(
                name_file="in_memory",
                output_root=output_root,
                ideal=True,
                simulation=True,
                q_enc=2,
            )
            self._mock_backend_setup(qfm)
            with patch(
                "pqfmlib.maps.cd_ising.mutual_information_matrix",
                return_value=self._interaction_matrix(),
            ) as mutual_information:
                with patch(
                    "pqfmlib.maps.cd_ising.prepare_projected_feature_job",
                    return_value=sentinel.prepared_execution,
                ) as prepare_execution:
                    with patch(
                        "pqfmlib.maps.cd_ising.make_theta_matrix_all_blocks_hw",
                        side_effect=self._theta_side_effect,
                    ) as make_theta:
                        with patch(
                            "pqfmlib.maps.cd_ising.execute_prepared_projected_feature_job",
                            side_effect=self._execute_side_effect,
                        ) as execute:
                            fitted = qfm.fit(X_train)
                            fitted_J = qfm.J.copy()
                            fitted_blocks = copy.deepcopy(qfm.blocks)
                            fitted_phys_nodes = list(qfm.phys_nodes)
                            fitted_edges = list(qfm.edges_log)

                            Xq_val = qfm.transform(X_val)
                            Xq_test = qfm.transform(X_test)

        self.assertIs(fitted, qfm)
        self.assertTrue(qfm._is_fitted)
        self.assertEqual(qfm.n_features_in_, 3)
        self.assertIsNone(qfm.X_q_all)
        mutual_information.assert_called_once()
        np.testing.assert_array_equal(mutual_information.call_args.args[0], X_train)
        prepare_execution.assert_called_once()
        self.assertTrue(prepare_execution.call_args.kwargs["use_fixed_circuit_in_simulation"])
        self.assertTrue(prepare_execution.call_args.kwargs["validate_parameter_compatibility"])
        self.assertEqual(make_theta.call_count, 2)
        np.testing.assert_array_equal(make_theta.call_args_list[0].args[0], X_val)
        np.testing.assert_array_equal(make_theta.call_args_list[1].args[0], X_test)
        self.assertEqual(execute.call_count, 2)
        self.assertTrue(all(call.args[0] is sentinel.prepared_execution for call in execute.call_args_list))
        np.testing.assert_array_equal(qfm.J, fitted_J)
        self.assertEqual(qfm.blocks, fitted_blocks)
        self.assertEqual(qfm.phys_nodes, fitted_phys_nodes)
        self.assertEqual(qfm.edges_log, fitted_edges)
        np.testing.assert_array_equal(Xq_val, [[2.0, 3.0], [2.3, 3.3]])
        np.testing.assert_array_equal(Xq_test, [[3.0, 4.0]])

    def test_fit_transform_matches_fit_followed_by_transform(self):
        X = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ]
        )

        with tempfile.TemporaryDirectory() as output_root:
            direct = CDIsingProjectiveQFM(name_file="direct", output_root=output_root, q_enc=2)
            separate = CDIsingProjectiveQFM(name_file="separate", output_root=output_root, q_enc=2)
            self._mock_backend_setup(direct)
            self._mock_backend_setup(separate)
            with patch(
                "pqfmlib.maps.cd_ising.mutual_information_matrix",
                return_value=self._interaction_matrix(),
            ):
                with patch(
                    "pqfmlib.maps.cd_ising.prepare_projected_feature_job",
                    return_value=sentinel.prepared_execution,
                ):
                    with patch(
                        "pqfmlib.maps.cd_ising.make_theta_matrix_all_blocks_hw",
                        side_effect=self._theta_side_effect,
                    ):
                        with patch(
                            "pqfmlib.maps.cd_ising.execute_prepared_projected_feature_job",
                            side_effect=self._execute_side_effect,
                        ):
                            Xq_direct = direct.fit_transform(X)
                            separate.fit(X)
                            Xq_separate = separate.transform(X)

        np.testing.assert_array_equal(Xq_direct, Xq_separate)

    def test_hardware_aware_simulation_freezes_nodes_edges_and_prepared_circuit(self):
        X_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        X_val = np.array([[0.7, 0.8]])
        J = np.array([[0.0, 0.5], [0.5, 0.0]])

        with tempfile.TemporaryDirectory() as output_root:
            qfm = CDIsingProjectiveQFM(
                name_file="hardware_aware",
                output_root=output_root,
                ideal=False,
                simulation=True,
                fakebackend=False,
                q_enc=2,
            )
            self._mock_backend_setup(qfm)
            qfm.real_backend = sentinel.real_backend
            with patch("pqfmlib.maps.cd_ising.mutual_information_matrix", return_value=J):
                with patch(
                    "pqfmlib.maps.cd_ising.get_coupling_edges_and_costs",
                    return_value=([(5, 6)], {(5, 6): 0.01}),
                ):
                    with patch(
                        "pqfmlib.maps.cd_ising.pick_connected_subset_greedy",
                        return_value=[5, 6],
                    ) as pick_nodes:
                        with patch(
                            "pqfmlib.maps.cd_ising.GA.GA_assignment",
                            return_value=([0, 1], 0.0),
                        ) as assign_features:
                            with patch(
                                "pqfmlib.maps.cd_ising.prepare_projected_feature_job",
                                return_value=sentinel.hardware_prepared,
                            ) as prepare_execution:
                                with patch(
                                    "pqfmlib.maps.cd_ising.make_theta_matrix_all_blocks_hw",
                                    side_effect=self._theta_side_effect,
                                ):
                                    with patch(
                                        "pqfmlib.maps.cd_ising.execute_prepared_projected_feature_job",
                                        side_effect=self._execute_side_effect,
                                    ) as execute:
                                        qfm.fit(X_train)
                                        qfm.transform(X_val)
                                        qfm.transform(X_val)

        pick_nodes.assert_called_once()
        assign_features.assert_called_once()
        prepare_execution.assert_called_once()
        self.assertEqual(qfm.phys_nodes, [5, 6])
        self.assertEqual(qfm.edges_log, [(0, 1)])
        self.assertEqual(qfm.edges_act, [(0, 1)])
        self.assertTrue(all(call.args[0] is sentinel.hardware_prepared for call in execute.call_args_list))

    def test_public_transformer_api_rejects_real_qpu_mode(self):
        qfm = CDIsingProjectiveQFM(
            name_file="real_qpu",
            ideal=False,
            simulation=False,
            q_enc=2,
        )

        with self.assertRaisesRegex(NotImplementedError, "simulation=True"):
            qfm.fit([[0.1, 0.2], [0.3, 0.4]])

    def test_fixed_blocks_are_loaded_without_recomputing_mutual_information(self):
        X_train = np.array([[0.1, 0.2], [0.3, 0.4]])
        blocks_payload = [
            {
                "feat_ids": [0, 1],
                "J_dict_layer": [{"i": 0, "j": 1, "valor": 0.5}],
            }
        ]

        with tempfile.TemporaryDirectory() as folder:
            blocks_file = Path(folder) / "blocks.json"
            blocks_file.write_text(json.dumps(blocks_payload), encoding="utf-8")
            qfm = CDIsingProjectiveQFM(
                name_file="fixed_blocks",
                output_root=folder,
                ideal=True,
                simulation=True,
                q_enc=2,
                use_fixed_blocks=True,
                fixed_blocks_file=str(blocks_file),
            )
            self._mock_backend_setup(qfm)
            with patch("pqfmlib.maps.cd_ising.mutual_information_matrix") as mutual_information:
                with patch(
                    "pqfmlib.maps.cd_ising.prepare_projected_feature_job",
                    return_value=sentinel.fixed_prepared,
                ):
                    qfm.fit(X_train)

        mutual_information.assert_not_called()
        self.assertIsNone(qfm.J)
        self.assertEqual(qfm.blocks[0]["feat_ids"], [0, 1])
        self.assertEqual(qfm.blocks[0]["J_dict_layer"], {(0, 1): 0.5})


if __name__ == "__main__":
    unittest.main()
