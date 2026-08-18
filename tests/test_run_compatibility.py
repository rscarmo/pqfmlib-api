import unittest
from unittest.mock import Mock

import numpy as np

from pqfmlib import CDIsingProjectiveQFM, HeisenbergProjectiveQFM, XYZProjectiveQFM


class RunCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _record_pipeline(qfm, method_names):
        calls = Mock()
        for method_name in method_names:
            method = Mock(name=method_name)
            setattr(qfm, method_name, method)
            calls.attach_mock(method, method_name)
        return calls

    def test_cd_ising_run_keeps_pipeline_and_result_contract(self):
        qfm = CDIsingProjectiveQFM(name_file="dataset")
        calls = self._record_pipeline(
            qfm,
            (
                "load_data",
                "setup_backend_and_estimator",
                "prepare_output_folder",
                "compute_global_J_and_feature_blocks",
                "prepare_blocks_and_edges",
                "execute_quantum_feature_map",
                "save_quantum_features",
            ),
        )
        qfm.Xq_all_raw = np.array([[0.1]])
        qfm.pairs_2q = []
        qfm.csv_output_path = "features.csv"
        qfm.npy_output_path = "features.npy"
        qfm.base_folder = "output"

        result = qfm.run()

        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            [
                "load_data",
                "setup_backend_and_estimator",
                "prepare_output_folder",
                "compute_global_J_and_feature_blocks",
                "prepare_blocks_and_edges",
                "execute_quantum_feature_map",
                "save_quantum_features",
            ],
        )
        self.assertIs(result["Xq_all_raw"], qfm.Xq_all_raw)
        self.assertEqual(result["pairs_2q"], [])
        self.assertEqual(result["csv_path"], "features.csv")

    def test_xyz_run_keeps_pipeline_and_result_contract(self):
        qfm = XYZProjectiveQFM(name_file="dataset")
        calls = self._record_pipeline(
            qfm,
            (
                "load_data",
                "setup_backend_and_estimator",
                "prepare_output_folder",
                "compute_global_J_and_feature_blocks",
                "prepare_blocks_and_edges",
                "execute_quantum_feature_map",
                "save_quantum_features",
            ),
        )
        qfm.Xq_all_raw = np.array([[0.2]])
        qfm.obs_metadata = [("z", 0)]
        qfm.csv_output_path = "features.csv"
        qfm.npy_output_path = "features.npy"
        qfm.base_folder = "output"

        result = qfm.run()

        self.assertEqual(len(calls.mock_calls), 7)
        self.assertIs(result["Xq_all_raw"], qfm.Xq_all_raw)
        self.assertIs(result["obs_metadata"], qfm.obs_metadata)
        self.assertEqual(result["encoding_mode"], "multi_axis")

    def test_heisenberg_run_keeps_pipeline_and_result_contract(self):
        qfm = HeisenbergProjectiveQFM(name_file="dataset")
        calls = self._record_pipeline(
            qfm,
            (
                "load_data",
                "setup_backend_and_estimator",
                "prepare_output_folder",
                "prepare_layout",
                "execute_quantum_feature_map",
                "save_quantum_features",
            ),
        )
        qfm.Xq_all_raw = np.array([[0.3]])
        qfm.obs_metadata = [("z", 0)]
        qfm.csv_output_path = "features.csv"
        qfm.npy_output_path = "features.npy"
        qfm.base_folder = "output"
        qfm.theta_info = {"num_blocks": 1}
        qfm.circuit_info = {"R": 2}

        result = qfm.run()

        self.assertEqual(len(calls.mock_calls), 6)
        self.assertIs(result["Xq_all_raw"], qfm.Xq_all_raw)
        self.assertIs(result["theta_info"], qfm.theta_info)
        self.assertIs(result["circuit_info"], qfm.circuit_info)


if __name__ == "__main__":
    unittest.main()
