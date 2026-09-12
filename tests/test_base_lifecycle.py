import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.providers.fake_provider import GenericBackendV2

import pqfmlib
from pqfmlib.core.base import BaseProjectiveQFM
from pqfmlib.core.data import load_tabular_dataset, mutual_information_matrix, validate_numeric_dataframe


class FakeBackendQFM(BaseProjectiveQFM):
    def _load_real_backend(self):
        self.real_backend = GenericBackendV2(num_qubits=max(2, int(self.q_enc)))

    def run(self):
        return None


class BaseLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.qfm = BaseProjectiveQFM(name_file="in_memory")

    def test_public_version_matches_project_version(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)

        self.assertIsNotNone(match)
        self.assertEqual(pqfmlib.__version__, match.group(1))

    def test_fit_input_is_copied_and_records_feature_structure(self):
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        prepared = self.qfm._prepare_fit_input(X)
        X.iloc[0, 0] = 99.0

        np.testing.assert_array_equal(prepared, [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(self.qfm.n_features_in_, 2)
        np.testing.assert_array_equal(self.qfm.feature_names_in_, ["a", "b"])
        self.assertFalse(self.qfm._is_fitted)

    def test_transform_requires_fitted_state(self):
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            self.qfm._prepare_transform_input([[1.0, 2.0]], ("config",))

    def test_transform_reuses_feature_contract_and_config_signature(self):
        self.qfm._prepare_fit_input(pd.DataFrame({"a": [1.0], "b": [2.0]}))
        self.qfm._mark_fitted(("map", 2))

        transformed = self.qfm._prepare_transform_input(
            pd.DataFrame({"a": [3.0], "b": [4.0]}),
            ("map", 2),
        )

        np.testing.assert_array_equal(transformed, [[3.0, 4.0]])
        with self.assertRaisesRegex(ValueError, "names and order"):
            self.qfm._prepare_transform_input(
                pd.DataFrame({"b": [4.0], "a": [3.0]}),
                ("map", 2),
            )
        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            self.qfm._prepare_transform_input(
                pd.DataFrame({"a": [3.0], "b": [4.0]}),
                ("map", 3),
            )

    def test_transform_rejects_different_feature_count(self):
        self.qfm._prepare_fit_input([[1.0, 2.0]])
        self.qfm._mark_fitted(("config",))

        with self.assertRaisesRegex(ValueError, "fitted with 2 features"):
            self.qfm._prepare_transform_input([[1.0, 2.0, 3.0]], ("config",))

    def test_input_must_be_numeric_finite_nonempty_and_2d(self):
        invalid_inputs = (
            [1.0, 2.0],
            np.empty((0, 2)),
            np.empty((2, 0)),
            [[1.0, np.inf]],
            [[1.0, np.nan]],
            [["not", "numeric"]],
        )

        for X in invalid_inputs:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    self.qfm._prepare_fit_input(X)

    def test_numeric_dataframe_rejects_non_finite_values(self):
        dataframe = pd.DataFrame({"finite": [1.0, 2.0], "invalid": [3.0, np.inf]})

        with self.assertRaisesRegex(ValueError, "non-finite.*invalid"):
            validate_numeric_dataframe(dataframe)

    def test_tabular_dataset_requires_feature_and_target_columns(self):
        with tempfile.TemporaryDirectory() as data_dir:
            pd.DataFrame({"target": [0.0, 1.0]}).to_csv(
                Path(data_dir) / "target_only.csv",
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "feature column"):
                load_tabular_dataset("target_only", data_dir)

    def test_mutual_information_requires_finite_2d_features(self):
        invalid_inputs = (
            ([1.0, 2.0], "2D feature matrix"),
            (np.empty((3, 0)), "at least one feature"),
            (np.array([[1.0, np.inf], [2.0, 3.0]]), "only finite values"),
        )

        for X, message in invalid_inputs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    mutual_information_matrix(X)

    def test_aer_statevector_max_qubits_matches_q_enc(self):
        qfm = BaseProjectiveQFM(name_file="statevector", ideal=True, q_enc=37)

        qfm.setup_backend_and_estimator()

        self.assertEqual(qfm.backend.num_qubits, 37)

    def test_aer_mps_max_qubits_matches_q_enc(self):
        qfm = BaseProjectiveQFM(name_file="mps", ideal=True, q_enc=80, mps=True)

        qfm.setup_backend_and_estimator()

        self.assertEqual(qfm.backend.num_qubits, 80)

    def test_fakebackend_aer_max_qubits_covers_q_enc(self):
        qfm = FakeBackendQFM(
            name_file="fakebackend",
            ideal=False,
            simulation=True,
            fakebackend=True,
            q_enc=7,
        )

        qfm.setup_backend_and_estimator()

        self.assertGreaterEqual(qfm.backend.num_qubits, 7)


if __name__ == "__main__":
    unittest.main()
