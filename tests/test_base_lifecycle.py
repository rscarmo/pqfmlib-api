import unittest

import numpy as np
import pandas as pd

from pqfmlib.core.base import BaseProjectiveQFM


class BaseLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.qfm = BaseProjectiveQFM(name_file="in_memory")

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


if __name__ == "__main__":
    unittest.main()
