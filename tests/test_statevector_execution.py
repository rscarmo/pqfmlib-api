"""Deterministic tests for exact PQFM execution (no credentials or QPU)."""
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from sklearn.base import clone

from pqfmlib import CDIsingProjectiveQFM, HeisenbergProjectiveQFM, XYZProjectiveQFM
from pqfmlib.core.execution import prepare_projected_feature_job, execute_prepared_projected_feature_job
from pqfmlib.core.observables import pauli_observables_axis_pairs


class StatevectorTests(unittest.TestCase):
    def test_one_evolution_per_sample_and_cross_observables(self):
        a, b = Parameter('a'), Parameter('b')
        circuit = QuantumCircuit(2)
        circuit.ry(a, 0)
        circuit.rx(b, 1)
        circuit.cx(0, 1)
        obs, metadata = pauli_observables_axis_pairs(2, measure_cross_observables=True)
        backend = AerSimulator(method='statevector', device='CPU')
        prepared = prepare_projected_feature_job(circuit, [b, a], [0, 1], backend,
                                                 obs, metadata, simulation=True, base_folder=None)
        theta = np.array([[0.3, 0.7], [0.8, 1.2], [0., 0.]])
        estimator = Mock()
        with patch.object(backend, 'run', wraps=backend.run) as run:
            actual, returned_metadata = execute_prepared_projected_feature_job(
                prepared, theta, backend, estimator, simulation=True,
                expectation_method='statevector', shots=999,
            )
            self.assertEqual(run.call_count, len(theta))
        expected = np.array([
            [Statevector.from_instruction(circuit.assign_parameters({b: row[0], a: row[1]})).expectation_value(o).real for o in obs]
            for row in theta
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertEqual(returned_metadata, metadata)
        estimator.run.assert_not_called()
        again, _ = execute_prepared_projected_feature_job(
            prepared, theta, backend, estimator, simulation=True,
            expectation_method='statevector', shots=1,
        )
        np.testing.assert_array_equal(actual, again)

    def test_validation_clone_and_set_params(self):
        for map_type in (CDIsingProjectiveQFM, XYZProjectiveQFM, HeisenbergProjectiveQFM):
            for kwargs in ({'simulation': False}, {'fakebackend': True}, {'mps': True},
                           {'expectation_method': 'unknown'}):
                with self.subTest(map=map_type.__name__, kwargs=kwargs), self.assertRaises(ValueError):
                    map_type(name_file='test', q_enc=2,
                             **({'expectation_method': 'statevector'} | kwargs))
            for ideal in (True, False):
                qfm = map_type(name_file='test', q_enc=2, ideal=ideal,
                              expectation_method='statevector')
                self.assertEqual(clone(qfm).expectation_method, 'statevector')
                # Backend setup without loading any remote service.
                with patch.object(qfm, '_load_real_backend'):
                    qfm.setup_backend_and_estimator()
                self.assertEqual(qfm.backend.options.method, 'statevector')
                self.assertIsNone(qfm.estimator)
                qfm.set_params(fakebackend=True)
                with self.assertRaises(ValueError):
                    qfm.setup_backend_and_estimator()

    def test_all_maps_transform_and_legacy_execution_agree(self):
        X = np.random.default_rng(42).normal(size=(8, 2))
        for map_type in (CDIsingProjectiveQFM, XYZProjectiveQFM, HeisenbergProjectiveQFM):
            with self.subTest(map=map_type.__name__), tempfile.TemporaryDirectory() as tmp:
                qfm = map_type(name_file='exact', q_enc=3 if map_type is HeisenbergProjectiveQFM else 2,
                              expectation_method='statevector', output_root=tmp)
                actual = qfm.fit_transform(X)
                np.testing.assert_allclose(qfm.transform(X), actual, atol=1e-12)
                self.assertIsNone(qfm.estimator)
                qfm.X_q_all = X
                qfm.base_folder = tmp
                qfm.execute_quantum_feature_map()
                np.testing.assert_allclose(qfm.Xq_all_raw, actual, atol=1e-12)
                qfm.set_params(expectation_method='shots')
                with self.assertRaises(RuntimeError):
                    qfm.transform(X)

    def test_nonunitary_circuit_rejected(self):
        circuit = QuantumCircuit(2)
        circuit.reset(0)
        obs, meta = pauli_observables_axis_pairs(2)
        backend = AerSimulator(method='statevector')
        prepared = prepare_projected_feature_job(circuit, [], [0, 1], backend,
                                                 obs, meta, simulation=True, base_folder=None)
        with self.assertRaises(ValueError):
            execute_prepared_projected_feature_job(prepared, np.empty((1, 0)), backend, None,
                                                   simulation=True, expectation_method='statevector')


if __name__ == '__main__':
    unittest.main()
