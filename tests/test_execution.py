import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from pqfmlib.core.execution import (
    execute_prepared_projected_feature_job,
    prepare_projected_feature_job,
    run_projected_feature_job,
)


class _SimulationJob:
    def __init__(self, evs):
        self._evs = evs

    def result(self):
        return [SimpleNamespace(data=SimpleNamespace(evs=self._evs))]


class ExecutionPreparationTests(unittest.TestCase):
    def _circuit_and_inputs(self):
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)
        observable = SparsePauliOp.from_list([("Z", 1.0)])
        return circuit, [theta], [observable], [("z", 0)]

    def test_prepared_simulation_can_execute_multiple_batches_without_repreparing(self):
        circuit, param_order, observables, metadata = self._circuit_and_inputs()
        estimator = Mock()
        estimator.run.side_effect = [
            _SimulationJob(np.array([[0.1, 0.2]])),
            _SimulationJob(np.array([[0.3]])),
        ]

        with tempfile.TemporaryDirectory() as folder:
            prepared = prepare_projected_feature_job(
                circuit,
                param_order,
                [0],
                backend=None,
                obs_list=observables,
                obs_metadata=metadata,
                simulation=True,
                base_folder=folder,
            )
            prepared_circuit = prepared.qc_t
            first = execute_prepared_projected_feature_job(
                prepared,
                np.array([[0.0], [1.0]]),
                backend=None,
                estimator=estimator,
                simulation=True,
                base_folder=folder,
            )
            second = execute_prepared_projected_feature_job(
                prepared,
                np.array([[2.0]]),
                backend=None,
                estimator=estimator,
                simulation=True,
                base_folder=folder,
            )

        self.assertIs(prepared.qc_t, prepared_circuit)
        self.assertEqual(estimator.run.call_count, 2)
        np.testing.assert_array_equal(first[0], [[0.1], [0.2]])
        np.testing.assert_array_equal(second[0], [[0.3]])
        self.assertIs(first[1], metadata)

    def test_legacy_wrapper_preserves_pure_simulation_behavior(self):
        circuit, param_order, observables, metadata = self._circuit_and_inputs()
        estimator = Mock()
        estimator.run.return_value = _SimulationJob(np.array([[0.5, 0.6]]))

        with tempfile.TemporaryDirectory() as folder:
            missing_fixed_circuit = str(Path(folder) / "does_not_exist.qpy")
            result = run_projected_feature_job(
                circuit,
                param_order,
                np.array([[0.0], [1.0]]),
                [0],
                backend=None,
                estimator=estimator,
                obs_list=observables,
                obs_metadata=metadata,
                simulation=True,
                fakebackend=False,
                base_folder=folder,
                fixed_circuit=missing_fixed_circuit,
            )

        np.testing.assert_array_equal(result[0], [[0.5], [0.6]])
        self.assertIs(result[1], metadata)

    def test_legacy_wrapper_delegates_to_prepare_and_execute(self):
        circuit, param_order, observables, metadata = self._circuit_and_inputs()
        prepared = object()
        expected = (np.array([[0.25]]), metadata)

        with patch("pqfmlib.core.execution.prepare_projected_feature_job", return_value=prepared) as prepare:
            with patch("pqfmlib.core.execution.execute_prepared_projected_feature_job", return_value=expected) as execute:
                result = run_projected_feature_job(
                    circuit,
                    param_order,
                    np.array([[0.0]]),
                    [0],
                    backend="backend",
                    estimator="estimator",
                    obs_list=observables,
                    obs_metadata=metadata,
                    simulation=True,
                )

        self.assertIs(result, expected)
        prepare.assert_called_once()
        execute.assert_called_once()

    def test_theta_matrix_must_be_two_dimensional(self):
        circuit, param_order, observables, metadata = self._circuit_and_inputs()
        with tempfile.TemporaryDirectory() as folder:
            prepared = prepare_projected_feature_job(
                circuit,
                param_order,
                [0],
                backend=None,
                obs_list=observables,
                obs_metadata=metadata,
                simulation=True,
                base_folder=folder,
            )
            with self.assertRaisesRegex(ValueError, "2D matrix"):
                execute_prepared_projected_feature_job(
                    prepared,
                    np.array([0.0]),
                    backend=None,
                    estimator=Mock(),
                    simulation=True,
                    base_folder=folder,
                )

    def test_opt_in_fixed_circuit_is_loaded_during_simulation(self):
        circuit, param_order, observables, metadata = self._circuit_and_inputs()

        with tempfile.TemporaryDirectory() as folder:
            fixed_path = Path(folder) / "fixed.qpy"
            with fixed_path.open("wb") as stream:
                qpy.dump(circuit, stream)
            prepared = prepare_projected_feature_job(
                circuit,
                param_order,
                [0],
                backend=None,
                obs_list=observables,
                obs_metadata=metadata,
                simulation=True,
                base_folder=folder,
                fixed_circuit=str(fixed_path),
                use_fixed_circuit_in_simulation=True,
                validate_parameter_compatibility=True,
            )

        self.assertTrue(prepared.fixed_circuit_loaded)
        self.assertEqual([parameter.name for parameter in prepared.qc_t.parameters], ["theta"])

    def test_incompatible_fixed_circuit_is_rejected_during_preparation(self):
        circuit, param_order, observables, metadata = self._circuit_and_inputs()
        incompatible = QuantumCircuit(1)
        incompatible.ry(Parameter("different_parameter"), 0)

        with tempfile.TemporaryDirectory() as folder:
            fixed_path = Path(folder) / "incompatible.qpy"
            with fixed_path.open("wb") as stream:
                qpy.dump(incompatible, stream)
            with self.assertRaisesRegex(ValueError, "incompatible with the fitted map"):
                prepare_projected_feature_job(
                    circuit,
                    param_order,
                    [0],
                    backend=None,
                    obs_list=observables,
                    obs_metadata=metadata,
                    simulation=True,
                    base_folder=folder,
                    fixed_circuit=str(fixed_path),
                    use_fixed_circuit_in_simulation=True,
                    validate_parameter_compatibility=True,
                )


if __name__ == "__main__":
    unittest.main()
