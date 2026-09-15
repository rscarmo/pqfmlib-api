# Changelog

## 0.2.0

- Added opt-in exact statevector expectations to all three PQFMs for noiseless simulation, reusing one state per sample across observables.
- Preserved shot-based execution as the default and added validation for incompatible execution options.
- Updated the hybrid Breast Cancer example with manual cross-validation, reusable fold features, and six-component PCA.
- Moved statevector and notebook regression tests into `tests/`, keeping `examples/` for examples.

## 0.1.1

- Fixed Qiskit observable label ordering across all projected feature maps.
- Configured Aer simulator qubit capacity from `q_enc`.
- Hardened edge-case handling for tabular data, CD-Ising, and Heisenberg.
- Ensured fixed QPY circuits are honored and validated during legacy simulated runs.
- Exposed `pqfmlib.__version__` from the public package API.
- Added regression tests for version consistency and corrected behavior.
