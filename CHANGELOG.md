# Changelog

## 0.1.1

- Fixed Qiskit observable label ordering across all projected feature maps.
- Configured Aer simulator qubit capacity from `q_enc`.
- Hardened edge-case handling for tabular data, CD-Ising, and Heisenberg.
- Ensured fixed QPY circuits are honored and validated during legacy simulated runs.
- Exposed `pqfmlib.__version__` from the public package API.
- Added regression tests for version consistency and corrected behavior.
