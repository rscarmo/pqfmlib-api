"""FP32/FP64 TileLang simulation with exact or finite-shot expectations."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from importlib import import_module
from time import perf_counter
from numbers import Integral

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp

from pqfmlib.core.execution import validate_circuit_parameter_compatibility
from pqfmlib.core.framework import validate_precision, validate_shots
from pqfmlib.core.tilelang_plan import causal_cones, cone_work, full_cone, fuse_gates, local_masks, matrix_batch, sampling_plan
from pqfmlib.core.tilelang_sampling import SHOT_CHUNK, sample_cone
from pqfmlib.core.tilelang_matrices import MatrixBuilder


_BASIS = ["u", "cx", "rxx", "ryy", "rzz"]


def _validate_unitary_circuit(circuit):
    if circuit.num_clbits:
        raise ValueError("TileLang requires a unitary circuit without classical bits.")
    for instruction in circuit.data:
        gate = instruction.operation
        if (gate.name in {"measure", "reset", "initialize", "kraus", "superop", "delay"}
                or getattr(gate, "condition", None) is not None or getattr(gate, "blocks", None)):
            raise ValueError(f"TileLang does not support circuit instruction {gate.name!r}.")


def _pauli_terms(observable, qubits):
    if not isinstance(observable, SparsePauliOp) or observable.num_qubits != qubits:
        raise ValueError("TileLang requires SparsePauliOp observables matching the prepared circuit width.")
    terms = []
    for label, coefficient in observable.simplify(atol=0).to_list():
        coefficient = complex(coefficient)
        if not np.isfinite(coefficient) or abs(coefficient.imag) > 1e-12:
            raise ValueError("TileLang requires finite, Hermitian Pauli observables.")
        x_mask, z_mask, ny = 0, 0, 0
        for q, axis in enumerate(reversed(label)):
            if axis in "XY":
                x_mask |= 1 << q
            if axis in "ZY":
                z_mask |= 1 << q
            ny += axis == "Y"
        terms.append((x_mask, z_mask, ny % 4, coefficient.real))
    return tuple(terms)


@dataclass(frozen=True)
class _CompiledJob:
    circuit: object
    columns: tuple
    observable_terms: tuple
    gates: tuple
    matrix_keys: tuple
    terms: tuple
    observable_indices: tuple
    cones: tuple
    sampling_groups: tuple
    sampling_supports: tuple
    sampling_cones: tuple


def _compile_job(prepared):
    validate_circuit_parameter_compatibility(prepared.qc_t, prepared.param_order)
    _validate_unitary_circuit(prepared.qc_t)
    try:
        circuit = transpile(prepared.qc_t, basis_gates=_BASIS, optimization_level=0)
    except Exception as exc:
        raise ValueError("Circuit cannot be lowered to the TileLang unitary gate basis.") from exc
    _validate_unitary_circuit(circuit)
    positions = {p.name: i for i, p in enumerate(prepared.param_order)}
    # QPY preserves parameter names, not identity with a newly constructed map.
    columns = tuple(positions[p.name] for p in circuit.parameters)
    observables = []
    for group in prepared.obs_isa_broadcast:
        if len(group) != 1:
            raise ValueError("TileLang expects one operator per observable group.")
        observables.append(_pauli_terms(group[0], circuit.num_qubits))
    gates, keys = fuse_gates(circuit, prepared.param_order)
    unique, indices, observables_indexed = [], {}, []
    for terms in observables:
        indexed = []
        for x, z, phase, coefficient in terms:
            if coefficient == 0:
                continue
            key = (x, z, phase)
            if key not in indices:
                indices[key] = len(unique)
                unique.append(key)
            indexed.append((indices[key], coefficient))
        observables_indexed.append(tuple(indexed))
    cones = causal_cones(circuit.num_qubits, gates, unique)
    groups, supports, shot_cones = sampling_plan(circuit.num_qubits, gates, unique)
    return _CompiledJob(circuit, columns, tuple(observables), gates, keys, tuple(unique),
                        tuple(observables_indexed), cones, groups, supports, shot_cones)


def _load_runtime():
    try:
        torch = import_module("torch")
        kernels = import_module("pqfmlib.core._tilelang_kernels")
    except ImportError as exc:
        raise ImportError(
            "framework='tilelang' requires optional dependencies; install pqfmlib[tilelang] "
            "and a compatible CUDA compiler/runtime."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("TileLang requires an accessible NVIDIA CUDA GPU; no CPU fallback is available.")
    return torch, kernels


class TileLangExecutor:
    """Exact GPU execution; strategy='reference' retains the phase-2 baseline.

    ``auto`` chooses full or reduced states by estimated amplitude/gate work.
    ``full`` and ``cones`` are diagnostics using the same fused batched engine.
    Profiling adds synchronized stage timings; benchmark it separately.
    """

    precision = "double"
    device = "GPU"
    target = "tilelang_statevector"
    mps = False

    def __init__(self, *, precision="double", seed=42, batch_size=128, strategy="auto", profile=False,
                 matrix_backend="auto", kernel_backend="auto"):
        validate_precision(precision)
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
            raise ValueError("TileLang seed must be an integer between 0 and 2**64-1.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 65535:
            raise ValueError("TileLang batch_size must be an integer between 1 and 65535.")
        if strategy not in ("auto", "full", "cones", "reference"):
            raise ValueError("TileLang strategy must be 'auto', 'full', 'cones' or 'reference'.")
        if matrix_backend not in ("auto", "cpu", "gpu"):
            raise ValueError("TileLang matrix_backend must be 'auto', 'cpu' or 'gpu'.")
        if kernel_backend not in ("auto", "baseline", "optimized"):
            raise ValueError("TileLang kernel_backend must be 'auto', 'baseline' or 'optimized'.")
        self._torch, self._kernels = _load_runtime()
        self.precision, self.seed = precision, int(seed)
        self._dtype = "float32" if precision == "single" else "float64"
        self._real_dtype = self._torch.float32 if precision == "single" else self._torch.float64
        self._complex_dtype = self._torch.complex64 if precision == "single" else self._torch.complex128
        self.batch_size, self.strategy, self.profile = batch_size, strategy, bool(profile)
        self.matrix_backend = matrix_backend
        self.kernel_backend = kernel_backend
        self._prepared = None
        self._compiled = None
        self._constants = {}
        self._matrix_builder = None
        self._cone_metadata = {}
        self._kernel_cache = {}
        self.last_execution_stats = None

    def execute(self, prepared, theta_values_all, *, expectation_method="exact", shots=4096):
        if self.strategy == "reference":
            return self._execute_reference(prepared, theta_values_all, expectation_method=expectation_method, shots=shots)
        self.last_execution_stats = None
        if expectation_method not in ("exact", "statevector", "shots"):
            raise ValueError("TileLang expectation_method must be 'exact', 'statevector' or 'shots'.")
        sampling = expectation_method == "shots"
        if sampling:
            validate_shots(shots)
            shots = int(shots)
        theta = np.asarray(theta_values_all, dtype=float)
        if theta.ndim != 2 or theta.shape[1] != len(prepared.param_order) or not np.isfinite(theta).all():
            raise ValueError("TileLang theta must be finite and have one column per prepared parameter.")
        started = perf_counter()
        compile_seconds = 0.
        if self._prepared is not prepared:
            compiled = _compile_job(prepared)
            self._compiled, self._prepared, self._constants = compiled, prepared, {}
            self._matrix_builder, self._cone_metadata = None, {}
            compile_seconds = perf_counter() - started
        job = self._compiled
        n = job.circuit.num_qubits
        if n < 1:
            raise ValueError("TileLang requires at least one qubit.")
        full = full_cone(n, job.gates, job.sampling_supports if sampling else job.terms)
        cones = job.sampling_cones if sampling else job.cones
        use_full = self.strategy == "full" or (self.strategy == "auto" and cone_work(cones) >= cone_work(full))
        if use_full:
            cones = full
        width = max((len(c.sites) for c in cones), default=0)
        if width > 62:
            raise MemoryError(f"TileLang selected a {width}-qubit state, exceeding full-state memory limits.")
        stats = dict(n_samples=len(theta), n_observables=len(job.observable_terms),
                     gate_calls=0, pauli_calls=0, precision=self.precision,
                     execution_path="batched_full" if use_full else "batched_cones",
                     expectation_method=expectation_method, compile_seconds=compile_seconds,
                     source_qubits=n, max_cone_qubits=width, n_cones=len(cones), n_batches=0,
                     source_gate_count=sum(i.operation.name != "barrier" for i in job.circuit.data),
                     fused_gate_count=len(job.gates), distinct_matrix_count=len(job.matrix_keys),
                     simulated_amplitudes_per_row=sum(1 << len(c.sites) for c in cones),
                     estimated_work_per_row=cone_work(cones), reference_work_per_row=cone_work(full),
                     matrix_seconds=0., profiled=self.profile, state_dtype=self._dtype,
                     shots=shots if sampling else None, seed=self.seed if sampling else None,
                     sample_calls=0, n_measurement_groups=len(job.sampling_groups) if sampling else 0,
                     sampled_shots=len(theta) * shots * len(job.sampling_groups) if sampling else 0)
        stats.update(kernel_backend=self.kernel_backend, grouped_gate_calls=0, fast_pauli_calls=0)
        if self._matrix_builder is None and self.matrix_backend != "cpu" and job.matrix_keys:
            self._matrix_builder = MatrixBuilder(self._torch, self._kernels, job.matrix_keys)
        builder = self._matrix_builder if self.matrix_backend != "cpu" else None
        gpu_matrices = builder is not None and len(builder.plan["fallback"]) < len(job.matrix_keys)
        stats["matrix_backend"] = ("hybrid" if builder.plan["fallback"] else "gpu") if gpu_matrices else "cpu"
        stats["matrix_fallback_count"] = len(builder.plan["fallback"]) if builder is not None else len(job.matrix_keys)
        features = np.zeros((len(theta), len(job.observable_terms)), dtype=float)
        torch = self._torch
        generator = torch.Generator(device="cuda").manual_seed(self.seed) if sampling else None
        size = 1 << width
        # Include both states, shared matrices and the largest partial reduction.
        tile = 1024 if self._fast_reduction(size) else 128
        partials_per_row = 8 * 64 * ((size + tile - 1) // tile)
        scalar_bytes = 4 if self.precision == "single" else 8
        bytes_per_row = 4 * scalar_bytes * size + 32 * scalar_bytes * len(job.matrix_keys) + partials_per_row
        if sampling:
            bytes_per_row = ((6 * scalar_bytes + 16) * size + 32 * scalar_bytes * len(job.matrix_keys)
                             + 16 * min(shots, SHOT_CHUNK) + 8192 + 64 * scalar_bytes)
        if gpu_matrices:
            bytes_per_row += 8 * (max(1, theta.shape[1]) + len(builder.plan["values"]))
        batch_limit = min(self.batch_size, len(theta)) if len(theta) else 0
        if batch_limit and cones:
            free, _total = torch.cuda.mem_get_info()
            reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            usable = free + reusable
            batch_limit = min(batch_limit, int(.8 * usable // max(1, bytes_per_row)))
            if batch_limit < 1:
                raise MemoryError(f"TileLang selected a {width}-qubit state requiring approximately "
                                  f"{bytes_per_row} bytes per row; available GPU memory is {free} bytes.")
        stats["batch_size"] = batch_limit
        stats["estimated_buffer_bytes"] = bytes_per_row * batch_limit
        for start in range(0, len(theta), max(1, batch_limit)):
            rows = theta[start:start + batch_limit]
            batch = len(rows)
            stats["n_batches"] += 1
            term_values = np.ones((batch, len(job.terms)), dtype=float)
            if cones:
                if gpu_matrices:
                    matrices = builder.build(rows, self._dtype, stats, self._stage)
                else:
                    before = perf_counter()
                    host_matrices = matrix_batch(job.matrix_keys, rows, self._constants)
                    stats["matrix_seconds"] += perf_counter() - before
                    with self._stage(stats, "host_to_device_seconds"):
                        matrices = torch.view_as_real(torch.tensor(host_matrices, dtype=self._complex_dtype, device="cuda")).contiguous()
                matrix_views = tuple(matrices[i] for i in range(len(job.matrix_keys)))
                for cone in cones:
                    cache_key = (cone, sampling, self.kernel_backend)
                    if cache_key not in self._cone_metadata:
                        local = {q: i for i, q in enumerate(cone.sites)}
                        calls = tuple((job.gates[i].matrix_id, tuple(local[q] for q in job.gates[i].sites))
                                      for i in cone.gates)
                        groups = []
                        if not sampling:
                            partitions = (cone.terms,)
                            if self._fast_reduction(1 << len(cone.sites)):
                                partitions = (tuple(t for t in cone.terms if job.terms[t][0] == 0),
                                              tuple(t for t in cone.terms if job.terms[t][0] != 0))
                            with self._stage(stats, "host_to_device_seconds"):
                                for partition in partitions:
                                    for offset in range(0, len(partition), 64):
                                        selected = partition[offset:offset + 64]
                                        masks = torch.tensor([local_masks(job.terms[t], cone.sites) for t in selected],
                                                             dtype=torch.int64, device="cuda")
                                        groups.append((selected, masks, self._fast_reduction(1 << len(cone.sites))
                                                       and job.terms[selected[0]][0] != 0))
                        self._cone_metadata[cache_key] = calls, tuple(groups)
                    calls, groups = self._cone_metadata[cache_key]
                    cone_size = 1 << len(cone.sites)
                    grouped = self.kernel_backend == "optimized" or (
                        self.kernel_backend == "auto" and cone_size >= 4096 and cone_size * batch >= 65536)
                    gate_name = "apply_batched_gate_grouped" if grouped else "apply_batched_gate"
                    with self._stage(stats, "evolution_seconds"):
                        state = torch.zeros((batch, cone_size, 2), dtype=self._real_dtype, device="cuda")
                        state[:, 0, 0] = 1.
                        output = torch.empty_like(state)
                        for matrix_id, sites in calls:
                            self._launcher(gate_name, cone_size, batch, len(sites), self._dtype)(
                                state, matrix_views[matrix_id], output,
                                sites[0], sites[1] if len(sites) == 2 else 0)
                            state, output = output, state
                            stats["gate_calls"] += 1
                            stats["grouped_gate_calls"] += int(grouped)
                    if sampling:
                        del output
                        sampled = sample_cone(self, job, cone, state, shots, generator, stats)
                        for term, values in sampled.items():
                            term_values[:, term] = values
                        del state
                        continue
                    for selected, masks, paired in groups:
                        fast = self._fast_reduction(cone_size)
                        tile = 1024 if fast else 128
                        with self._stage(stats, "reduction_seconds"):
                            reduction_size = cone_size // 2 if paired else cone_size
                            partials = torch.empty((len(selected), batch, (reduction_size + tile - 1) // tile),
                                                   dtype=torch.float64, device="cuda")
                            pauli_name = "pauli_partials_fast" if fast else "pauli_partials"
                            signature = (cone_size, len(cone.sites), batch, len(selected), self._dtype)
                            if fast:
                                signature += (paired,)
                            self._launcher(pauli_name, *signature)(
                                state, masks, partials)
                            reduced = partials.sum(dim=2)
                            stats["pauli_calls"] += 1
                            stats["fast_pauli_calls"] += int(fast)
                        with self._stage(stats, "device_to_host_seconds"):
                            term_values[:, list(selected)] = reduced.cpu().numpy().T
                        del partials, reduced, masks
                    del state, output
                del matrix_views, matrices
            for column, indexed in enumerate(job.observable_indices):
                for term, coefficient in indexed:
                    features[start:start + batch, column] += coefficient * term_values[:, term]
        stats["total_seconds"] = perf_counter() - started
        self.last_execution_stats = stats
        return features, prepared.obs_metadata

    def _launcher(self, name, *signature):
        if self.kernel_backend == "baseline":
            return getattr(self._kernels, name)(*signature)
        key = (name, *signature)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = getattr(self._kernels, name)(*signature)
        return self._kernel_cache[key]

    def _fast_reduction(self, size):
        return self.kernel_backend == "optimized" or (self.kernel_backend == "auto" and size >= 1024)

    @contextmanager
    def _stage(self, stats, name):
        if self.profile:
            self._torch.cuda.synchronize()
            started = perf_counter()
        yield
        if self.profile:
            self._torch.cuda.synchronize()
            stats[name] = stats.get(name, 0.) + perf_counter() - started

    def _execute_reference(self, prepared, theta_values_all, *, expectation_method="exact", shots=4096):
        """Preserved phase-2 per-row/per-observable implementation for comparisons."""
        self.last_execution_stats = None
        if expectation_method not in ("exact", "statevector", "shots"):
            raise ValueError("TileLang expectation_method must be 'exact', 'statevector' or 'shots'.")
        sampling = expectation_method == "shots"
        if sampling:
            validate_shots(shots)
            shots = int(shots)
        theta = np.asarray(theta_values_all, dtype=float)
        if theta.ndim != 2 or theta.shape[1] != len(prepared.param_order) or not np.isfinite(theta).all():
            raise ValueError("TileLang theta must be finite and have one column per prepared parameter.")
        started = perf_counter()
        if self._prepared is not prepared:
            compiled = _compile_job(prepared)
            self._compiled, self._prepared = compiled, prepared
            self._constants, self._matrix_builder, self._cone_metadata = {}, None, {}
        compiled = self._compiled
        features = np.empty((len(theta), len(compiled.observable_terms)), dtype=float)
        n = compiled.circuit.num_qubits
        if n < 1 or n > 62:
            raise ValueError("TileLang full-state execution requires between 1 and 62 qubits.")
        size = 1 << n
        stats = dict(n_samples=len(theta), n_observables=features.shape[1], gate_calls=0,
                     pauli_calls=0, precision=self.precision, execution_path="full_statevector",
                     expectation_method=expectation_method, state_dtype=self._dtype,
                     shots=shots if sampling else None, seed=self.seed if sampling else None,
                     sample_calls=0, n_measurement_groups=len(compiled.sampling_groups) if sampling else 0,
                     sampled_shots=len(theta) * shots * len(compiled.sampling_groups) if sampling else 0)
        torch = self._torch
        generator = torch.Generator(device="cuda").manual_seed(self.seed) if sampling else None
        if len(theta) and features.shape[1]:
            free, _total = torch.cuda.mem_get_info()
            scalar_bytes = 4 if self.precision == "single" else 8
            needed = ((8 * scalar_bytes + 16) * size + 16 * min(shots, SHOT_CHUNK) + 8192
                      if sampling else (4 * scalar_bytes + 8) * size)
            if needed > .9 * free:
                raise MemoryError(f"TileLang full state for {n} qubits needs at least {needed} bytes; "
                                  f"available GPU memory is {free} bytes.")
            state = torch.empty((size, 2), dtype=self._real_dtype, device="cuda")
            output = torch.empty_like(state)
            contributions = torch.empty((size,), dtype=torch.float64, device="cuda") if not sampling else None
            for row, values in enumerate(theta):
                # Bind the loaded/lowered circuit's own Parameter objects.
                bound = compiled.circuit.assign_parameters(dict(zip(compiled.circuit.parameters,
                                                                    values[list(compiled.columns)])))
                state.zero_()
                state[0, 0] = 1.
                for instruction in bound.data:
                    if instruction.operation.name == "barrier":
                        continue
                    sites = [bound.find_bit(q).index for q in instruction.qubits]
                    if len(sites) not in (1, 2):
                        raise ValueError("Lowered TileLang gates must act on one or two qubits.")
                    matrix = torch.view_as_real(torch.tensor(np.asarray(instruction.operation.to_matrix()),
                                                            dtype=self._complex_dtype, device="cuda")).contiguous()
                    self._kernels.apply_gate(size, len(sites), self._dtype)(
                        state, matrix, output, sites[0], sites[1] if len(sites) == 2 else 0)
                    state, output = output, state
                    stats["gate_calls"] += 1
                if sampling:
                    cone = full_cone(n, compiled.gates, compiled.sampling_supports)
                    term_values = np.ones((1, len(compiled.terms)))
                    if cone:
                        for term, values in sample_cone(self, compiled, cone[0], state[None], shots, generator, stats).items():
                            term_values[:, term] = values
                    for column, indexed in enumerate(compiled.observable_indices):
                        features[row, column] = sum(coefficient * term_values[0, term] for term, coefficient in indexed)
                    continue
                for column, terms in enumerate(compiled.observable_terms):
                    value = 0.
                    for x_mask, z_mask, phase, coefficient in terms:
                        if coefficient == 0:
                            continue
                        if not (x_mask or z_mask):
                            value += coefficient
                        else:
                            self._kernels.pauli_contributions(size, n, self._dtype)(state, contributions, x_mask, z_mask, phase)
                            value += coefficient * float(contributions.sum().item())
                            stats["pauli_calls"] += 1
                    features[row, column] = value
        stats["total_seconds"] = perf_counter() - started
        self.last_execution_stats = stats
        return features, prepared.obs_metadata
