"""CPU plans for exact fusion, batched matrices and backward observable cones."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from pqfmlib.core.observables import group_paulis_qubitwise


@dataclass(frozen=True)
class FusedGate:
    sites: tuple
    key: tuple
    matrix_id: int = 0


@dataclass(frozen=True)
class Cone:
    sites: tuple
    gates: tuple
    terms: tuple


@dataclass(frozen=True)
class MeasurementGroup:
    axes: tuple
    members: tuple


def sampling_plan(qubits, gates, terms):
    """Joint measurement bases and backward cones of their complete supports."""
    selected = [i for i, (x, z, _phase) in enumerate(terms) if x or z]
    labels = []
    for i in selected:
        x, z, _phase = terms[i]
        labels.append("".join("IXZY"[(((x >> q) & 1) + 2 * ((z >> q) & 1))]
                              for q in reversed(range(qubits))))
    groups, supports = [], []
    for basis, members in group_paulis_qubitwise(labels):
        axes = tuple((q, axis) for q, axis in enumerate(reversed(basis)) if axis != "I")
        x = sum(1 << q for q, a in axes if a in "XY")
        z = sum(1 << q for q, a in axes if a in "ZY")
        supports.append((x, z, sum(a == "Y" for _q, a in axes) % 4))
        groups.append(MeasurementGroup(axes, tuple(selected[i] for i in members)))
    return tuple(groups), tuple(supports), causal_cones(qubits, gates, supports)


def _angle(value, positions):
    params = tuple(sorted(getattr(value, "parameters", ()), key=lambda p: p.name))
    if not params:
        return ("affine", float(value), ())
    try:
        coefficients = []
        for p in params:
            derivative = value.gradient(p)
            if getattr(derivative, "parameters", ()):
                raise ValueError("nonlinear angle")
            coefficients.append((positions[p.name], float(derivative)))
        offset = float(value.bind({p: 0 for p in params}))
        return ("affine", offset, tuple(coefficients))
    except (ValueError, TypeError, ArithmeticError):
        # Preserve general QPY expressions; do not linearize nonlinear angles.
        return ("expression", value, tuple((p, positions[p.name]) for p in params))


def fuse_gates(circuit, param_order):
    """Fuse consecutive operations whose union contains at most two sites."""
    positions = {p.name: i for i, p in enumerate(param_order)}
    groups, pending, active = [], [], set()

    def flush():
        if pending:
            sites = tuple(sorted(active))
            local = {q: i for i, q in enumerate(sites)}
            key = tuple((name, angles, tuple(local[q] for q in operands))
                        for name, angles, operands in pending)
            groups.append(FusedGate(sites, key))

    for item in circuit.data:
        if item.operation.name == "barrier":
            continue
        sites = tuple(circuit.find_bit(q).index for q in item.qubits)
        if len(active.union(sites)) > 2:
            flush()
            pending, active = [], set()
        active.update(sites)
        pending.append((item.operation.name, tuple(_angle(p, positions) for p in item.operation.params), sites))
    flush()
    keys = tuple(dict.fromkeys(g.key for g in groups))
    indices = {key: i for i, key in enumerate(keys)}
    return tuple(replace(g, matrix_id=indices[g.key]) for g in groups), keys


def _angle_values(plan, theta):
    kind, value, mapping = plan
    if kind == "affine":
        result = np.full(len(theta), value, dtype=float)
        for column, coefficient in mapping:
            result += coefficient * theta[:, column]
        return result
    return np.array([float(value.bind({p: row[column] for p, column in mapping})) for row in theta])


def _native_matrix(name, angles, batch):
    if name == "u":
        theta, phi, lam = angles
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        out = np.empty((batch, 2, 2), dtype=complex)
        out[:, 0, 0], out[:, 0, 1] = c, -np.exp(1j * lam) * s
        out[:, 1, 0], out[:, 1, 1] = np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c
        return out
    if name == "cx":
        return np.broadcast_to(np.array([[1, 0, 0, 0], [0, 0, 0, 1],
                                          [0, 0, 1, 0], [0, 1, 0, 0]], complex), (batch, 4, 4))
    if name not in ("rxx", "ryy", "rzz"):
        raise ValueError(f"Unexpected lowered TileLang gate: {name}")
    half = angles[0] / 2
    if name == "rzz":
        out = np.zeros((batch, 4, 4), dtype=complex)
        out[:, np.arange(4), np.arange(4)] = np.exp(-1j * half[:, None] * np.array([1, -1, -1, 1]))
    else:
        out = np.cos(half)[:, None, None] * np.eye(4, dtype=complex)[None]
        signs = np.array([1, 1, 1, 1]) if name == "rxx" else np.array([-1, 1, 1, -1])
        out[:, np.arange(4), np.arange(4) ^ 3] = -1j * np.sin(half)[:, None] * signs
    return out


@lru_cache(maxsize=16)
def _embedding(width, sites):
    """Indices embedding an ordered gate matrix into little-endian local sites."""
    rows, columns, native_rows, native_columns = [], [], [], []
    mask = sum(1 << q for q in sites)
    for row in range(1 << width):
        for column in range(1 << width):
            if (row & ~mask) == (column & ~mask):
                rows.append(row)
                columns.append(column)
                native_rows.append(sum(((row >> q) & 1) << k for k, q in enumerate(sites)))
                native_columns.append(sum(((column >> q) & 1) << k for k, q in enumerate(sites)))
    return rows, columns, native_rows, native_columns


def matrix_batch(keys, theta, constants):
    """Evaluate each distinct matrix once per batch; reuse fixed CPU matrices."""
    batch = len(theta)
    matrices = np.zeros((len(keys), batch, 4, 4), dtype=complex)
    angles_cache, native_cache = {}, {}
    for index, key in enumerate(keys):
        width = 1 + max(q for _name, _angles, sites in key for q in sites)
        dimension = 1 << width
        if key in constants:
            matrices[index, :, :dimension, :dimension] = constants[key]
            continue
        result = np.broadcast_to(np.eye(dimension, dtype=complex), (batch, dimension, dimension)).copy()
        for name, plans, sites in key:
            native_key = (name, plans)
            if native_key not in native_cache:
                for plan in plans:
                    if plan not in angles_cache:
                        angles_cache[plan] = _angle_values(plan, theta)
                native_cache[native_key] = _native_matrix(name, [angles_cache[p] for p in plans], batch)
            rows, cols, nr, nc = _embedding(width, sites)
            embedded = np.zeros_like(result)
            embedded[:, rows, cols] = native_cache[native_key][:, nr, nc]
            result = embedded @ result
        matrices[index, :, :dimension, :dimension] = result
        if all(p[0] == "affine" and not p[2] for _name, plans, _sites in key for p in plans):
            constants[key] = result[0].copy()
    return matrices


def full_cone(qubits, gates, terms):
    active = tuple(i for i, (x, z, _phase) in enumerate(terms) if x or z)
    return (Cone(tuple(range(qubits)), tuple(range(len(gates))), active),) if active else ()


def causal_cones(qubits, gates, terms):
    """Exact backward support; merge nested supports with gate unions in order."""
    cones, lookup = [], {}
    for term, (x, z, _phase) in enumerate(terms):
        active = {q for q in range(qubits) if ((x | z) >> q) & 1}
        if not active:
            continue
        selected = []
        for index in range(len(gates) - 1, -1, -1):
            if active.intersection(gates[index].sites):
                active.update(gates[index].sites)
                selected.append(index)
        key = (tuple(sorted(active)), tuple(reversed(selected)))
        if key not in lookup:
            lookup[key] = len(cones)
            cones.append([key[0], key[1], []])
        cones[lookup[key]][2].append(term)
    maximal = [i for i, (sites, _gates, _terms) in enumerate(cones)
               if not any(i != j and set(sites) <= set(other_sites)
                          and (len(sites) < len(other_sites) or j < i)
                          for j, (other_sites, _other_gates, _other_terms) in enumerate(cones))]
    merged = [[cones[i][0], set(), []] for i in maximal]
    for sites, selected, indices in cones:
        target = next(c for c in merged if set(sites) <= set(c[0]))
        target[1].update(selected)
        target[2].extend(indices)
    return tuple(Cone(sites, tuple(sorted(selected)), tuple(indices)) for sites, selected, indices in merged)


def cone_work(cones):
    return sum((1 << len(c.sites)) * (len(c.gates) + len(c.terms)) for c in cones)


def local_masks(term, sites):
    x, z, phase = term
    return (sum(((x >> q) & 1) << i for i, q in enumerate(sites)),
            sum(((z >> q) & 1) << i for i, q in enumerate(sites)), phase)
