"""Reusable sparse angle plans and GPU fused-matrix preparation."""

import numpy as np

from pqfmlib.core.tilelang_plan import matrix_batch


def pack_matrix_plan(keys):
    """Pack affine keys; arbitrary QPY expressions retain the CPU binder."""
    angles = {("affine", 0., ()): 0}
    instructions, ranges, fallback = [], [], []
    kinds = {"u": 0, "cx": 1, "rxx": 2, "ryy": 3, "rzz": 4}
    for index, key in enumerate(keys):
        start = len(instructions)
        if any(p[0] != "affine" for _name, plans, _sites in key for p in plans):
            fallback.append(index)
            ranges.append((start, start, 0))
            continue
        width = 1 + max(q for _name, _plans, sites in key for q in sites)
        for name, plans, sites in key:
            ids = []
            for plan in plans:
                if plan not in angles:
                    angles[plan] = len(angles)
                ids.append(angles[plan])
            ids += [0] * (3 - len(ids))
            instructions.append((kinds[name], sites[0], sites[1] if len(sites) == 2 else 0, *ids))
        ranges.append((start, len(instructions), 1 << width))
    offsets, columns, coefficients, values = [0], [], [], []
    for _kind, value, mapping in angles:
        values.append(value)
        for column, coefficient in mapping:
            columns.append(column)
            coefficients.append(coefficient)
        offsets.append(len(columns))
    return dict(instructions=np.asarray(instructions or [(0,) * 6], dtype=np.int32),
                ranges=np.asarray(ranges, dtype=np.int32).reshape(-1, 3),
                offsets=np.asarray(offsets, dtype=np.int32),
                columns=np.asarray(columns or [0], dtype=np.int32),
                coefficients=np.asarray(coefficients or [0.], dtype=np.float64),
                values=np.asarray(values, dtype=np.float64), fallback=tuple(fallback))


class MatrixBuilder:
    """Upload fixed metadata once; upload parameters rather than all matrices."""

    def __init__(self, torch, kernels, keys):
        self.torch, self.kernels, self.keys = torch, kernels, keys
        self.plan = pack_matrix_plan(keys)
        self.device = None
        self.constants = {}

    def build(self, theta, dtype, stats, stage):
        torch, plan = self.torch, self.plan
        batch = len(theta)
        with stage(stats, "host_to_device_seconds"):
            if self.device is None:
                self.device = {k: torch.tensor(v, device="cuda") for k, v in plan.items() if k != "fallback"}
                self.device["fallback"] = torch.tensor(plan["fallback"], dtype=torch.int64, device="cuda")
            host = np.ascontiguousarray(theta) if theta.shape[1] else np.zeros((batch, 1))
            # NumPy can mark a singleton slice contiguous while retaining a negative stride.
            if any(stride < 0 for stride in host.strides):
                host = host.copy()
            parameters = torch.tensor(host,
                                      dtype=torch.float64, device="cuda").contiguous()
        device = self.device
        with stage(stats, "matrix_device_seconds"):
            angles = torch.empty((len(plan["values"]), batch), dtype=torch.float64, device="cuda")
            matrices = torch.empty((len(self.keys), batch, 4, 4, 2),
                                   dtype=torch.float32 if dtype == "float32" else torch.float64, device="cuda")
            self.kernels.affine_angles(batch, parameters.shape[1], len(plan["values"]), len(plan["columns"]))(parameters,
                device["offsets"], device["columns"], device["coefficients"], device["values"], angles)
            self.kernels.fused_matrices(batch, len(self.keys), len(plan["instructions"]), len(plan["values"]), dtype)(
                device["ranges"], device["instructions"], angles, matrices)
        if plan["fallback"]:
            from time import perf_counter
            before = perf_counter()
            host = matrix_batch(tuple(self.keys[i] for i in plan["fallback"]), theta, self.constants)
            stats["matrix_seconds"] += perf_counter() - before
            with stage(stats, "host_to_device_seconds"):
                values = torch.view_as_real(torch.tensor(host, device="cuda",
                    dtype=torch.complex64 if dtype == "float32" else torch.complex128)).contiguous()
                matrices.index_copy_(0, device["fallback"], values)
        return matrices
