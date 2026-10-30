"""Joint Born sampling on GPU, with bounded shot chunks and local RNG state."""

import numpy as np

from pqfmlib.core.tilelang_plan import local_masks


SHOT_CHUNK = 4096


def sample_cone(executor, job, cone, state, shots, generator, stats):
    """Measure compatible terms using the same bitstrings, preserving covariance."""
    torch, kernels = executor._torch, executor._kernels
    batch, size, _ = state.shape
    local = {q: i for i, q in enumerate(cone.sites)}
    c = np.sqrt(.5)
    bases = {"X": np.array([[c, c], [c, -c]], complex),
             "Y": np.array([[c, -1j * c], [-1j * c, c]], complex)}
    matrices = {}
    needed = {a for group in cone.terms for _q, a in job.sampling_groups[group].axes if a in "XY"}
    with executor._stage(stats, "host_to_device_seconds"):
        for axis in needed:
            matrix = np.zeros((batch, 4, 4), complex)
            matrix[:, :2, :2] = bases[axis]
            matrices[axis] = torch.view_as_real(torch.tensor(matrix, dtype=executor._complex_dtype, device="cuda")).contiguous()
    estimates = {}
    for index in cone.terms:
        group = job.sampling_groups[index]
        rotations = [(local[q], a) for q, a in group.axes if a in "XY"]
        with executor._stage(stats, "sampling_seconds"):
            measured = state.clone() if rotations else state
            scratch = torch.empty_like(state) if rotations else None
            for q, axis in rotations:
                kernels.apply_batched_gate(size, batch, 1, executor._dtype)(measured, matrices[axis], scratch, q, 0)
                measured, scratch = scratch, measured
            probabilities = torch.empty((batch, size), dtype=torch.float64, device="cuda")
            kernels.born_probabilities(size, batch, executor._dtype)(measured, probabilities)
            cdf = probabilities.cumsum(dim=1)
            del probabilities, measured, scratch
            norms = cdf[:, -1:].clone()
            if not bool(torch.isfinite(norms).all().item()) or bool((norms <= 0).any().item()):
                raise RuntimeError("TileLang sampling encountered a nonfinite or zero state norm.")
            cdf.div_(norms)
            del norms
            masks = []
            for term in group.members:
                x, z, _phase = local_masks(job.terms[term], cone.sites)
                masks.append(x | z)
            supports = torch.tensor(masks, dtype=torch.int64, device="cuda")
            counts = torch.zeros((len(group.members), batch), dtype=torch.int64, device="cuda")
            for start in range(0, shots, SHOT_CHUNK):
                count = min(SHOT_CHUNK, shots - start)
                uniforms = torch.rand((batch, count), dtype=torch.float64, device="cuda", generator=generator)
                samples = torch.searchsorted(cdf, uniforms, right=True)
                del uniforms
                stats["sample_calls"] += 1
                for offset in range(0, len(group.members), 64):
                    end = min(len(group.members), offset + 64)
                    partials = torch.empty((end - offset, batch, (count + 127) // 128), dtype=torch.int32, device="cuda")
                    kernels.sampled_pauli_partials(count, len(cone.sites), batch, end - offset)(
                        samples, supports[offset:end], partials)
                    counts[offset:end].add_(partials.sum(dim=2, dtype=torch.int64))
                    stats["pauli_calls"] += 1
                    del partials
                del samples
            del cdf, supports
        with executor._stage(stats, "device_to_host_seconds"):
            values = counts.cpu().numpy().T / float(shots)
        for column, term in enumerate(group.members):
            estimates[term] = values[:, column]
    return estimates
