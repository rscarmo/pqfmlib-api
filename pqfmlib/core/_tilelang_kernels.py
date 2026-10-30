"""FP32/FP64 CUDA kernels. Import only when the optional TileLang engine is selected."""

import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def affine_angles(batch: int, parameters: int, angles: int, coefficients: int):
    @T.prim_func
    def kernel(theta: T.Tensor((batch, parameters), "float64"),
               offsets: T.Tensor((angles + 1,), "int32"),
               columns: T.Tensor((coefficients,), "int32"),
               weights: T.Tensor((coefficients,), "float64"),
               constants: T.Tensor((angles,), "float64"),
               output: T.Tensor((angles, batch), "float64")):
        with T.Kernel(T.ceildiv(angles * batch, 128), threads=128) as block:
            i = block * 128 + T.get_thread_binding()
            if i < angles * batch:
                a, b = i // batch, i % batch
                value = T.alloc_local((1,), "float64")
                value[0] = constants[a]
                for j in T.serial(offsets[a], offsets[a + 1]):
                    value[0] += weights[j] * theta[b, columns[j]]
                output[a, b] = value[0]
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def fused_matrices(batch: int, matrices: int, instructions: int, angles: int, dtype: str):
    """Each lane evolves one column of a fused matrix in FP64 registers."""
    @T.prim_func
    def kernel(ranges: T.Tensor((matrices, 3), "int32"),
               ops: T.Tensor((instructions, 6), "int32"),
               values: T.Tensor((angles, batch), "float64"),
               output: T.Tensor((matrices, batch, 4, 4, 2), dtype)):
        with T.Kernel(T.ceildiv(matrices * batch * 4, 128), threads=128) as block:
            index = block * 128 + T.get_thread_binding()
            if index < matrices * batch * 4:
                col, b, matrix = index % 4, (index // 4) % batch, index // (4 * batch)
                real = T.alloc_local((4,), "float64")
                imag = T.alloc_local((4,), "float64")
                next_real = T.alloc_local((4,), "float64")
                next_imag = T.alloc_local((4,), "float64")
                dim = ranges[matrix, 2]
                for row in T.unroll(4):
                    real[row] = T.if_then_else(row == col and row < dim, T.float64(1), T.float64(0))
                    imag[row] = T.float64(0)
                for op in T.serial(ranges[matrix, 0], ranges[matrix, 1]):
                    kind, q0, q1 = ops[op, 0], ops[op, 1], ops[op, 2]
                    angle = values[ops[op, 3], b]
                    phi = values[ops[op, 4], b]
                    lam = values[ops[op, 5], b]
                    c, s = T.cos(angle / T.float64(2)), T.sin(angle / T.float64(2))
                    for row in T.unroll(4):
                        next_real[row], next_imag[row] = real[row], imag[row]
                        if row < dim:
                            if kind == 0:
                                other = row ^ (1 << q0)
                                phase = T.alloc_local((1,), "float64")
                                own = T.alloc_local((1,), "float64")
                                cross = T.alloc_local((1,), "float64")
                                if ((row >> q0) & 1) == 0:
                                    phase[0], own[0], cross[0] = lam, c, -s
                                    next_real[row] = own[0] * real[row] + cross[0] * (T.cos(phase[0]) * real[other] - T.sin(phase[0]) * imag[other])
                                    next_imag[row] = own[0] * imag[row] + cross[0] * (T.cos(phase[0]) * imag[other] + T.sin(phase[0]) * real[other])
                                else:
                                    next_real[row] = s * (T.cos(phi) * real[other] - T.sin(phi) * imag[other]) + c * (T.cos(phi + lam) * real[row] - T.sin(phi + lam) * imag[row])
                                    next_imag[row] = s * (T.cos(phi) * imag[other] + T.sin(phi) * real[other]) + c * (T.cos(phi + lam) * imag[row] + T.sin(phi + lam) * real[row])
                            elif kind == 1:
                                source = row ^ T.if_then_else(((row >> q0) & 1) != 0, 1 << q1, 0)
                                next_real[row], next_imag[row] = real[source], imag[source]
                            elif kind == 4:
                                sign = T.if_then_else(((row >> q0) & 1) == ((row >> q1) & 1), T.float64(1), T.float64(-1))
                                next_real[row] = c * real[row] + sign * s * imag[row]
                                next_imag[row] = c * imag[row] - sign * s * real[row]
                            else:
                                source = row ^ (1 << q0) ^ (1 << q1)
                                sign = T.if_then_else(kind == 3 and ((row >> q0) & 1) == ((row >> q1) & 1), T.float64(-1), T.float64(1))
                                next_real[row] = c * real[row] + sign * s * imag[source]
                                next_imag[row] = c * imag[row] - sign * s * real[source]
                    for row in T.unroll(4):
                        real[row], imag[row] = next_real[row], next_imag[row]
                for row in T.unroll(4):
                    output[matrix, b, row, col, 0] = T.cast(real[row], dtype)
                    output[matrix, b, row, col, 1] = T.cast(imag[row], dtype)
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def apply_gate(size: int, width: int, dtype: str = "float64"):
    dimension = 1 << width

    @T.prim_func
    def kernel(state: T.Tensor((size, 2), dtype),
               matrix: T.Tensor((dimension, dimension, 2), dtype),
               output: T.Tensor((size, 2), dtype), q0: T.int32, q1: T.int32):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as block:
            i = T.cast(block * 128 + T.get_thread_binding(), "int64")
            if i < size:
                row = (i >> q0) & 1
                if width == 2:
                    row += 2 * ((i >> q1) & 1)
                real = T.alloc_local((1,), dtype)
                imag = T.alloc_local((1,), dtype)
                real[0], imag[0] = 0.0, 0.0
                for column in T.serial(dimension):
                    source = (i & ~(T.int64(1) << q0)) | ((column & 1) << q0)
                    if width == 2:
                        source = (source & ~(T.int64(1) << q1)) | (((column >> 1) & 1) << q1)
                    ar, ai = state[source, 0], state[source, 1]
                    mr, mi = matrix[row, column, 0], matrix[row, column, 1]
                    real[0] += mr * ar - mi * ai
                    imag[0] += mr * ai + mi * ar
                output[i, 0], output[i, 1] = real[0], imag[0]
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def apply_batched_gate(size: int, batch: int, width: int, dtype: str = "float64"):
    dimension = 1 << width

    @T.prim_func
    def kernel(state: T.Tensor((batch, size, 2), dtype),
               matrix: T.Tensor((batch, 4, 4, 2), dtype),
               output: T.Tensor((batch, size, 2), dtype), q0: T.int32, q1: T.int32):
        with T.Kernel(T.ceildiv(size, 128), batch, threads=128) as (block, b):
            i = T.cast(block * 128 + T.get_thread_binding(), "int64")
            if i < size:
                row = (i >> q0) & 1
                if width == 2:
                    row += 2 * ((i >> q1) & 1)
                real = T.alloc_local((1,), dtype)
                imag = T.alloc_local((1,), dtype)
                real[0], imag[0] = 0.0, 0.0
                for column in T.serial(dimension):
                    source = (i & ~(T.int64(1) << q0)) | ((column & 1) << q0)
                    if width == 2:
                        source = (source & ~(T.int64(1) << q1)) | (((column >> 1) & 1) << q1)
                    ar, ai = state[b, source, 0], state[b, source, 1]
                    mr, mi = matrix[b, row, column, 0], matrix[b, row, column, 1]
                    real[0] += mr * ar - mi * ai
                    imag[0] += mr * ai + mi * ar
                output[b, i, 0], output[b, i, 1] = real[0], imag[0]
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def apply_batched_gate_grouped(size: int, batch: int, width: int, dtype: str = "float64"):
    """Load each two-/four-amplitude group once and compute all its outputs."""
    dimension = 1 << width

    @T.prim_func
    def kernel(state: T.Tensor((batch, size, 2), dtype),
               matrix: T.Tensor((batch, 4, 4, 2), dtype),
               output: T.Tensor((batch, size, 2), dtype), q0: T.int32, q1: T.int32):
        with T.Kernel(T.ceildiv(size // dimension, 128), batch, threads=128) as (block, b):
            group = T.cast(block * 128 + T.get_thread_binding(), "int64")
            if group < size // dimension:
                low = T.min(q0, q1) if width == 2 else q0
                base = T.alloc_local((1,), "int64")
                base[0] = (group & ((T.int64(1) << low) - 1)) | ((group >> low) << (low + 1))
                if width == 2:
                    high = T.max(q0, q1)
                    base[0] = (base[0] & ((T.int64(1) << high) - 1)) | ((base[0] >> high) << (high + 1))
                sources = T.alloc_local((dimension,), "int64")
                ar = T.alloc_local((dimension,), dtype)
                ai = T.alloc_local((dimension,), dtype)
                for column in T.unroll(dimension):
                    sources[column] = base[0] | ((column & 1) << q0)
                    if width == 2:
                        sources[column] = sources[column] | (((column >> 1) & 1) << q1)
                    ar[column], ai[column] = state[b, sources[column], 0], state[b, sources[column], 1]
                for row in T.unroll(dimension):
                    real = T.alloc_local((1,), dtype)
                    imag = T.alloc_local((1,), dtype)
                    real[0], imag[0] = 0.0, 0.0
                    for column in T.unroll(dimension):
                        mr, mi = matrix[b, row, column, 0], matrix[b, row, column, 1]
                        real[0] += mr * ar[column] - mi * ai[column]
                        imag[0] += mr * ai[column] + mi * ar[column]
                    output[b, sources[row], 0], output[b, sources[row], 1] = real[0], imag[0]
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def pauli_partials_fast(size: int, qubits: int, batch: int, terms: int, dtype: str = "float64", paired: bool = False):
    effective_size = size // 2 if paired else size
    tiles = T.ceildiv(effective_size, 1024)

    @T.prim_func
    def kernel(state: T.Tensor((batch, size, 2), dtype),
               masks: T.Tensor((terms, 3), "int64"),
               output: T.Tensor((terms, batch, tiles), "float64")):
        with T.Kernel(tiles, terms, batch, threads=128) as (block, term, b):
            values = T.alloc_fragment((1024,), "float64")
            reduced = T.alloc_fragment((1,), "float64")
            for lane in T.Parallel(1024):
                index = T.cast(block * 1024 + lane, "int64")
                values[lane] = 0.0
                if index < effective_size:
                    x, z, phase = masks[term, 0], masks[term, 1], masks[term, 2]
                    i = T.alloc_local((1,), "int64")
                    i[0] = index
                    if paired:
                        # Hermitian Pauli: the real contributions at i and i^x are equal.
                        low_mask = (x & -x) - 1
                        i[0] = (index & low_mask) | ((index & ~low_mask) << 1)
                    j = i[0] ^ x
                    ar, ai = T.cast(state[b, i[0], 0], "float64"), T.cast(state[b, i[0], 1], "float64")
                    value = T.alloc_local((1,), "float64")
                    if (phase & 1) == 0:
                        real = ar * T.cast(state[b, j, 0], "float64") + ai * T.cast(state[b, j, 1], "float64")
                        value[0] = T.if_then_else(phase == 0, real, -real)
                    else:
                        imag = T.cast(state[b, j, 0], "float64") * ai - T.cast(state[b, j, 1], "float64") * ar
                        value[0] = T.if_then_else(phase == 1, -imag, imag)
                    parity = T.call_pure_extern("int32", "__popcll", T.cast(i[0] & z, "uint64")) & 1
                    factor = T.float64(2) if paired else T.float64(1)
                    values[lane] = factor * T.if_then_else(parity == 0, value[0], -value[0])
            T.reduce_sum(values, reduced, dim=0)
            output[term, b, block] = reduced[0]
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def pauli_partials(size: int, qubits: int, batch: int, terms: int, dtype: str = "float64"):
    tiles = T.ceildiv(size, 128)

    @T.prim_func
    def kernel(state: T.Tensor((batch, size, 2), dtype),
               masks: T.Tensor((terms, 3), "int64"),
               output: T.Tensor((terms, batch, tiles), "float64")):
        with T.Kernel(tiles, terms, batch, threads=128) as (block, term, b):
            values = T.alloc_fragment((128,), "float64")
            reduced = T.alloc_fragment((1,), "float64")
            for lane in T.Parallel(128):
                i = T.cast(block * 128 + lane, "int64")
                values[lane] = 0.0
                if i < size:
                    x, z, phase = masks[term, 0], masks[term, 1], masks[term, 2]
                    j = i ^ x
                    real = T.cast(state[b, j, 0], "float64") * T.cast(state[b, i, 0], "float64") + T.cast(state[b, j, 1], "float64") * T.cast(state[b, i, 1], "float64")
                    imag = T.cast(state[b, j, 0], "float64") * T.cast(state[b, i, 1], "float64") - T.cast(state[b, j, 1], "float64") * T.cast(state[b, i, 0], "float64")
                    parity = T.alloc_local((1,), "int32")
                    parity[0] = 0
                    for bit in T.serial(qubits):
                        parity[0] = parity[0] ^ T.cast(((i & z) >> bit) & 1, "int32")
                    value = T.alloc_local((1,), "float64")
                    if phase == 0:
                        value[0] = real
                    elif phase == 1:
                        value[0] = -imag
                    elif phase == 2:
                        value[0] = -real
                    else:
                        value[0] = imag
                    values[lane] = T.if_then_else(parity[0] == 0, value[0], -value[0])
            T.reduce_sum(values, reduced, dim=0)
            output[term, b, block] = reduced[0]
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def pauli_contributions(size: int, qubits: int, dtype: str = "float64"):
    @T.prim_func
    def kernel(state: T.Tensor((size, 2), dtype),
               output: T.Tensor((size,), "float64"),
               x_mask: T.int64, z_mask: T.int64, phase: T.int32):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as block:
            i = T.cast(block * 128 + T.get_thread_binding(), "int64")
            if i < size:
                j = i ^ x_mask
                real = T.cast(state[j, 0], "float64") * T.cast(state[i, 0], "float64") + T.cast(state[j, 1], "float64") * T.cast(state[i, 1], "float64")
                imag = T.cast(state[j, 0], "float64") * T.cast(state[i, 1], "float64") - T.cast(state[j, 1], "float64") * T.cast(state[i, 0], "float64")
                parity = T.alloc_local((1,), "int32")
                parity[0] = 0
                for bit in T.serial(qubits):
                    parity[0] = parity[0] ^ T.cast(((i & z_mask) >> bit) & 1, "int32")
                value = T.alloc_local((1,), "float64")
                if phase == 0:
                    value[0] = real
                elif phase == 1:
                    value[0] = -imag
                elif phase == 2:
                    value[0] = -real
                else:
                    value[0] = imag
                output[i] = T.if_then_else(parity[0] == 0, value[0], -value[0])
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def born_probabilities(size: int, batch: int, dtype: str = "float64"):
    @T.prim_func
    def kernel(state: T.Tensor((batch, size, 2), dtype),
               output: T.Tensor((batch, size), "float64")):
        with T.Kernel(T.ceildiv(size, 128), batch, threads=128) as (block, b):
            i = T.cast(block, "int64") * 128 + T.get_thread_binding()
            if i < size:
                real = T.cast(state[b, i, 0], "float64")
                imag = T.cast(state[b, i, 1], "float64")
                output[b, i] = real * real + imag * imag
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def sampled_pauli_partials(shots: int, qubits: int, batch: int, terms: int):
    tiles = T.ceildiv(shots, 128)

    @T.prim_func
    def kernel(samples: T.Tensor((batch, shots), "int64"),
               supports: T.Tensor((terms,), "int64"),
               output: T.Tensor((terms, batch, tiles), "int32")):
        with T.Kernel(tiles, terms, batch, threads=128) as (block, term, b):
            values = T.alloc_fragment((128,), "int32")
            reduced = T.alloc_fragment((1,), "int32")
            for lane in T.Parallel(128):
                index = block * 128 + lane
                values[lane] = 0
                if index < shots:
                    bits = samples[b, index] & supports[term]
                    parity = T.alloc_local((1,), "int32")
                    parity[0] = 0
                    for bit in T.serial(qubits):
                        parity[0] = parity[0] ^ T.cast((bits >> bit) & 1, "int32")
                    values[lane] = 1 - 2 * parity[0]
            T.reduce_sum(values, reduced, dim=0)
            output[term, b, block] = reduced[0]
    return kernel
