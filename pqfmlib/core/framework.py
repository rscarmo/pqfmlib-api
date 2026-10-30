"""Framework selection without importing optional simulation dependencies."""

from __future__ import annotations

from numbers import Integral


def validate_framework(framework: str) -> None:
    if framework not in ("qiskit", "cudaq", "tilelang"):
        raise ValueError("framework must be 'qiskit', 'cudaq' or 'tilelang'.")


def validate_precision(precision: str) -> None:
    if precision not in ("single", "double"):
        raise ValueError("precision must be 'single' (FP32) or 'double' (FP64).")


def validate_shots(shots) -> None:
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
        raise ValueError("shots must be a positive integer.")


def validate_aer_precision(precision: str, method: str) -> None:
    validate_precision(precision)
    if precision == "single" and method not in (
        "statevector", "density_matrix", "unitary", "superop", "tensor_network",
    ):
        raise ValueError(
            f"precision='single' is unsupported for Aer method={method!r}; "
            "choose an explicit FP32-capable method or precision='double'."
        )


def validate_framework_configuration(
    framework: str, *, simulation: bool, ideal: bool, fakebackend: bool,
    native_topology: bool = False,
) -> None:
    """Allow explicit native, noiseless CUDA-Q simulation without changing defaults."""
    validate_framework(framework)
    if framework == "tilelang":
        if not simulation or fakebackend or (not ideal and not native_topology):
            raise ValueError(
                "framework='tilelang' requires simulation=True, fakebackend=False "
                "and either ideal=True or explicit native topology."
            )
        return
    if framework == "cudaq" and native_topology:
        if not simulation or fakebackend:
            raise ValueError(
                "framework='cudaq' requires simulation=True and fakebackend=False "
                "for native topology; noisy simulation and QPU execution are not supported."
            )
        return
    if framework == "cudaq" and (not simulation or not ideal or fakebackend):
        raise ValueError(
            "framework='cudaq' requires simulation=True, ideal=True and "
            "fakebackend=False; non-ideal simulation, fake backends and QPU "
            "execution are not supported."
        )
