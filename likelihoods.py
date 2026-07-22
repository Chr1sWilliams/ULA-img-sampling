"""Log-likelihood functions that can be injected into diffusion guidance."""

from functools import partial
import importlib
from typing import Any, Callable

import torch as th


LogLikelihood = Callable[[th.Tensor], th.Tensor]


def zero_log_likelihood(x: th.Tensor) -> th.Tensor:
    """Return one differentiable zero log-likelihood value per sample."""
    if x.ndim < 1:
        raise ValueError("x must include a batch dimension.")
    return x.reshape(x.shape[0], -1).sum(dim=1) * 0.0


def interferometric_log_likelihood(
    x: th.Tensor,
    forward_operator: Any,
) -> th.Tensor:
    """Example EHT closure-phase and log-closure-amplitude likelihood.

    Args:
        x: Image batch with shape ``(B, C, H, W)``.
        forward_operator: Configured ``InterferometerOperator`` instance.

    Returns:
        One log-likelihood value per image, with shape ``(B,)``.
    """
    if x.ndim != 4:
        raise ValueError(
            f"x must have shape (B, C, H, W), got {tuple(x.shape)}"
        )

    # Preserve the orientation convention of the original implementation.
    x = th.flip(th.rot90(x, 3, dims=(2, 3)), dims=(3,))

    cp_model, lc_model = forward_operator.forward(x)
    residual_cp = (
        th.exp(1j * forward_operator.cphase).unsqueeze(0)
        - th.exp(1j * cp_model)
    )
    residual_lc = forward_operator.camp.unsqueeze(0) - lc_model

    sigma_inv_res_cp = th.linalg.solve(
        forward_operator.cov_cp,
        residual_cp.T,
    ).T
    sigma_inv_res_lc = th.linalg.solve(
        forward_operator.cov_lc,
        residual_lc.T,
    ).T

    log_like_cp = -0.5 * th.einsum(
        "ij,ij->i",
        th.conj(residual_cp),
        sigma_inv_res_cp,
    )
    log_like_lc = -0.5 * th.einsum(
        "ij,ij->i",
        residual_lc,
        sigma_inv_res_lc,
    )
    return log_like_cp.real + log_like_lc.real


def make_interferometric_log_likelihood(
    uvfile: str,
    img_size: int,
    device: th.device,
    psize: float,
) -> LogLikelihood:
    """Build the example likelihood and capture its forward operator."""
    from bh_util.measurements import InterferometerOperator

    forward_operator = InterferometerOperator(
        uvfile,
        img_size=img_size,
        device=device,
        psize=psize,
    )
    return partial(
        interferometric_log_likelihood,
        forward_operator=forward_operator,
    )


def load_log_likelihood(
    specification: str,
    *,
    uvfile: str,
    img_size: int,
    device: th.device,
    psize: float,
) -> LogLikelihood:
    """Load a built-in or importable log-likelihood from a CLI specification.

    Supported specifications are:

    - ``interferometric``: build the EHT example using the supplied parameters.
    - ``zero`` or ``none``: use the differentiable zero map.
    - ``module:function``: import a custom callable with signature ``fn(x)``.
    """
    specification = str(specification).strip()
    normalized = specification.lower()

    if normalized in {"zero", "none"}:
        return zero_log_likelihood
    if normalized in {"interferometric", "eht"}:
        return make_interferometric_log_likelihood(
            uvfile=uvfile,
            img_size=img_size,
            device=device,
            psize=psize,
        )

    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "log-likelihood specification must be zero, interferometric, "
            "or module:function."
        )

    module = importlib.import_module(module_name)
    try:
        log_likelihood = getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module {module_name!r} has no attribute {function_name!r}."
        ) from exc
    if not callable(log_likelihood):
        raise TypeError(
            f"Imported log likelihood {specification!r} is not callable."
        )
    return log_likelihood
