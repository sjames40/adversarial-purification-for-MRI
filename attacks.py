from __future__ import annotations

import torch

from models import networks
from train_MoDL import recon
from util.metrics import magnitude


def _adjoint(measured_kspace, sensitivity, mask):
    return networks.OPAT(sensitivity)(measured_kspace, mask)


def measurement_attack(
    model,
    clean_kspace: torch.Tensor,
    target: torch.Tensor,
    sensitivity: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int = 30,
    method: str = "pgd",
    reference: str = "clean_reconstruction",
    block_iter: int = 6,
    cg_tol: float = 1e-6,
    lambda_reg: float = 1.0,
    random_start: bool = True,
    loss_domain: str = "complex",
) -> tuple[torch.Tensor, torch.Tensor]:
    """L-infinity attack on acquired real/imag k-space components.

    PGD implements Eq. (3). Momentum is a reproducible MI-FGSM-style
    approximation because the paper does not publish enough AUTO details to
    reconstruct that optimizer exactly.
    """
    if method not in {"pgd", "momentum"}:
        raise ValueError(f"Unknown method: {method}")
    if reference not in {"clean_reconstruction", "ground_truth"}:
        raise ValueError(f"Unknown reference: {reference}")
    if loss_domain not in {"complex", "magnitude"}:
        raise ValueError(f"Unknown loss_domain: {loss_domain}")
    if epsilon < 0 or step_size <= 0 or steps <= 0:
        raise ValueError("epsilon, step_size, and steps must be valid positive values")

    acquired = mask[:, None].expand_as(clean_kspace)
    with torch.no_grad():
        clean_input = _adjoint(clean_kspace, sensitivity, mask)
        if reference == "clean_reconstruction":
            reference_image = recon(
                model,
                clean_input,
                sensitivity,
                mask,
                block_iter,
                cg_tol,
                lambda_reg,
            ).detach()
        else:
            reference_image = target.detach()

    if random_start:
        delta = torch.empty_like(clean_kspace).uniform_(-epsilon, epsilon)
        delta = delta * acquired
    else:
        delta = torch.zeros_like(clean_kspace)
    momentum = torch.zeros_like(delta)

    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        adversarial_kspace = clean_kspace + delta * acquired
        adversarial_input = _adjoint(adversarial_kspace, sensitivity, mask)
        reconstruction = recon(
            model,
            adversarial_input,
            sensitivity,
            mask,
            block_iter,
            cg_tol,
            lambda_reg,
        )
        if loss_domain == "complex":
            # The released author code instantiates nn.MSELoss directly on the
            # two real/imaginary channels. The paper itself leaves L generic.
            loss = torch.mean((reconstruction - reference_image) ** 2)
        else:
            loss = torch.mean(
                (magnitude(reconstruction) - magnitude(reference_image)) ** 2
            )
        gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        gradient = gradient * acquired

        if method == "momentum":
            normalizer = gradient.abs().mean(
                dim=tuple(range(1, gradient.ndim)), keepdim=True
            ).clamp_min(1e-12)
            momentum = momentum + gradient / normalizer
            direction = momentum.sign()
        else:
            direction = gradient.sign()

        delta = torch.clamp(
            delta.detach() + step_size * direction,
            min=-epsilon,
            max=epsilon,
        )
        delta = delta * acquired

    return (clean_kspace + delta).detach(), delta.detach()
