"""Composite losses for vector electric-field graph prediction."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_e_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor,
    boundary_mask: Optional[torch.Tensor] = None,
    node_weight: float = 1.0,
    grad_weight: float = 0.5,
    boundary_weight: float = 0.25,
    magnitude_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return total loss and scalar components for `[Ex, Ey]` predictions."""

    node_loss_all = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    node_loss = node_loss_all.mean()

    src, dst = edge_index
    pred_diff = pred[src] - pred[dst]
    target_diff = target[src] - target[dst]
    grad_loss = F.smooth_l1_loss(pred_diff, target_diff)

    pred_mag = torch.linalg.norm(pred, dim=-1)
    target_mag = torch.linalg.norm(target, dim=-1)
    magnitude_loss = F.smooth_l1_loss(pred_mag, target_mag)

    if boundary_mask is not None and boundary_weight > 0.0:
        mask = boundary_mask.to(pred.device).view(-1)
        boundary_loss = (node_loss_all * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        boundary_loss = pred.new_tensor(0.0)

    total = (
        node_weight * node_loss
        + grad_weight * grad_loss
        + boundary_weight * boundary_loss
        + magnitude_weight * magnitude_loss
    )
    components = {
        "total": float(total.detach().cpu().item()),
        "node": float(node_loss.detach().cpu().item()),
        "grad": float(grad_loss.detach().cpu().item()),
        "boundary": float(boundary_loss.detach().cpu().item()),
        "magnitude": float(magnitude_loss.detach().cpu().item()),
    }
    return total, components

