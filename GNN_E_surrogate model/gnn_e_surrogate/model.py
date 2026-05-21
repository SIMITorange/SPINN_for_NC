"""MeshGraphNet-style model for joint Ex/Ey prediction."""

from typing import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a compact MLP used by encoders, processors, and decoder heads."""

    activations: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    act_factory = activations.get(activation, nn.GELU)
    dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_factory())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GraphNetBlock(nn.Module):
    """Residual edge-to-node message passing block implemented with torch ops."""

    def __init__(
        self,
        hidden_dim: int,
        position_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        edge_input_dim = hidden_dim * 2 + position_dim + 1
        self.edge_mlp = build_mlp(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            dropout=dropout,
        )
        self.node_mlp = build_mlp(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        rel = pos[src] - pos[dst]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        edge_input = torch.cat([x[src], x[dst], rel, dist], dim=-1)
        messages = self.edge_mlp(edge_input)

        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst[:, None].expand_as(messages), messages)
        counts = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        counts.scatter_add_(0, dst[:, None], torch.ones_like(counts[dst]))
        agg = agg / counts.clamp_min(1.0)

        update = self.node_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(x + update)


class ElectricFieldMeshGraphNet(nn.Module):
    """Joint vector-output MeshGraphNet for `[ElectricField_x, ElectricField_y]`."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_message_passing_steps: int = 5,
        position_dim: int = 3,
        output_dim: int = 2,
        activation: str = "gelu",
        dropout: float = 0.05,
        use_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.position_dim = position_dim
        self.output_dim = output_dim
        self.use_grad_checkpoint = use_grad_checkpoint

        self.input_proj = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                GraphNetBlock(
                    hidden_dim=hidden_dim,
                    position_dim=position_dim,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(num_message_passing_steps)
            ]
        )
        self.decoder = build_mlp(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, data) -> torch.Tensor:
        """Return normalized electric field with shape `[num_nodes, 2]`."""

        x = self.input_proj(data.x)
        pos = data.pos
        if pos.shape[-1] < self.position_dim:
            pad = self.position_dim - pos.shape[-1]
            pos = torch.nn.functional.pad(pos, (0, pad))
        elif pos.shape[-1] > self.position_dim:
            pos = pos[:, : self.position_dim]

        for block in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = checkpoint(block, x, data.edge_index, pos, use_reentrant=False)
            else:
                x = block(x, data.edge_index, pos)
        return self.decoder(x)

