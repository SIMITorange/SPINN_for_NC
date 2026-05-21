"""Inference API for the cell-level electric-field surrogate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .config import CellGraphConfig, ModelConfig, latest_checkpoint
from .data import SimpleGraphData
from .model import ElectricFieldMeshGraphNet
from .normalization import EFieldNormalizer, as_3d_position


class ElectricFieldSurrogate:
    """Load a trained Ex/Ey surrogate and evaluate cell graphs."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        normalizer_path: Optional[Path] = None,
        graph_config: Optional[CellGraphConfig] = None,
        model_config: Optional[ModelConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.graph_config = graph_config or CellGraphConfig()
        self.model_config = model_config or ModelConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else latest_checkpoint(
            self.graph_config.checkpoint_dir
        )
        self.normalizer_path = (
            Path(normalizer_path)
            if normalizer_path
            else self.graph_config.normalizer_dir / "e_field_normalizer.npz"
        )
        self.normalizer = EFieldNormalizer.load(self.normalizer_path)
        self.model = self._load_model()

    def _load_model(self) -> ElectricFieldMeshGraphNet:
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        input_dim = int(checkpoint.get("input_dim", len(self.graph_config.feature_names)))
        model = ElectricFieldMeshGraphNet(
            input_dim=input_dim,
            hidden_dim=self.model_config.hidden_dim,
            num_message_passing_steps=self.model_config.num_message_passing_steps,
            position_dim=self.model_config.position_dim,
            output_dim=self.model_config.output_dim,
            activation=self.model_config.activation,
            dropout=self.model_config.dropout,
            use_grad_checkpoint=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_graph(self, graph) -> np.ndarray:
        """Predict physical-unit `[Ex, Ey]` for an existing graph object."""

        graph = graph.to(self.device)
        pred_norm = self.model(graph).detach().cpu().numpy()
        return self.normalizer.inverse_targets(pred_norm)

    @torch.no_grad()
    def predict_arrays(
        self,
        pos: np.ndarray,
        edge_index: np.ndarray,
        doping: np.ndarray,
        vds: float,
        vgs: float,
        temperature: float,
        die_xy: Sequence[float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Build a graph from arrays and predict physical-unit `[Ex, Ey]`."""

        pos3 = as_3d_position(pos)
        x = self.normalizer.transform_inputs(
            pos=pos3,
            doping=doping,
            vds=vds,
            vgs=vgs,
            temperature=temperature,
            die_xy=die_xy,
        )
        graph = SimpleGraphData(
            x=torch.from_numpy(x).float(),
            edge_index=torch.from_numpy(edge_index.astype(np.int64)).long(),
            pos=torch.from_numpy(pos3).float(),
        ).to(self.device)
        pred_norm = self.model(graph).detach().cpu().numpy()
        return self.normalizer.inverse_targets(pred_norm)

