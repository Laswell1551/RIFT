"""Shared lightweight predictor used by all continual-learning methods."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


TRAJECTORY_SCALE = np.asarray((100.0, 100.0, 30.0), np.float32)
CONTEXT_SCALE = np.asarray((25.0, 70.0, 15.0, 15.0, 25.0, 5.0), np.float32)


def model_inputs(
    obs: np.ndarray, context: np.ndarray, interaction: np.ndarray | None = None
) -> np.ndarray:
    trajectory = (obs / TRAJECTORY_SCALE).reshape(len(obs), -1)
    blocks = [trajectory, context / CONTEXT_SCALE]
    if interaction is not None:
        blocks.append(interaction.reshape(len(interaction), -1))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def model_targets(future: np.ndarray) -> np.ndarray:
    return (future / TRAJECTORY_SCALE).reshape(len(future), -1).astype(np.float32)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int = 27, hidden_dim: int = 128, output_dim: int = 30):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = nn.functional.silu(self.normalization(self.input_layer(inputs)))
        return nn.functional.silu(self.hidden_layer(hidden))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.features(inputs))


class TrajectoryDenoiser(nn.Module):
    """Residual denoiser for the normalized observed-trajectory block."""

    def __init__(
        self,
        observed_steps: int = 5,
        trajectory_dimensions: int = 3,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.observed_steps = observed_steps
        self.trajectory_dimensions = trajectory_dimensions
        width = observed_steps * trajectory_dimensions
        self.network = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        width = self.observed_steps * self.trajectory_dimensions
        trajectory = inputs[:, :width]
        correction = self.network(trajectory).reshape(
            -1, self.observed_steps, self.trajectory_dimensions
        )
        # The final observed point is the local origin after preprocessing.
        correction = correction - correction[:, -1:, :]
        denoised = inputs.clone()
        denoised[:, :width] = (
            trajectory.reshape(-1, self.observed_steps, self.trajectory_dimensions)
            + correction
        ).reshape(-1, width)
        return denoised


class DenoisingPredictor(nn.Module):
    """NATRA-inspired denoising plug-in around a trajectory predictor."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.denoiser = TrajectoryDenoiser()
        self.backbone = backbone
        # Gradient-memory code uses the final layer explicitly.
        self.output_layer = backbone.output_layer

    def denoise(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.denoiser(inputs)

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(self.denoise(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.denoise(inputs))


class TemporalTransformer(nn.Module):
    """Compact shared temporal backbone for architecture-robustness checks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 30,
        observed_steps: int = 5,
        trajectory_dimensions: int = 3,
    ):
        super().__init__()
        self.observed_steps = observed_steps
        self.trajectory_dimensions = trajectory_dimensions
        trajectory_width = observed_steps * trajectory_dimensions
        context_dim = input_dim - trajectory_width
        if context_dim <= 0:
            raise ValueError("TemporalTransformer requires context features after the trajectory")
        self.trajectory_projection = nn.Linear(trajectory_dimensions, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, observed_steps, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.position_embedding, std=0.02)

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        trajectory_width = self.observed_steps * self.trajectory_dimensions
        trajectory = inputs[:, :trajectory_width].reshape(
            -1, self.observed_steps, self.trajectory_dimensions
        )
        context = inputs[:, trajectory_width:]
        tokens = self.trajectory_projection(trajectory) + self.position_embedding
        temporal = self.temporal_encoder(tokens).mean(dim=1)
        operational = self.context_projection(context)
        return self.fusion(torch.cat((temporal, operational), dim=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.features(inputs))


class InteractionSetTransformer(nn.Module):
    """Temporal predictor with permutation-invariant multi-neighbor attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 30,
        observed_steps: int = 5,
        trajectory_dimensions: int = 3,
        context_dim: int = 6,
        neighbor_count: int = 8,
        neighbor_dim: int = 9,
    ):
        super().__init__()
        self.observed_steps = observed_steps
        self.trajectory_dimensions = trajectory_dimensions
        self.context_dim = context_dim
        self.neighbor_count = neighbor_count
        self.neighbor_dim = neighbor_dim
        expected = observed_steps * trajectory_dimensions + context_dim + neighbor_count * neighbor_dim
        if input_dim != expected:
            raise ValueError(f"InteractionSetTransformer expected input_dim={expected}, found {input_dim}")

        self.trajectory_projection = nn.Linear(trajectory_dimensions, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, observed_steps, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.neighbor_projection = nn.Sequential(
            nn.Linear(neighbor_dim - 1, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.query_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _blocks(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory_width = self.observed_steps * self.trajectory_dimensions
        trajectory = inputs[:, :trajectory_width].reshape(
            -1, self.observed_steps, self.trajectory_dimensions
        )
        context_start = trajectory_width
        context = inputs[:, context_start : context_start + self.context_dim]
        neighbors = inputs[:, context_start + self.context_dim :].reshape(
            -1, self.neighbor_count, self.neighbor_dim
        )
        mask = neighbors[:, :, -1] > 0.5
        return trajectory, context, neighbors[:, :, :-1], mask

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        trajectory, context, neighbors, mask = self._blocks(inputs)
        tokens = self.trajectory_projection(trajectory) + self.position_embedding
        temporal = self.temporal_encoder(tokens).mean(dim=1)
        operational = self.context_projection(context)
        neighbor_tokens = self.neighbor_projection(neighbors)
        query = self.query_projection(torch.cat((temporal, operational), dim=1))[:, None, :]
        score = torch.sum(query * self.key_projection(neighbor_tokens), dim=2) / np.sqrt(
            neighbor_tokens.shape[-1]
        )
        score = score.masked_fill(~mask, -1e4)
        attention = torch.softmax(score, dim=1) * mask.to(score.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        interaction = torch.sum(
            attention[:, :, None] * self.value_projection(neighbor_tokens), dim=1
        )
        return self.fusion(torch.cat((temporal, operational, interaction), dim=1))

    def interaction_attention(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized neighbor attention for mechanism diagnostics."""
        trajectory, context, neighbors, mask = self._blocks(inputs)
        temporal = self.temporal_encoder(
            self.trajectory_projection(trajectory) + self.position_embedding
        ).mean(dim=1)
        operational = self.context_projection(context)
        neighbor_tokens = self.neighbor_projection(neighbors)
        query = self.query_projection(torch.cat((temporal, operational), dim=1))[:, None, :]
        score = torch.sum(query * self.key_projection(neighbor_tokens), dim=2) / np.sqrt(
            neighbor_tokens.shape[-1]
        )
        score = score.masked_fill(~mask, -1e4)
        attention = torch.softmax(score, dim=1) * mask.to(score.dtype)
        return attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.features(inputs))


class HiVTStyleTransformer(nn.Module):
    """High-capacity clean-room bridge following HiVT's hierarchy.

    The model keeps HiVT's temporal vector encoding, local interaction encoding,
    stacked global interaction attention, and future decoding principles, while
    adapting them to the fixed RIFT tensor interface. It is not an exact
    reproduction of the Argoverse model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 30,
        observed_steps: int = 5,
        trajectory_dimensions: int = 3,
        context_dim: int = 6,
        neighbor_count: int = 8,
        neighbor_dim: int = 9,
    ):
        super().__init__()
        self.observed_steps = observed_steps
        self.trajectory_dimensions = trajectory_dimensions
        self.context_dim = context_dim
        self.neighbor_count = neighbor_count
        self.neighbor_dim = neighbor_dim
        expected = observed_steps * trajectory_dimensions + context_dim + neighbor_count * neighbor_dim
        if input_dim != expected:
            raise ValueError(f"HiVTStyleTransformer expected input_dim={expected}, found {input_dim}")
        if output_dim % trajectory_dimensions:
            raise ValueError("Output width must be divisible by trajectory dimensions")
        self.future_steps = output_dim // trajectory_dimensions

        self.trajectory_projection = nn.Linear(trajectory_dimensions, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, observed_steps, hidden_dim))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=4, norm=nn.LayerNorm(hidden_dim)
        )
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.neighbor_projection = nn.Sequential(
            nn.Linear(neighbor_dim - 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        local_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_interactor = nn.TransformerEncoder(
            local_layer, num_layers=2, norm=nn.LayerNorm(hidden_dim)
        )
        self.ownship_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.token_type_embedding = nn.Parameter(torch.zeros(1, 1 + neighbor_count, hidden_dim))
        global_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_interactor = nn.TransformerEncoder(
            global_layer, num_layers=3, norm=nn.LayerNorm(hidden_dim)
        )
        self.future_queries = nn.Parameter(torch.zeros(1, self.future_steps, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.future_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=2, norm=nn.LayerNorm(hidden_dim)
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.token_type_embedding, std=0.02)
        nn.init.normal_(self.future_queries, std=0.02)

    def _blocks(self, inputs: torch.Tensor):
        trajectory_width = self.observed_steps * self.trajectory_dimensions
        trajectory = inputs[:, :trajectory_width].reshape(
            -1, self.observed_steps, self.trajectory_dimensions
        )
        context = inputs[:, trajectory_width : trajectory_width + self.context_dim]
        neighbors = inputs[:, trajectory_width + self.context_dim :].reshape(
            -1, self.neighbor_count, self.neighbor_dim
        )
        valid = neighbors[:, :, -1] > 0.5
        return trajectory, context, neighbors[:, :, :-1], valid

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        trajectory, context, neighbors, valid = self._blocks(inputs)
        temporal_tokens = self.trajectory_projection(trajectory) + self.position_embedding
        temporal = self.temporal_encoder(temporal_tokens).mean(dim=1)
        operational = self.context_projection(context)
        ownship = self.ownship_fusion(torch.cat((temporal, operational), dim=1))

        neighbor_tokens = self.neighbor_projection(neighbors)
        safe_valid = valid.clone()
        safe_valid[:, 0] = True
        neighbor_tokens = self.local_interactor(
            neighbor_tokens, src_key_padding_mask=~safe_valid
        )
        neighbor_tokens = neighbor_tokens * valid[:, :, None].to(neighbor_tokens.dtype)
        scene_tokens = torch.cat((ownship[:, None, :], neighbor_tokens), dim=1)
        scene_tokens = scene_tokens + self.token_type_embedding
        scene_mask = torch.cat(
            (torch.zeros(len(inputs), 1, dtype=torch.bool, device=inputs.device), ~valid),
            dim=1,
        )
        global_tokens = self.global_interactor(
            scene_tokens, src_key_padding_mask=scene_mask
        )
        queries = self.future_queries.expand(len(inputs), -1, -1)
        future = self.future_decoder(
            queries, global_tokens, memory_key_padding_mask=scene_mask
        ).mean(dim=1)
        return self.feature_fusion(torch.cat((global_tokens[:, 0], future), dim=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.features(inputs))

def build_predictor(
    backbone: str,
    input_dim: int,
    output_dim: int,
    neighbor_count: int = 8,
    neighbor_dim: int = 9,
) -> nn.Module:
    if backbone == "mlp":
        return ResidualMLP(input_dim=input_dim, output_dim=output_dim)
    if backbone == "temporal_transformer":
        return TemporalTransformer(input_dim=input_dim, output_dim=output_dim)
    if backbone == "interaction_transformer":
        return InteractionSetTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            neighbor_count=neighbor_count,
            neighbor_dim=neighbor_dim,
        )
    if backbone == "hivt_style_transformer":
        return HiVTStyleTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            neighbor_count=neighbor_count,
            neighbor_dim=neighbor_dim,
        )
    raise ValueError(f"Unknown backbone: {backbone}")


def last_layer_gradient_embedding(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    projection_dim: int = 64,
    seed: int = 913,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compressed per-example squared-loss gradient of the final layer."""
    blocks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = torch.from_numpy(inputs[start : start + batch_size])
            y = torch.from_numpy(targets[start : start + batch_size])
            features = model.features(x)
            residual = model.output_layer(features) - y
            gradient = torch.einsum("no,nh->noh", residual, features).flatten(1)
            gradient = torch.cat((gradient, residual), dim=1)
            blocks.append(gradient.cpu().numpy().astype(np.float32))
    full = np.concatenate(blocks, axis=0)
    magnitude = np.linalg.norm(full, axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    projection = rng.choice((-1.0, 1.0), size=(full.shape[1], projection_dim)).astype(np.float32)
    projection /= np.sqrt(projection_dim)
    compressed = full @ projection
    norm = np.linalg.norm(compressed, axis=1, keepdims=True)
    embedding = (compressed / np.maximum(norm, 1e-6)).astype(np.float32)
    return embedding, magnitude


def predict_numpy(model: nn.Module, inputs: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            tensor = torch.from_numpy(inputs[start : start + batch_size])
            outputs.append(model(tensor).cpu().numpy())
    flat = np.concatenate(outputs, axis=0)
    dimensions = flat.shape[1] // 10
    scale = TRAJECTORY_SCALE[:dimensions]
    return flat.reshape(-1, 10, dimensions) * scale




