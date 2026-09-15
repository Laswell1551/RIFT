"""Run shared-backbone continual trajectory-prediction experiments."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from memory import CGSMDualMemory, H2CAdaptedMemory, ReplayMemory, concatenate, coverage_diagnostics
from models import (
    TRAJECTORY_SCALE, DenoisingPredictor, build_predictor,
    last_layer_gradient_embedding, model_inputs, model_targets, predict_numpy,
)


def observable_memory_embedding(
    inputs: np.ndarray, risk_score: np.ndarray, risk_distance: np.ndarray
) -> np.ndarray:
    finite_distance = np.where(np.isfinite(risk_distance), risk_distance, 10.0)
    risk_features = np.column_stack((risk_score, np.clip(finite_distance, 0.0, 10.0) / 10.0))
    return np.concatenate((inputs, risk_features.astype(np.float32)), axis=1).astype(np.float32)


class EWCState:
    def __init__(self, strength: float = 20.0):
        self.strength = strength
        self.fisher: dict[str, torch.Tensor] | None = None
        self.anchor: dict[str, torch.Tensor] | None = None

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if self.fisher is None or self.anchor is None:
            return torch.zeros((), dtype=torch.float32)
        return self.strength * sum(
            (self.fisher[name] * (parameter - self.anchor[name]).square()).sum()
            for name, parameter in model.named_parameters()
        )

    def consolidate(self, model: nn.Module, data: dict[str, np.ndarray], batch_size: int) -> None:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(data["inputs"]), torch.from_numpy(data["targets"])),
            batch_size=batch_size, shuffle=False,
        )
        fisher = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
        criterion = nn.SmoothL1Loss(beta=0.1)
        model.eval()
        batches = 0
        for inputs, targets in loader:
            model.zero_grad(set_to_none=True)
            criterion(model(inputs), targets).backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().square()
            batches += 1
        for name in fisher:
            fisher[name] /= max(batches, 1)
        if self.fisher is None:
            self.fisher = fisher
        else:
            self.fisher = {name: self.fisher[name] + fisher[name] for name in fisher}
        self.anchor = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def load_split(
    cache: Path,
    task: str,
    split: str,
    limit: int | None,
    seed: int,
    interaction_mode: str = "auto",
) -> dict[str, np.ndarray]:
    archive = np.load(cache / f"{task}__{split}.npz")
    n = len(archive["obs"])
    indices = np.arange(n)
    if limit is not None and n > limit:
        indices = np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))
    if interaction_mode == "set":
        if "interaction_set" not in archive.files:
            raise KeyError(f"{cache} does not contain interaction_set")
        interaction = archive["interaction_set"][indices]
    elif interaction_mode == "critical":
        if "interaction" not in archive.files:
            raise KeyError(f"{cache} does not contain interaction")
        interaction = archive["interaction"][indices]
    elif interaction_mode == "none":
        interaction = None
    elif interaction_mode == "auto":
        interaction = archive["interaction"][indices] if "interaction" in archive.files else None
    else:
        raise ValueError(f"Unknown interaction_mode: {interaction_mode}")
    inputs = model_inputs(archive["obs"][indices], archive["context"][indices], interaction)
    future = archive["future"][indices].astype(np.float32)
    risk_score = archive["risk_score"][indices].astype(np.float32)
    risk_distance = archive["risk_distance"][indices].astype(np.float32)
    return {
        "inputs": inputs,
        "targets": model_targets(future),
        "future": future,
        "risk_score": risk_score,
        "risk_distance": risk_distance,
        "observable_embedding": observable_memory_embedding(inputs, risk_score, risk_distance),
    }


def with_task(data: dict[str, np.ndarray], task_index: int) -> dict[str, np.ndarray]:
    copied = {key: value.copy() for key, value in data.items()}
    copied["task_index"] = np.full(len(data["inputs"]), task_index, np.int16)
    copied["selection_magnitude"] = np.zeros(len(data["inputs"]), np.float32)
    copied["replay_priority"] = np.zeros(len(data["inputs"]), np.float32)
    return copied


def training_arrays(
    current: dict[str, np.ndarray], replay: dict[str, np.ndarray] | None,
    method: str, cfg: dict, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if replay is None or not len(replay["inputs"]):
        return current["inputs"], current["targets"]
    count = len(current["inputs"])
    if method == "cgsm_dual" and np.any(replay["replay_priority"] > 0.5):
        rng = np.random.default_rng(seed)
        safety = np.flatnonzero(replay["replay_priority"] > 0.5)
        general = np.flatnonzero(replay["replay_priority"] <= 0.5)
        safety_count = int(round(count * float(cfg.get("safety_replay_fraction", 0.2))))
        safety_count = min(max(safety_count, 0), count)
        general_count = count - safety_count
        safety_draw = rng.choice(safety, safety_count, replace=True)
        general_pool = general if len(general) else np.arange(len(replay["inputs"]))
        general_draw = rng.choice(general_pool, general_count, replace=True)
        draw = np.concatenate((safety_draw, general_draw))
        rng.shuffle(draw)
        replay_x = replay["inputs"][draw]
        replay_y = replay["targets"][draw]
    else:
        repeats = int(np.ceil(count / len(replay["inputs"])))
        replay_x = np.tile(replay["inputs"], (repeats, 1))[:count]
        replay_y = np.tile(replay["targets"], (repeats, 1))[:count]
    return np.concatenate((current["inputs"], replay_x)), np.concatenate((current["targets"], replay_y))


def fit_task(
    model: nn.Module,
    current: dict[str, np.ndarray],
    replay: dict[str, np.ndarray] | None,
    cfg: dict,
    seed: int,
    method: str,
    ewc_state: EWCState,
    slow_model: nn.Module | None = None,
) -> float:
    if method == "gss_denoise":
        if not isinstance(model, DenoisingPredictor):
            raise TypeError("gss_denoise requires DenoisingPredictor")
        return fit_denoising(model, current, replay, cfg, seed)
    if method == "cgsm_loss":
        return fit_risk_calibrated(model, current, replay, cfg, seed)
    if method == "agem" and replay is not None:
        return fit_agem(model, current, replay, cfg, seed)
    if method == "derpp" and replay is not None:
        return fit_derpp(model, current, replay, cfg, seed)
    if method == "syrem_adapted" and replay is not None:
        return fit_syrem(model, current, replay, cfg, seed)
    if method == "dualls_adapted":
        if slow_model is None:
            raise ValueError("dualls_adapted requires a slow model")
        return fit_dualls(model, slow_model, current, replay, cfg, seed)
    x, y = training_arrays(current, replay, method, cfg, seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=cfg["batch_size"], shuffle=True, generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.1)
    start = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs_per_task"]):
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            if method == "ewc":
                loss = loss + ewc_state.penalty(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return time.perf_counter() - start


def corrupt_inputs_tensor(
    inputs: torch.Tensor,
    sigma_m: torch.Tensor,
    generator: torch.Generator,
    observed_steps: int = 5,
    trajectory_dimensions: int = 3,
) -> torch.Tensor:
    """Apply paired 3-D Gaussian noise to normalized trajectory inputs."""
    width = observed_steps * trajectory_dimensions
    noisy = inputs.clone()
    trajectory = noisy[:, :width].reshape(-1, observed_steps, trajectory_dimensions)
    noise = torch.randn(
        trajectory.shape, generator=generator, device=trajectory.device, dtype=trajectory.dtype
    )
    scale = torch.as_tensor(
        TRAJECTORY_SCALE[:trajectory_dimensions], dtype=trajectory.dtype, device=trajectory.device
    )
    noise = noise * sigma_m[:, None, None] / scale[None, None, :]
    noise = noise - noise[:, -1:, :]
    trajectory.add_(noise)
    return noisy


def fit_denoising(
    model: DenoisingPredictor,
    current: dict[str, np.ndarray],
    replay: dict[str, np.ndarray] | None,
    cfg: dict,
    seed: int,
) -> float:
    """NATRA-inspired reconstruction and ranking training for GSS replay."""
    x, y = training_arrays(current, replay, "gss_denoise", cfg, seed)
    generator = torch.Generator().manual_seed(seed)
    noise_generator = torch.Generator().manual_seed(seed + 100_003)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=cfg["batch_size"], shuffle=True, generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4
    )
    levels = torch.tensor(
        cfg.get("denoise_train_noise_std_m", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
        dtype=torch.float32,
    )
    rec_weight = float(cfg.get("denoise_reconstruction_weight", 1.0))
    rank_weight = float(cfg.get("denoise_ranking_weight", 0.2))
    clean_weight = float(cfg.get("denoise_clean_prediction_weight", 0.5))
    margin = float(cfg.get("denoise_ranking_margin", 0.002))
    width = 5 * 3
    start = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs_per_task"]):
        for inputs, targets in loader:
            level_index = torch.randint(
                len(levels), (len(inputs),), generator=noise_generator
            )
            noisy = corrupt_inputs_tensor(inputs, levels[level_index], noise_generator)
            denoised_inputs = model.denoise(noisy)
            denoised_prediction = model.backbone(denoised_inputs)
            clean_prediction = model(inputs)
            raw_noisy_prediction = model.backbone(noisy)

            denoised_item = nn.functional.smooth_l1_loss(
                denoised_prediction, targets, beta=0.1, reduction="none"
            ).mean(dim=1)
            clean_loss = nn.functional.smooth_l1_loss(
                clean_prediction, targets, beta=0.1
            )
            reconstruction = nn.functional.mse_loss(
                denoised_inputs[:, :width], inputs[:, :width]
            )
            with torch.no_grad():
                noisy_item = nn.functional.smooth_l1_loss(
                    raw_noisy_prediction, targets, beta=0.1, reduction="none"
                ).mean(dim=1)
            ranking = torch.relu(denoised_item - noisy_item + margin).mean()
            loss = (
                denoised_item.mean()
                + clean_weight * clean_loss
                + rec_weight * reconstruction
                + rank_weight * ranking
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return time.perf_counter() - start

def fit_syrem(
    model: nn.Module,
    current: dict[str, np.ndarray],
    replay: dict[str, np.ndarray],
    cfg: dict,
    seed: int,
) -> float:
    """Shared-backbone SyReM adaptation with aligned rehearsal and projection.

    The memory is gradient-diverse. For each current batch, stored examples are
    ranked by cosine similarity to the batch's last-layer gradient direction.
    A-GEM projection then prevents an increase along the selected replay
    gradient. The method is labeled adapted because it does not reproduce the
    original predictor or one-pass stream.
    """
    current_embedding, _ = last_layer_gradient_embedding(
        model,
        current["inputs"],
        current["targets"],
        seed=int(cfg["seed"]),
    )
    indices = np.arange(len(current["inputs"]), dtype=np.int64)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(current["inputs"]),
            torch.from_numpy(current["targets"]),
            torch.from_numpy(indices),
        ),
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.1)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    rehearsal_ratio = float(cfg.get("syrem_rehearsal_ratio", 1.0))
    start = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs_per_task"]):
        for inputs, targets, batch_index in loader:
            direction = current_embedding[batch_index.numpy()].mean(axis=0)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            similarity = replay["selection_embedding"] @ direction
            count = min(
                max(1, int(round(len(inputs) * rehearsal_ratio))),
                len(replay["inputs"]),
            )
            selected = np.argpartition(similarity, -count)[-count:]

            optimizer.zero_grad(set_to_none=True)
            criterion(model(inputs), targets).backward()
            current_grad = [
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
                for parameter in parameters
            ]

            optimizer.zero_grad(set_to_none=True)
            replay_loss = criterion(
                model(torch.from_numpy(replay["inputs"][selected])),
                torch.from_numpy(replay["targets"][selected]),
            )
            replay_loss.backward()
            memory_grad = [
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
                for parameter in parameters
            ]
            dot = sum((left * right).sum() for left, right in zip(current_grad, memory_grad))
            norm = sum((component * component).sum() for component in memory_grad).clamp_min(1e-12)
            coefficient = torch.minimum(dot / norm, torch.zeros_like(dot))
            for parameter, gradient, memory_component in zip(
                parameters, current_grad, memory_grad
            ):
                parameter.grad = gradient - coefficient * memory_component
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
    return time.perf_counter() - start


def fit_dualls(
    model: nn.Module,
    slow_model: nn.Module,
    current: dict[str, np.ndarray],
    replay: dict[str, np.ndarray] | None,
    cfg: dict,
    seed: int,
) -> float:
    """Shared-backbone Dual-LS adaptation with fast/slow model coordination."""
    x, y = training_arrays(current, replay, "dualls_adapted", cfg, seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.1)
    decay = float(cfg.get("dualls_slow_decay", 0.99))
    start = time.perf_counter()
    model.train()
    slow_model.eval()
    for _ in range(cfg["epochs_per_task"]):
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            with torch.no_grad():
                slow_prediction = slow_model(inputs)
            loss = loss + float(cfg.get("dualls_consistency", 0.1)) * nn.functional.mse_loss(
                prediction, slow_prediction
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                for slow_parameter, fast_parameter in zip(
                    slow_model.parameters(), model.parameters()
                ):
                    slow_parameter.mul_(decay).add_(fast_parameter, alpha=1.0 - decay)
    return time.perf_counter() - start


def fit_risk_calibrated(
    model: nn.Module,
    current: dict[str, np.ndarray],
    replay: dict[str, np.ndarray] | None,
    cfg: dict,
    seed: int,
) -> float:
    """GSS replay with a risk-reweighted loss and unchanged update budget."""
    if replay is None or not len(replay["inputs"]):
        x, y, risk = current["inputs"], current["targets"], current["risk_score"]
    else:
        count = len(current["inputs"])
        repeats = int(np.ceil(count / len(replay["inputs"])))
        replay_x = np.tile(replay["inputs"], (repeats, 1))[:count]
        replay_y = np.tile(replay["targets"], (repeats, 1))[:count]
        replay_risk = np.tile(replay["risk_score"], repeats)[:count]
        x = np.concatenate((current["inputs"], replay_x))
        y = np.concatenate((current["targets"], replay_y))
        risk = np.concatenate((current["risk_score"], replay_risk))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(risk)),
        batch_size=cfg["batch_size"], shuffle=True, generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    strength = float(cfg.get("risk_strength", 1.0))
    power = float(cfg.get("risk_power", 1.0))
    start = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs_per_task"]):
        for inputs, targets, risk_score in loader:
            elementwise = nn.functional.smooth_l1_loss(
                model(inputs), targets, beta=0.1, reduction="none"
            ).mean(dim=1)
            weights = 1.0 + strength * torch.pow(torch.clamp_min(risk_score, 1e-8), power)
            loss = torch.sum(weights * elementwise) / torch.sum(weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return time.perf_counter() - start


def fit_agem(model: nn.Module, current: dict[str, np.ndarray], replay: dict[str, np.ndarray], cfg: dict, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(current["inputs"]), torch.from_numpy(current["targets"])),
        batch_size=cfg["batch_size"], shuffle=True, generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.1)
    rng = np.random.default_rng(seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    start = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs_per_task"]):
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(inputs), targets).backward()
            current_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in parameters]
            count = min(len(inputs), len(replay["inputs"]))
            index = rng.choice(len(replay["inputs"]), count, replace=False)
            optimizer.zero_grad(set_to_none=True)
            replay_loss = criterion(
                model(torch.from_numpy(replay["inputs"][index])),
                torch.from_numpy(replay["targets"][index]),
            )
            replay_loss.backward()
            memory_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in parameters]
            dot = sum((g * m).sum() for g, m in zip(current_grad, memory_grad))
            norm = sum((m * m).sum() for m in memory_grad).clamp_min(1e-12)
            coefficient = torch.minimum(dot / norm, torch.zeros_like(dot))
            for parameter, grad, memory_component in zip(parameters, current_grad, memory_grad):
                parameter.grad = grad - coefficient * memory_component
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
    return time.perf_counter() - start


def fit_derpp(model: nn.Module, current: dict[str, np.ndarray], replay: dict[str, np.ndarray], cfg: dict, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(current["inputs"]), torch.from_numpy(current["targets"])),
        batch_size=cfg["batch_size"], shuffle=True, generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.1)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs_per_task"]):
        for inputs, targets in loader:
            count = min(len(inputs), len(replay["inputs"]))
            index = rng.choice(len(replay["inputs"]), count, replace=False)
            replay_inputs = torch.from_numpy(replay["inputs"][index])
            prediction = model(inputs)
            replay_prediction = model(replay_inputs)
            loss = criterion(prediction, targets)
            loss = loss + 0.5 * criterion(replay_prediction, torch.from_numpy(replay["targets"][index]))
            loss = loss + 0.5 * nn.functional.mse_loss(
                replay_prediction, torch.from_numpy(replay["logits"][index])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return time.perf_counter() - start


def evaluate(
    model: nn.Module,
    data: dict[str, np.ndarray],
    auxiliary_model: nn.Module | None = None,
    blend: float = 0.5,
) -> dict[str, float]:
    prediction = predict_numpy(model, data["inputs"])
    if auxiliary_model is not None:
        prediction = (
            (1.0 - blend) * prediction
            + blend * predict_numpy(auxiliary_model, data["inputs"])
        )
    displacement = np.linalg.norm(prediction - data["future"], axis=2)
    high_risk = data["risk_distance"] <= 1.0
    metrics = {
        "ade": float(displacement.mean()),
        "fde": float(displacement[:, -1].mean()),
        "high_risk_fraction": float(high_risk.mean()),
    }
    for horizon in (1, 3, 5, 7, 10):
        if horizon > displacement.shape[1]:
            continue
        metrics[f"ade_h{horizon}"] = float(displacement[:, :horizon].mean())
        metrics[f"fde_h{horizon}"] = float(displacement[:, horizon - 1].mean())
    if high_risk.any():
        metrics["risk_ade"] = float(displacement[high_risk].mean())
        metrics["risk_fde"] = float(displacement[high_risk, -1].mean())
        for horizon in (1, 3, 5, 7, 10):
            if horizon > displacement.shape[1]:
                continue
            metrics[f"risk_ade_h{horizon}"] = float(
                displacement[high_risk, :horizon].mean()
            )
            metrics[f"risk_fde_h{horizon}"] = float(
                displacement[high_risk, horizon - 1].mean()
            )
    else:
        metrics["risk_ade"] = float("nan")
        metrics["risk_fde"] = float("nan")
        for horizon in (1, 3, 5, 7, 10):
            if horizon <= displacement.shape[1]:
                metrics[f"risk_ade_h{horizon}"] = float("nan")
                metrics[f"risk_fde_h{horizon}"] = float("nan")
    cpa_bands = (
        ("cpa_0_0p5", 0.0, 0.5),
        ("cpa_0p5_1", 0.5, 1.0),
        ("cpa_1_2", 1.0, 2.0),
        ("cpa_ge_2", 2.0, float("inf")),
    )
    for name, lower, upper in cpa_bands:
        selected = (data["risk_distance"] >= lower) & (data["risk_distance"] < upper)
        metrics[f"{name}_count"] = int(selected.sum())
        metrics[f"{name}_fde"] = (
            float(displacement[selected, -1].mean()) if selected.any() else float("nan")
        )
    # Secondary threshold-sensitivity diagnostics. The primary metric remains
    # the preregistered d_CPA <= 1.0 definition above; these fixed thresholds
    # test whether qualitative conclusions hinge on that single cutoff.
    for threshold in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        selected = data["risk_distance"] <= threshold
        tag = str(float(threshold)).replace(".", "p")
        metrics[f"risk_thr_{tag}_count"] = int(selected.sum())
        metrics[f"risk_thr_{tag}_fraction"] = float(selected.mean())
        metrics[f"risk_thr_{tag}_ade"] = (
            float(displacement[selected].mean()) if selected.any() else float("nan")
        )
        metrics[f"risk_thr_{tag}_fde"] = (
            float(displacement[selected, -1].mean()) if selected.any() else float("nan")
        )
    return metrics


def perturb_observed_positions(
    inputs: np.ndarray,
    sigma_m: float,
    seed: int,
    observed_steps: int = 5,
    trajectory_dimensions: int = 3,
) -> np.ndarray:
    """Apply deterministic 3-D Gaussian position noise before local re-centering."""
    perturbed = inputs.copy()
    width = observed_steps * trajectory_dimensions
    trajectory = perturbed[:, :width].reshape(
        -1, observed_steps, trajectory_dimensions
    )
    rng = np.random.default_rng(seed)
    noise_m = rng.normal(0.0, sigma_m, size=trajectory.shape).astype(np.float32)
    noise_m -= noise_m[:, -1:, :]
    trajectory += noise_m / TRAJECTORY_SCALE[None, None, :]
    return perturbed


def run_method(method: str, cfg: dict, train: dict, test: dict, output: Path) -> None:
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    first_task = cfg["task_order"][0]
    model = build_predictor(
        str(cfg.get("backbone", "mlp")),
        input_dim=train[first_task]["inputs"].shape[1],
        output_dim=train[first_task]["targets"].shape[1],
        neighbor_count=int(cfg.get("neighbor_count", 8)),
        neighbor_dim=int(cfg.get("neighbor_dim", 9)),
    )
    if method == "gss_denoise":
        model = DenoisingPredictor(model)
    selector = {
        "ewc": "finetune", "agem": "reservoir", "derpp": "reservoir",
        "cgsm_loss": "gss_adapted",
        "gss_denoise": "gss_adapted",
        "syrem_adapted": "gss_adapted",
        "dualls_adapted": "h2c_adapted",
    }.get(method, method)
    if method in {"h2c_adapted", "dualls_adapted"}:
        memory = H2CAdaptedMemory(int(cfg["memory_budget"]), seed)
    elif method == "cgsm_dual":
        memory = CGSMDualMemory(
            int(cfg["memory_budget"]), seed, float(cfg.get("risk_power", 4.0)),
            float(cfg.get("general_fraction", 0.5)),
        )
    else:
        memory = ReplayMemory(
            int(cfg["memory_budget"]), selector, seed,
            temperature=float(cfg.get("facility_temperature", 0.75)),
            risk_power=float(cfg.get("risk_power", 2.0)),
            risk_strength=float(cfg.get("risk_strength", 1.0)),
        )
    ewc_state = EWCState(float(cfg.get("ewc_strength", 20.0)))
    slow_model = copy.deepcopy(model) if method == "dualls_adapted" else None
    rows = []
    historical_pool = None
    for step, task in enumerate(cfg["task_order"]):
        print(f"{method:12s} starting step={step} task={task}", flush=True)
        current = with_task(train[task], step)
        elapsed = fit_task(
            model,
            current,
            memory.data,
            cfg,
            seed + step,
            method,
            ewc_state,
            slow_model,
        )
        with torch.no_grad():
            current["logits"] = model(torch.from_numpy(current["inputs"])).cpu().numpy().astype(np.float32)
        if selector != "finetune":
            current["selection_embedding"], current["selection_magnitude"] = last_layer_gradient_embedding(
                model, current["inputs"], current["targets"], seed=seed
            )
        if selector != "finetune":
            historical_pool = concatenate(historical_pool, current)
        memory.update(current)
        if method == "ewc":
            ewc_state.consolidate(model, current, cfg["batch_size"])
        coverage = coverage_diagnostics(historical_pool, memory.data) if memory.data is not None else {
            "weighted_cover_mean": float("nan"), "high_risk_cover_max": float("nan")
        }
        for eval_index, eval_task in enumerate(cfg["task_order"][: step + 1]):
            if cfg.get("export_predictions", False) and (
                step == eval_index or step == len(cfg["task_order"]) - 1
            ):
                prediction = predict_numpy(model, test[eval_task]["inputs"])
                if slow_model is not None:
                    mix = float(cfg.get("dualls_blend", 0.5))
                    prediction = (1.0 - mix) * prediction + mix * predict_numpy(
                        slow_model, test[eval_task]["inputs"]
                    )
                snapshots = output / "predictions"
                snapshots.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    snapshots / f"{method}__seed{seed}__step{step}__{eval_task}.npz",
                    prediction=prediction.astype(np.float32),
                    fde=np.linalg.norm(
                        prediction[:, -1] - test[eval_task]["future"][:, -1], axis=1
                    ).astype(np.float32),
                )
            metrics = evaluate(
                model,
                test[eval_task],
                slow_model,
                float(cfg.get("dualls_blend", 0.5)),
            )
            if step == len(cfg["task_order"]) - 1:
                for level_index, sigma_m in enumerate(
                    cfg.get("observation_noise_std_m", [])
                ):
                    noisy_inputs = perturb_observed_positions(
                        test[eval_task]["inputs"],
                        float(sigma_m),
                        seed + 50_000 + 100 * eval_index + level_index,
                    )
                    noisy_data = {**test[eval_task], "inputs": noisy_inputs}
                    noisy_metrics = evaluate(
                        model,
                        noisy_data,
                        slow_model,
                        float(cfg.get("dualls_blend", 0.5)),
                    )
                    tag = str(float(sigma_m)).replace(".", "p")
                    for key in ("ade", "fde", "risk_ade", "risk_fde"):
                        metrics[f"noise_{tag}_{key}"] = noisy_metrics[key]
            rows.append({
                "method": method, "seed": seed, "learn_step": step,
                "learn_task": task, "eval_task_index": eval_index, "eval_task": eval_task,
                "train_seconds": elapsed, "memory_size": 0 if memory.data is None else len(memory.data["inputs"]),
                **coverage, **metrics,
            })
        print(f"{method:10s} step={step} task={task:10s} train={elapsed:.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(output / f"metrics__{method}__seed{seed}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/cgsm"))
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-order", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--memory-budget", type=int)
    parser.add_argument("--general-fraction", type=float)
    parser.add_argument("--risk-power", type=float)
    parser.add_argument("--safety-replay-fraction", type=float)
    parser.add_argument("--risk-strength", type=float)
    parser.add_argument("--backbone", choices=("mlp", "temporal_transformer", "interaction_transformer", "hivt_style_transformer"))
    parser.add_argument("--interaction-mode", choices=("auto", "none", "critical", "set"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.task_order:
        cfg["task_order"] = args.task_order
    if args.methods:
        cfg["methods"] = args.methods
    if args.memory_budget is not None:
        cfg["memory_budget"] = args.memory_budget
    if args.general_fraction is not None:
        cfg["general_fraction"] = args.general_fraction
    if args.risk_power is not None:
        cfg["risk_power"] = args.risk_power
    if args.safety_replay_fraction is not None:
        cfg["safety_replay_fraction"] = args.safety_replay_fraction
    if args.risk_strength is not None:
        cfg["risk_strength"] = args.risk_strength
    if args.backbone is not None:
        cfg["backbone"] = args.backbone
    if args.interaction_mode is not None:
        cfg["interaction_mode"] = args.interaction_mode
    torch.set_num_threads(int(cfg.get("torch_threads", 2)))
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = cfg["task_order"]
    interaction_mode = str(cfg.get("interaction_mode", "auto"))
    train = {
        task: load_split(
            args.cache,
            task,
            "train",
            cfg.get("train_limit_per_task"),
            cfg["seed"] + i,
            interaction_mode,
        )
        for i, task in enumerate(tasks)
    }
    test = {
        task: load_split(
            args.cache,
            task,
            args.eval_split,
            cfg.get("eval_limit_per_task"),
            cfg["seed"] + 100 + i,
            interaction_mode,
        )
        for i, task in enumerate(tasks)
    }
    cfg["evaluation_split"] = args.eval_split
    (args.output / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for method in cfg["methods"]:
        run_method(method, cfg, train, test, args.output)


if __name__ == "__main__":
    main()

