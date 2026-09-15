"""Export real example-level geometry for RIFT mechanism figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import build_predictor, last_layer_gradient_embedding
from run_continual import load_split


def train_reference(model: nn.Module, data: dict[str, np.ndarray], seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data["inputs"]), torch.from_numpy(data["targets"])),
        batch_size=128,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.1)
    model.train()
    for _ in range(15):
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=4000)
    args = parser.parse_args()

    data = load_split(args.cache, "baseline", "train", args.limit, args.seed, "none")
    torch.manual_seed(args.seed)
    model = build_predictor(
        "mlp", data["inputs"].shape[1], data["targets"].shape[1]
    )
    train_reference(model, data, args.seed)
    gradient, magnitude = last_layer_gradient_embedding(
        model,
        data["inputs"],
        data["targets"],
        seed=args.seed,
        projection_dim=64,
    )

    observable = RobustScaler().fit_transform(data["observable_embedding"])
    observable_xy = PCA(n_components=2, random_state=args.seed).fit_transform(observable)
    gradient_xy = PCA(n_components=2, random_state=args.seed).fit_transform(gradient)
    magnitude_rank = pd.Series(magnitude).rank(method="average", pct=True).to_numpy()
    risk_rank = pd.Series(data["risk_score"]).rank(method="average", pct=True).to_numpy()

    output = pd.DataFrame(
        {
            "example_index": np.arange(len(data["inputs"]), dtype=np.int64),
            "observable_pc1": observable_xy[:, 0],
            "observable_pc2": observable_xy[:, 1],
            "gradient_pc1": gradient_xy[:, 0],
            "gradient_pc2": gradient_xy[:, 1],
            "gradient_magnitude": magnitude,
            "gradient_magnitude_rank": magnitude_rank,
            "risk_score": data["risk_score"],
            "risk_rank": risk_rank,
            "risk_distance": data["risk_distance"],
            "high_risk": data["risk_distance"] <= 1.0,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.describe(include="all").to_string())
    print("spearman risk vs gradient magnitude:", output[["risk_score", "gradient_magnitude"]].corr(method="spearman").iloc[0, 1])


if __name__ == "__main__":
    main()
