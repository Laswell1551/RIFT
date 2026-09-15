"""Fixed-budget replay memories and coverage diagnostics."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


ARRAY_KEYS = (
    "inputs", "targets", "future", "risk_score", "risk_distance",
    "task_index", "logits", "selection_embedding", "observable_embedding",
    "selection_magnitude", "replay_priority",
)


def subset(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: data[key][indices] for key in ARRAY_KEYS}


def concatenate(left: dict[str, np.ndarray] | None, right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if left is None or not len(left["inputs"]):
        return {key: value.copy() for key, value in right.items() if key in ARRAY_KEYS}
    return {key: np.concatenate((left[key], right[key]), axis=0) for key in ARRAY_KEYS}


def standardized_embedding(inputs: np.ndarray) -> np.ndarray:
    center = np.median(inputs, axis=0, keepdims=True)
    scale = np.median(np.abs(inputs - center), axis=0, keepdims=True) * 1.4826
    scale[scale < 1e-3] = 1.0
    embedding = (inputs - center) / scale
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    return (embedding / np.maximum(norm, 1e-6)).astype(np.float32)


def pairwise_sqdist(embedding: np.ndarray) -> np.ndarray:
    gram = embedding @ embedding.T
    squared = np.sum(embedding * embedding, axis=1, keepdims=True)
    return np.maximum(squared + squared.T - 2.0 * gram, 0.0).astype(np.float32)


def distance_to_center(embedding: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.linalg.norm(embedding - center[None, :], axis=1).astype(np.float32)


def kcenter_indices(embedding: np.ndarray, budget: int, first: int | None = None) -> np.ndarray:
    n = len(embedding)
    if n <= budget:
        return np.arange(n)
    selected = [int(np.argmax(np.linalg.norm(embedding, axis=1))) if first is None else int(first)]
    nearest = distance_to_center(embedding, embedding[selected[0]])
    for _ in range(1, budget):
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, distance_to_center(embedding, embedding[index]))
    return np.asarray(selected, np.int64)


def conditioned_kcenter_indices(
    embedding: np.ndarray, budget: int, initial_indices: np.ndarray
) -> np.ndarray:
    """Select new gradient centers after accounting for fixed safety anchors."""
    n = len(embedding)
    if budget <= 0 or n == 0:
        return np.empty(0, np.int64)
    initial = np.asarray(initial_indices, np.int64)
    available = np.ones(n, dtype=bool)
    available[initial] = False
    budget = min(budget, int(available.sum()))
    if len(initial):
        nearest = np.full(n, np.inf, np.float32)
        for index in initial:
            nearest = np.minimum(nearest, distance_to_center(embedding, embedding[index]))
    else:
        nearest = np.linalg.norm(embedding, axis=1).astype(np.float32)
    chosen: list[int] = []
    for _ in range(budget):
        score = np.where(available, nearest, -np.inf)
        index = int(np.argmax(score))
        chosen.append(index)
        available[index] = False
        nearest = np.minimum(nearest, distance_to_center(embedding, embedding[index]))
    return np.asarray(chosen, np.int64)


def risk_weighted_kcenter_indices(embedding: np.ndarray, weights: np.ndarray, budget: int) -> np.ndarray:
    """Greedy coverage whose uncovered-distance priority is risk weighted."""
    n = len(embedding)
    if n <= budget:
        return np.arange(n)
    weights = np.asarray(weights, np.float32)
    weights = weights / max(float(weights.max()), 1e-6)
    selected = [int(np.argmax(weights))]
    nearest = distance_to_center(embedding, embedding[selected[0]])
    for _ in range(1, budget):
        index = int(np.argmax(nearest * weights))
        selected.append(index)
        nearest = np.minimum(nearest, distance_to_center(embedding, embedding[index]))
    return np.asarray(selected, np.int64)


def lazy_facility_indices(
    embedding: np.ndarray,
    weights: np.ndarray,
    budget: int,
    temperature: float = 0.75,
) -> np.ndarray:
    """Exact lazy-greedy maximization of weighted RBF facility location."""
    n = len(embedding)
    if n <= budget:
        return np.arange(n)
    distances = pairwise_sqdist(embedding)
    similarity = np.exp(-distances / max(temperature, 1e-6)).astype(np.float32)
    weights = np.asarray(weights, np.float64)
    weights = weights / weights.sum()
    initial = similarity.T @ weights
    heap = [(-float(initial[j]), j, 0) for j in range(n)]
    heapq.heapify(heap)
    best = np.zeros(n, np.float32)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    iteration = 0
    while len(chosen) < budget:
        neg_bound, candidate, stamp = heapq.heappop(heap)
        if candidate in chosen_set:
            continue
        if stamp == iteration:
            chosen.append(candidate)
            chosen_set.add(candidate)
            best = np.maximum(best, similarity[:, candidate])
            iteration += 1
            continue
        gain = float(np.dot(weights, np.maximum(similarity[:, candidate] - best, 0.0)))
        heapq.heappush(heap, (-gain, candidate, iteration))
    return np.asarray(chosen, np.int64)


def coverage_diagnostics(data: dict[str, np.ndarray], memory: dict[str, np.ndarray]) -> dict[str, float]:
    if not len(data["inputs"]) or not len(memory["inputs"]):
        return {"weighted_cover_mean": float("nan"), "weighted_cover_max": float("nan")}
    combined = np.concatenate((data["observable_embedding"], memory["observable_embedding"]), axis=0)
    embedding = standardized_embedding(combined)
    source = embedding[: len(data["inputs"])]
    stored = embedding[len(data["inputs"]) :]
    distance = np.sqrt(np.maximum(
        np.sum(source * source, axis=1, keepdims=True)
        + np.sum(stored * stored, axis=1)[None, :]
        - 2.0 * source @ stored.T,
        0.0,
    )).min(axis=1)
    weights = np.maximum(data["risk_score"], 1e-8)
    weights = weights / weights.sum()
    return {
        "weighted_cover_mean": float(np.dot(weights, distance)),
        "high_risk_cover_max": float(np.max(distance[data["risk_distance"] <= 1.0]))
        if np.any(data["risk_distance"] <= 1.0) else float("nan"),
    }


@dataclass
class ReplayMemory:
    budget: int
    method: str
    seed: int
    temperature: float = 0.75
    risk_power: float = 2.0
    risk_strength: float = 1.0

    def __post_init__(self) -> None:
        self.data: dict[str, np.ndarray] | None = None
        self.seen = 0
        self.rng = np.random.default_rng(self.seed)

    def update(self, current: dict[str, np.ndarray]) -> None:
        if self.method == "finetune" or self.budget <= 0:
            self.data = None
            return
        if self.method == "reservoir":
            self._reservoir_update(current)
            return
        candidates = concatenate(self.data, current)
        embedding = standardized_embedding(candidates["inputs"])
        if self.method == "risk_only":
            indices = np.argsort(-candidates["risk_score"], kind="stable")[: self.budget]
        elif self.method == "kcenter":
            indices = kcenter_indices(embedding, self.budget)
        elif self.method == "facility":
            indices = lazy_facility_indices(embedding, np.ones(len(embedding)), self.budget, self.temperature)
        elif self.method == "cgsm":
            weights = 0.1 + np.power(candidates["risk_score"], self.risk_power)
            indices = lazy_facility_indices(embedding, weights, self.budget, self.temperature)
        elif self.method == "gss_adapted":
            indices = kcenter_indices(candidates["selection_embedding"], self.budget)
        elif self.method == "cgsm_gradient":
            magnitude = candidates["selection_magnitude"]
            ranks = np.empty(len(magnitude), np.float32)
            ranks[np.argsort(magnitude, kind="stable")] = np.linspace(0.0, 1.0, len(magnitude), dtype=np.float32)
            influence = 0.1 + 0.9 * ranks
            priority = np.power(np.maximum(candidates["risk_score"], 1e-8), self.risk_power) * influence
            weights = 1.0 + self.risk_strength * priority
            indices = risk_weighted_kcenter_indices(
                candidates["selection_embedding"], weights, self.budget
            )
        elif self.method == "risk_kcenter":
            weights = 0.05 + np.power(candidates["risk_score"], self.risk_power)
            indices = risk_weighted_kcenter_indices(embedding, weights, self.budget)
        else:
            raise ValueError(f"Unknown memory method: {self.method}")
        self.data = subset(candidates, np.asarray(indices, np.int64))

    def _reservoir_update(self, current: dict[str, np.ndarray]) -> None:
        if self.data is None:
            take = min(self.budget, len(current["inputs"]))
            self.data = subset(current, np.arange(take))
            self.seen = take
            start = take
        else:
            start = 0
        for index in range(start, len(current["inputs"])):
            self.seen += 1
            if len(self.data["inputs"]) < self.budget:
                self.data = concatenate(self.data, subset(current, np.asarray([index])))
                continue
            replacement = int(self.rng.integers(0, self.seen))
            if replacement < self.budget:
                for key in ARRAY_KEYS:
                    self.data[key][replacement] = current[key][index]


class H2CAdaptedMemory:
    """Clean-room shared-backbone adaptation of H2C's two-buffer principle.

    One half retains gradient-diverse cases and the other half uses true
    reservoir sampling.  This is intentionally labeled adapted because the
    original method uses its own UQnet predictor and online scoring details.
    """

    def __init__(self, budget: int, seed: int):
        self.separation_budget = budget // 2
        self.separation: dict[str, np.ndarray] | None = None
        self.completion = ReplayMemory(budget - self.separation_budget, "reservoir", seed + 1)

    @property
    def data(self) -> dict[str, np.ndarray] | None:
        if self.separation is None:
            return self.completion.data
        return concatenate(self.separation, self.completion.data) if self.completion.data is not None else self.separation

    def update(self, current: dict[str, np.ndarray]) -> None:
        candidates = concatenate(self.separation, current)
        indices = kcenter_indices(candidates["selection_embedding"], self.separation_budget)
        self.separation = subset(candidates, indices)
        self.completion.update(current)


class CGSMDualMemory:
    """Risk anchors followed by gradient-diverse conditional completion."""

    def __init__(self, budget: int, seed: int, risk_power: float = 4.0, general_fraction: float = 0.5):
        self.general_budget = int(round(budget * general_fraction))
        self.general_budget = min(max(self.general_budget, 0), budget)
        self.safety_budget = budget - self.general_budget
        self.general: dict[str, np.ndarray] | None = None
        self.safety: dict[str, np.ndarray] | None = None
        self.seed = seed
        self.risk_power = risk_power

    @property
    def data(self) -> dict[str, np.ndarray] | None:
        if self.general is None:
            return self.safety
        return concatenate(self.general, self.safety) if self.safety is not None else self.general

    def update(self, current: dict[str, np.ndarray]) -> None:
        candidates = concatenate(self.data, current)
        if self.safety_budget:
            observable = standardized_embedding(candidates["observable_embedding"])
            weights = np.power(np.maximum(candidates["risk_score"], 1e-8), self.risk_power)
            safety_indices = risk_weighted_kcenter_indices(
                observable, weights, min(self.safety_budget, len(candidates["inputs"]))
            )
            self.safety = subset(candidates, safety_indices)
            self.safety["replay_priority"][:] = 1.0
        else:
            safety_indices = np.empty(0, np.int64)
            self.safety = None

        if self.general_budget:
            general_indices = conditioned_kcenter_indices(
                candidates["selection_embedding"], self.general_budget, safety_indices
            )
            self.general = subset(candidates, general_indices)
            self.general["replay_priority"][:] = 0.0
        else:
            self.general = None
