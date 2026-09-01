"""
AMDEC — Centralized Configuration & Hyperparameter Management.

Provides machine-readable configuration for all experimental evaluation stages.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, Optional


@dataclass
class WorkloadConfig:
    num_pages: int = 10000
    near_dup_ratio: float = 0.80
    exact_dup_ratio: float = 0.07
    unique_ratio: float = 0.13
    num_regions: int = 8
    zipf_param: float = 1.3
    num_ticks: int = 40
    writes_per_tick: int = 250
    sim_start_time: float = 100.0


@dataclass
class M1SamplingConfig:
    hot_write_rate_threshold: float = 0.06
    cold_write_rate_threshold: float = 0.005
    min_age_for_cold: float = 10.0
    cow_penalty_weight: float = 2.5
    quarantine_duration: float = 30.0


@dataclass
class M2ClusteringConfig:
    block_size: int = 128
    num_hashes: int = 32
    num_bands: int = 8
    similarity_threshold: float = 0.70


@dataclass
class M3DeltaConfig:
    max_delta_ratio: float = 0.30
    base_selection_strategy: str = "medoid"


@dataclass
class AMDECConfig:
    """Master experiment configuration dataclass."""
    experiment_name: str = "AMDEC_Primary_Evaluation"
    seed: int = 42
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    m1_sampling: M1SamplingConfig = field(default_factory=M1SamplingConfig)
    m2_clustering: M2ClusteringConfig = field(default_factory=M2ClusteringConfig)
    m3_delta: M3DeltaConfig = field(default_factory=M3DeltaConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AMDECConfig":
        return cls(
            experiment_name=data.get("experiment_name", "AMDEC_Primary_Evaluation"),
            seed=data.get("seed", 42),
            workload=WorkloadConfig(**data.get("workload", {})),
            m1_sampling=M1SamplingConfig(**data.get("m1_sampling", {})),
            m2_clustering=M2ClusteringConfig(**data.get("m2_clustering", {})),
            m3_delta=M3DeltaConfig(**data.get("m3_delta", {})),
        )

    @classmethod
    def load_json(cls, file_path: str) -> "AMDECConfig":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
