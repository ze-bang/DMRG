"""Pydantic-validated configuration schema for run-from-YAML workflows.

The CLI loads a YAML file (or any mapping) into an
:class:`ExperimentConfig`, instantiates the requested model and DMRG runtime,
and dispatches the calculation.  The schema is intentionally narrow: each
model gets its own discriminated subtype so typos and bad parameter ranges
are caught at parse time rather than mid-sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dmrg_pp.algorithms.dmrg import DMRGConfig
from dmrg_pp.models.base import LatticeModel
from dmrg_pp.models.heisenberg import HeisenbergXXZ
from dmrg_pp.models.hubbard import HubbardChain
from dmrg_pp.models.ising import TransverseFieldIsing

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "RuntimeConfig",
    "load_config",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class HeisenbergConfig(_StrictModel):
    kind: Literal["heisenberg"] = "heisenberg"
    L: int = Field(gt=1)
    Jxy: float = 1.0
    Jz: float = 1.0
    h: float = 0.0

    def build(self) -> LatticeModel:
        return HeisenbergXXZ(L=self.L, Jxy=self.Jxy, Jz=self.Jz, h=self.h)


class IsingConfig(_StrictModel):
    kind: Literal["ising"] = "ising"
    L: int = Field(gt=1)
    J: float = 1.0
    g: float = 1.0

    def build(self) -> LatticeModel:
        return TransverseFieldIsing(L=self.L, J=self.J, g=self.g)


class HubbardConfig(_StrictModel):
    kind: Literal["hubbard"] = "hubbard"
    L: int = Field(gt=1)
    t: float = 1.0
    U: float = 4.0
    mu: float = 0.0

    def build(self) -> LatticeModel:
        return HubbardChain(L=self.L, t=self.t, U=self.U, mu=self.mu)


ModelConfig = Annotated[
    HeisenbergConfig | IsingConfig | HubbardConfig,
    Field(discriminator="kind"),
]


class RuntimeConfig(_StrictModel):
    n_sweeps: int = Field(default=10, gt=0)
    max_bond: int | list[int] = 64
    cutoff: float | list[float] = 1e-10
    e_tol: float = 1e-9
    var_tol: float = 0.0
    lanczos_tol: float = 1e-12
    lanczos_max_iter: int = Field(default=200, gt=0)
    seed: int | None = None
    initial_bond: int = Field(default=4, gt=0)
    log_every: int = Field(default=1, gt=0)

    def to_dmrg_config(self) -> DMRGConfig:
        return DMRGConfig(**self.model_dump())


class ExperimentConfig(_StrictModel):
    name: str = "dmrg_run"
    model: ModelConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: Path | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML/JSON file into a validated :class:`ExperimentConfig`."""
    p = Path(path)
    raw: dict[str, Any]
    text = p.read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"top-level config must be a mapping, got {type(raw).__name__}")
    return ExperimentConfig.model_validate(raw)
