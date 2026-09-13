"""Typed configuration for the Elo engine."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EloConfig(BaseModel):
    base_rating: float = 1500.0
    k_factor: float = 22.0
    home_field_advantage: float = 65.0
    # Neutral-site games get zero home_field_advantage applied.
    mov_multiplier_base: float = 2.2
    mov_multiplier_scale: float = 0.001
    # Fraction of the gap to the tier prior removed each offseason.
    # 0.0 = no regression, 1.0 = fully reset to prior every season.
    offseason_regression_weight: float = 0.35
    tier_priors: dict[str, float] = Field(default_factory=dict)
    conference_tiers: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EloConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            base_rating=raw.get("base_rating", 1500.0),
            tier_priors=raw.get("tier_priors", {}),
            conference_tiers=raw.get("conferences", {}),
        )

    def tier_for_conference(self, conference: str | None, is_fbs: bool = True) -> str:
        if not is_fbs:
            return "fcs"
        if conference is None:
            return "group_of_five"
        for tier, conferences in self.conference_tiers.items():
            if conference in conferences:
                return tier
        return "group_of_five"

    def prior_for_conference(self, conference: str | None, is_fbs: bool = True) -> float:
        tier = self.tier_for_conference(conference, is_fbs)
        return self.tier_priors.get(tier, self.base_rating)
