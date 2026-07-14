"""ExperimentSpec dataclass + tier-level spec generators.

Spec uniquely identifies a (model, kg, node-combo, seed, fold) run.
run_id is filesystem-safe and idempotent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


BEST_COMBOS: dict[str, tuple[str, ...]] = {
    "drkg":        ("ppi", "disease", "drug"),
    "openbiolink": ("ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"),
    "monarch":     ("ppi", "disease", "phenotype", "anatomy"),
    "hetionet":    ("ppi", "disease", "drug", "anatomy", "regulatory"),
    "primekg":     ("ppi",),
    "ibkh":        ("ppi", "disease", "drug"),
    "ogb_biokg":   ("ppi", "drug"),
}

MODELS: tuple[str, ...] = ("sparse_path", "path_attn", "bipartite_attn")
KGS: tuple[str, ...] = tuple(BEST_COMBOS.keys())


def combo_tag(node_types: tuple[str, ...]) -> str:
    return "+".join(node_types)


@dataclass(frozen=True)
class ExperimentSpec:
    model: str
    kg: str
    node_types: tuple[str, ...]
    combo_tag: str
    seed: int
    fold: int
    fmb_variant: str | None = None   # None | "mg2" | "mg5"

    @property
    def effective_kg(self) -> str:
        if self.fmb_variant:
            return f"{self.kg}_{self.fmb_variant}"
        return self.kg

    @property
    def run_id(self) -> str:
        suffix = f"_{self.fmb_variant}" if self.fmb_variant else ""
        return (f"{self.model}_{self.kg}{suffix}_{self.combo_tag}"
                f"_seed{self.seed}_fold{self.fold}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["node_types"] = list(self.node_types)
        return d


def tier_0_specs() -> list[ExperimentSpec]:
    """3 model × 7 KG × 1 fold (=0) × 1 seed (=42) = 21 specs."""
    out: list[ExperimentSpec] = []
    for model in MODELS:
        for kg in KGS:
            nt = BEST_COMBOS[kg]
            out.append(ExperimentSpec(
                model=model, kg=kg, node_types=nt,
                combo_tag=combo_tag(nt), seed=42, fold=0,
            ))
    return out


def tier_1_specs(
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> list[ExperimentSpec]:
    """3 model × 7 KG × 5 fold × 5 seed = 525 specs."""
    out: list[ExperimentSpec] = []
    for model in MODELS:
        for kg in KGS:
            nt = BEST_COMBOS[kg]
            for s in seeds:
                for f in folds:
                    out.append(ExperimentSpec(
                        model=model, kg=kg, node_types=nt,
                        combo_tag=combo_tag(nt), seed=s, fold=f,
                    ))
    return out


# KGs with mg2/mg5 FMB variants available under output/kg_features/
TIER_2_KGS: tuple[str, ...] = ("hetionet", "ogb_biokg", "primekg")
TIER_2_VARIANTS: tuple[str, ...] = ("mg2", "mg5")


def tier_2_specs(
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> list[ExperimentSpec]:
    """path_attn × 3 KG × 2 variants × 5 fold × 5 seed = 150 specs."""
    out: list[ExperimentSpec] = []
    for kg in TIER_2_KGS:
        nt = BEST_COMBOS[kg]
        for variant in TIER_2_VARIANTS:
            for s in seeds:
                for f in folds:
                    out.append(ExperimentSpec(
                        model="path_attn", kg=kg, node_types=nt,
                        combo_tag=combo_tag(nt), seed=s, fold=f,
                        fmb_variant=variant,
                    ))
    return out


# Tier 3 — confidence-interval booster on the best (model, kg, variant)
TIER_3_KG: str = "ogb_biokg"
TIER_3_VARIANT: str = "mg5"
TIER_3_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)


def tier_3_specs(
    seeds: tuple[int, ...] = TIER_3_SEEDS,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> list[ExperimentSpec]:
    """path_attn × ogb_biokg × mg5 × 10 seeds × 5 folds = 50 specs.

    First 25 (seeds 42-46) re-use cached runs from Tier 2; 25 new (seeds 47-51).
    """
    out: list[ExperimentSpec] = []
    nt = BEST_COMBOS[TIER_3_KG]
    for s in seeds:
        for f in folds:
            out.append(ExperimentSpec(
                model="path_attn", kg=TIER_3_KG, node_types=nt,
                combo_tag=combo_tag(nt), seed=s, fold=f,
                fmb_variant=TIER_3_VARIANT,
            ))
    return out


# Tier 4 — push the 7/11 ceiling. Lottery tickets on every config that has
# produced a 7/11 winner under any of the 10 cutoff rules.
TIER_4_CONFIGS: tuple[tuple[str, str | None], ...] = (
    ("drkg",      None),
    ("hetionet",  None),
    ("hetionet",  "mg2"),
    ("hetionet",  "mg5"),
    ("ibkh",      None),
    ("ogb_biokg", "mg5"),
    ("primekg",   None),
    ("primekg",   "mg2"),
    ("primekg",   "mg5"),
)
TIER_4_SEEDS: tuple[int, ...] = (47, 48, 49, 50, 51, 52, 53, 54, 55, 56)


def tier_4_specs(
    seeds: tuple[int, ...] = TIER_4_SEEDS,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> list[ExperimentSpec]:
    """path_attn × 9 (kg, variant) × 10 new seeds × 5 fold = 450 specs.

    Pure additive on top of Tier 1-3. ogb_biokg_mg5 seeds 47-51 reuse cache.
    """
    out: list[ExperimentSpec] = []
    for kg, variant in TIER_4_CONFIGS:
        nt = BEST_COMBOS[kg]
        for s in seeds:
            for f in folds:
                out.append(ExperimentSpec(
                    model="path_attn", kg=kg, node_types=nt,
                    combo_tag=combo_tag(nt), seed=s, fold=f,
                    fmb_variant=variant,
                ))
    return out


# Tier 5 — push the 8/11 ceiling. Top 10 configs by Tier-4 fixed_mean,
# adding fresh seeds 57-66 to multiply lottery tickets.
TIER_5_CONFIGS: tuple[tuple[str, str | None], ...] = (
    ("ibkh",      None),       # fixed_mean=5.32, fixed_max=7
    ("primekg",   "mg2"),      # 5.17, 7
    ("ogb_biokg", None),       # 4.96, 6 — only 25 runs so far
    ("primekg",   "mg5"),      # 4.96, 7
    ("ogb_biokg", "mg5"),      # 4.76, 7
    ("hetionet",  None),       # 4.71, 7
    ("primekg",   None),       # 4.68, 7
    ("drkg",      None),       # 4.68, 7
    ("hetionet",  "mg2"),      # 4.64, 8 ★ only 8/11 carrier
    ("hetionet",  "mg5"),      # 4.51, 7
)
TIER_5_SEEDS: tuple[int, ...] = (57, 58, 59, 60, 61, 62, 63, 64, 65, 66)


def tier_5_specs(
    seeds: tuple[int, ...] = TIER_5_SEEDS,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> list[ExperimentSpec]:
    """path_attn × 10 top configs × 10 new seeds × 5 fold = 500 specs."""
    out: list[ExperimentSpec] = []
    for kg, variant in TIER_5_CONFIGS:
        nt = BEST_COMBOS[kg]
        for s in seeds:
            for f in folds:
                out.append(ExperimentSpec(
                    model="path_attn", kg=kg, node_types=nt,
                    combo_tag=combo_tag(nt), seed=s, fold=f,
                    fmb_variant=variant,
                ))
    return out


# Tier 6 — push the 9/11 ceiling. Same 10 configs as Tier 5, fresh seeds 67-76.
TIER_6_CONFIGS = TIER_5_CONFIGS
TIER_6_SEEDS: tuple[int, ...] = (67, 68, 69, 70, 71, 72, 73, 74, 75, 76)


def tier_6_specs(
    seeds: tuple[int, ...] = TIER_6_SEEDS,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> list[ExperimentSpec]:
    """path_attn × 10 top configs × 10 new seeds (67-76) × 5 fold = 500 specs."""
    out: list[ExperimentSpec] = []
    for kg, variant in TIER_6_CONFIGS:
        nt = BEST_COMBOS[kg]
        for s in seeds:
            for f in folds:
                out.append(ExperimentSpec(
                    model="path_attn", kg=kg, node_types=nt,
                    combo_tag=combo_tag(nt), seed=s, fold=f,
                    fmb_variant=variant,
                ))
    return out
