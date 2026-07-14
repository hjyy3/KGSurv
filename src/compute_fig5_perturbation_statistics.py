"""Compute fixed-eight-cohort statistics for Fig. 5 perturbation risks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from statsmodels.duration.hazard_regression import PHReg

VARIANTS = (
    "full", "all_kg_off", "bioprocess_protein", "cellcomp_protein",
    "molfunc_protein", "pathway_protein", "x_ppi",
)
FIXED_COHORTS = (
    "Gandara", "Hugo", "Liu", "Miao", "PUSH", "Pleasance", "Ravi", "Riaz",
)
REQUIRED_COLUMNS = ("variant", "split", "cohort", "sample_id", "time", "event", "risk")
OUTPUT_NAMES = (
    "fig5_fixed8_state_cutoffs.csv", "fig5_fixed8_state_cox.csv",
    "fig5_fixed8_per_cohort_logrank.csv", "fig5_fixed8_sensitivity.csv",
    "fig5_fixed8_cutoff_audit.json",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute fixed-eight-cohort Fig. 5 Cox, log-rank, cutoff, and "
            "sensitivity statistics from a patient-level perturbation CSV."
        )
    )
    parser.add_argument("input_csv", type=Path, help="patient-level long-table CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="output directory (default: input CSV directory)",
    )
    parser.add_argument(
        "--expected-full-cutoff", type=float, default=None,
        help="require exact equality with the recomputed full-state cutoff",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a file's lowercase hexadecimal SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(values: list[str]) -> str:
    """Return a deterministic SHA-256 digest for a sorted string list."""
    payload = json.dumps(sorted(values), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_set(label: str, actual: set[str], expected: set[str]) -> None:
    """Require exact set equality and raise a detailed ``ValueError``."""
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load and strictly validate the perturbation long table."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    strings = {name: "string" for name in REQUIRED_COLUMNS[:4]}
    data = pd.read_csv(path, dtype=strings)
    missing = [name for name in REQUIRED_COLUMNS if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty:
        raise ValueError("Input CSV contains no rows")
    for name in REQUIRED_COLUMNS[:4]:
        if data[name].isna().any():
            raise ValueError(f"Column {name!r} contains missing values")
        data[name] = data[name].str.strip()
        if data[name].eq("").any():
            raise ValueError(f"Column {name!r} contains empty values")
    for name in ("time", "event", "risk"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
        bad = ~np.isfinite(data[name].to_numpy(dtype=float))
        if bad.any():
            raise ValueError(f"Column {name!r} has non-finite values at rows {data.index[bad].tolist()[:5]}")
    if (data["time"] < 0).any():
        raise ValueError(f"Column 'time' must be >= 0; invalid rows={data.index[data['time'] < 0].tolist()[:5]}")
    if not data["event"].isin([0, 1]).all():
        invalid = sorted(data.loc[~data["event"].isin([0, 1]), "event"].unique())
        raise ValueError(f"Column 'event' must be binary 0/1; invalid={invalid[:5]}")
    data["event"] = data["event"].astype(int)
    require_set("variant", set(data["variant"]), set(VARIANTS))
    require_set("split", set(data["split"]), {"train", "external"})
    duplicate = data.duplicated(["variant", "split", "cohort", "sample_id"])
    if duplicate.any():
        raise ValueError(f"Duplicate patient-state rows at rows {data.index[duplicate].tolist()[:5]}")

    train = data[data["split"].eq("train")]
    external = data[data["split"].eq("external")]
    bad_train = sorted(train.loc[train["cohort"].isin(FIXED_COHORTS), "cohort"].unique())
    if bad_train:
        raise ValueError(f"Training rows use fixed external cohort labels: {bad_train}")
    require_set("external cohort", set(external["cohort"]), set(FIXED_COHORTS))
    overlap = set(train["sample_id"]) & set(external["sample_id"])
    if overlap:
        raise ValueError(f"Training/external sample_id overlap; first values={sorted(overlap)[:5]}")
    validate_state_alignment(data, data[data["variant"].eq("full")])
    return data


def validate_state_alignment(data: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Require all states to contain identical patients and outcomes.

    Args:
        data: Complete validated long table.
        reference: Full-state rows used as the reference.

    Raises:
        ValueError: If membership, assignments, or outcomes differ.
    """
    key = ["split", "cohort", "sample_id"]
    columns = key + ["time", "event"]
    expected = reference[columns].sort_values(key).reset_index(drop=True)
    train_cohorts = set(reference.loc[reference["split"].eq("train"), "cohort"])
    for variant in VARIANTS:
        current = data[data["variant"].eq(variant)]
        require_set(
            f"training cohort for {variant}",
            set(current.loc[current["split"].eq("train"), "cohort"]),
            train_cohorts,
        )
        for cohort in FIXED_COHORTS:
            present = current["split"].eq("external") & current["cohort"].eq(cohort)
            if not present.any():
                raise ValueError(f"State {variant!r} has no rows for cohort {cohort!r}")
        observed = current[columns].sort_values(key).reset_index(drop=True)
        if not observed.equals(expected):
            raise ValueError(f"Patient IDs, assignments, time, or event differ for {variant!r}")


def compute_cutoffs(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute state-specific float32 training means and provenance.

    Args:
        data: Validated patient-level long table.

    Returns:
        Cutoff table and per-state audit dictionary.
    """
    rows, audit = [], {}
    for variant in VARIANTS:
        train = data[data["variant"].eq(variant) & data["split"].eq("train")]
        risks = train["risk"].to_numpy(dtype=np.float64)
        risk_sum = float(np.sum(risks))
        cutoff = float(np.mean(risks))
        repeated = float(np.sum(risks) / len(risks))
        sample_hash = stable_hash(train["sample_id"].astype(str).tolist())
        rows.append({
            "variant": variant, "cutoff": cutoff, "n_train": int(len(risks)),
            "train_risk_sum": risk_sum, "train_sample_ids_sha256": sample_hash,
            "cutoff_rule": "mean of same-state train risk",
        })
        audit[variant] = {
            "training_sample_ids_sha256": sample_hash, "n": int(len(risks)),
            "sum": risk_sum, "mean": cutoff,
            "mean_recomputed_from_sum_and_n": repeated,
            "repeated_calculation_exact": cutoff == repeated,
        }
    return pd.DataFrame(rows), audit


def fit_cox(data: pd.DataFrame, covariate: str, strata: str | None = None) -> dict[str, Any]:
    """Fit a lifelines Cox model and preserve non-estimable results.

    Args:
        data: Rows containing time, event, and the covariate.
        covariate: Cox covariate column.
        strata: Optional cohort strata column.

    Returns:
        Estimates, counts, estimability flag, and failure reason.
    """
    result = {
        "n": int(len(data)), "events": int(data["event"].sum()),
        "hr": np.nan, "hr_lo": np.nan, "hr_hi": np.nan, "p": np.nan,
        "estimable": False, "reason": "",
    }
    if data.empty or result["events"] == 0 or data[covariate].nunique() < 2:
        result["reason"] = (
            "no analysis rows" if data.empty else "no observed events"
            if result["events"] == 0 else f"{covariate} has no variation"
        )
        return result
    try:
        model = PHReg(
            data["time"].to_numpy(), data[[covariate]].to_numpy(),
            status=data["event"].to_numpy(),
            strata=data[strata].to_numpy() if strata else None, ties="breslow",
        )
        fitted = model.fit(disp=0)
        interval = fitted.conf_int()[0]
        result.update({
            "hr": float(np.exp(fitted.params[0])),
            "hr_lo": float(np.exp(interval[0])), "hr_hi": float(np.exp(interval[1])),
            "p": float(fitted.pvalues[0]), "estimable": True,
        })
    except Exception as exc:  # lifelines exposes several convergence error types.
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def compute_main_cox(data: pd.DataFrame, cutoffs: dict[str, float]) -> pd.DataFrame:
    """Fit pooled cutoff-group unstratified Cox models.

    Args:
        data: Validated patient-level long table.
        cutoffs: State-specific training-fold means.
    Returns:
        Seven-row high-versus-low risk Cox table.
    """
    rows = []
    for variant in VARIANTS:
        external = data[
            data["variant"].eq(variant) & data["split"].eq("external")
        ].copy()
        cutoff = cutoffs[variant]
        external["high_risk"] = (external["risk"] >= cutoff).astype(int)
        pooled_logrank = logrank_row(variant, "pooled_fixed8", external, cutoff)
        rows.append({
            "variant": variant, "cutoff": cutoff, "contrast": "high_vs_low",
            "analysis": "fixed8_pooled_state_cutoff_group_unstratified_cox",
            "cutoff_source": "same-state train mean", "n_cohorts": len(FIXED_COHORTS),
            "n_high": int(external["high_risk"].sum()),
            "n_low": int((1 - external["high_risk"]).sum()),
            "events_high": int(external.loc[external["high_risk"].eq(1), "event"].sum()),
            "events_low": int(external.loc[external["high_risk"].eq(0), "event"].sum()),
            "pooled_logrank_p": pooled_logrank["logrank_p"],
            **fit_cox(external, "high_risk"),
        })
    return pd.DataFrame(rows)


def logrank_row(variant: str, cohort: str, data: pd.DataFrame, cutoff: float) -> dict[str, Any]:
    """Compute one cohort's cutoff-based log-rank result.

    Args:
        variant: Perturbation state.
        cohort: Fixed external cohort.
        data: Cohort patient rows.
        cutoff: Threshold where ``risk >= cutoff`` is high risk.

    Returns:
        Counts, P value, estimability flag, and failure reason.
    """
    high = data["risk"] >= cutoff
    row = {
        "variant": variant, "cohort": cohort, "cutoff": cutoff,
        "n": int(len(data)), "events": int(data["event"].sum()),
        "n_high": int(high.sum()), "n_low": int((~high).sum()),
        "events_high": int(data.loc[high, "event"].sum()),
        "events_low": int(data.loc[~high, "event"].sum()),
        "logrank_p": np.nan, "significant": False,
        "estimable": False, "reason": "",
    }
    if row["n_high"] == 0 or row["n_low"] == 0 or row["events"] == 0:
        row["reason"] = (
            "one risk group is empty" if row["n_high"] == 0 or row["n_low"] == 0
            else "no observed events"
        )
        return row
    try:
        test = logrank_test(
            data.loc[high, "time"], data.loc[~high, "time"],
            data.loc[high, "event"], data.loc[~high, "event"],
        )
        p_value = float(test.p_value)
        row.update({
            "logrank_p": p_value, "significant": bool(p_value < 0.05),
            "estimable": True,
        })
    except Exception as exc:
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def compute_logrank_table(data: pd.DataFrame, cutoffs: dict[str, float]) -> pd.DataFrame:
    """Compute state-cutoff log-rank results for all 56 strata.

    Args:
        data: Validated patient-level long table.
        cutoffs: State-specific training means.

    Returns:
        Per-state, per-cohort table with ``n_sig/8`` summaries.
    """
    rows = []
    for variant in VARIANTS:
        for cohort in FIXED_COHORTS:
            current = data[
                data["variant"].eq(variant) & data["split"].eq("external")
                & data["cohort"].eq(cohort)
            ]
            rows.append(logrank_row(variant, cohort, current, cutoffs[variant]))
    output = pd.DataFrame(rows)
    n_sig = output.groupby("variant", sort=False)["significant"].sum().astype(int)
    output["n_sig"] = output["variant"].map(n_sig)
    output["n_sig_denominator"] = len(FIXED_COHORTS)
    output["n_sig_label"] = output["n_sig"].astype(str) + "/8"
    output["cutoff_source"] = "same-state train mean"
    return output


def compute_sensitivity(
    data: pd.DataFrame, cutoffs: dict[str, float], state_logrank: pd.DataFrame,
) -> pd.DataFrame:
    """Compute fixed-full-cutoff and continuous-risk Cox sensitivities.

    Args:
        data: Validated rows.
        cutoffs: State cutoffs.
        state_logrank: Main log-rank results.
    Returns:
        Tidy sensitivity table.
    """
    state_n_sig = state_logrank.groupby("variant", sort=False)["significant"].sum().astype(int)
    full_rows = []
    for variant in VARIANTS:
        for cohort in FIXED_COHORTS:
            current = data[
                data["variant"].eq(variant) & data["split"].eq("external")
                & data["cohort"].eq(cohort)
            ]
            full_rows.append(logrank_row(variant, cohort, current, cutoffs["full"]))
    full_frame = pd.DataFrame(full_rows)
    full_n_sig = full_frame.groupby("variant", sort=False)["significant"].sum().astype(int)
    rows = []
    for row in full_frame.to_dict("records"):
        variant = row["variant"]
        rows.append({
            "analysis": "full_cutoff_per_cohort_logrank", **row,
            "state_specific_cutoff": cutoffs[variant],
            "state_specific_n_sig": int(state_n_sig[variant]),
            "sensitivity_n_sig": int(full_n_sig[variant]),
            "n_sig_denominator": len(FIXED_COHORTS),
            "n_sig_label": f"{int(full_n_sig[variant])}/8",
            "hr": np.nan, "hr_lo": np.nan, "hr_hi": np.nan, "p": np.nan,
        })
    for variant in VARIANTS:
        external = data[
            data["variant"].eq(variant) & data["split"].eq("external")
        ].copy()
        continuous = fit_cox(external, "risk")
        rows.append({
            "analysis": "continuous_risk_unstratified_cox", "variant": variant,
            "cohort": "", "cutoff": np.nan, "n_high": np.nan, "n_low": np.nan,
            "events_high": np.nan, "events_low": np.nan, "logrank_p": np.nan,
            "significant": False, "state_specific_cutoff": cutoffs[variant],
            "state_specific_n_sig": int(state_n_sig[variant]), "sensitivity_n_sig": np.nan,
            "n_sig_denominator": len(FIXED_COHORTS), "n_sig_label": "", **continuous,
        })
        external["risk_z"] = np.nan
        zero_sd = []
        for cohort in FIXED_COHORTS:
            mask = external["cohort"].eq(cohort)
            risks = external.loc[mask, "risk"]
            sd = float(risks.std(ddof=1))
            if not math.isfinite(sd) or sd == 0:
                zero_sd.append(cohort)
            else:
                external.loc[mask, "risk_z"] = (risks - risks.mean()) / sd
        result = (
            {
                "n": int(len(external)), "events": int(external["event"].sum()),
                "hr": np.nan, "hr_lo": np.nan, "hr_hi": np.nan, "p": np.nan,
                "estimable": False,
                "reason": f"zero/non-finite within-cohort SD: {zero_sd}",
            }
            if zero_sd else fit_cox(external, "risk_z", strata="cohort")
        )
        rows.append({
            "analysis": "within_cohort_zscore_cohort_stratified_cox",
            "variant": variant, "cohort": "", "cutoff": np.nan,
            "n_high": np.nan, "n_low": np.nan,
            "events_high": np.nan, "events_low": np.nan,
            "logrank_p": np.nan, "significant": False,
            "state_specific_cutoff": cutoffs[variant],
            "state_specific_n_sig": int(state_n_sig[variant]),
            "sensitivity_n_sig": np.nan,
            "n_sig_denominator": len(FIXED_COHORTS), "n_sig_label": "", **result,
        })
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path, input_path: Path, data: pd.DataFrame,
    cutoffs: pd.DataFrame, cutoff_audit: dict[str, Any], main_cox: pd.DataFrame,
    logrank: pd.DataFrame, sensitivity: pd.DataFrame,
    expected_full_cutoff: float | None,
) -> None:
    """Write validated result tables and cutoff provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        OUTPUT_NAMES[0]: cutoffs, OUTPUT_NAMES[1]: main_cox,
        OUTPUT_NAMES[2]: logrank, OUTPUT_NAMES[3]: sensitivity,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, float_format="%.17g")
    train_ids = data.loc[data["split"].eq("train"), "sample_id"].drop_duplicates().astype(str).tolist()
    full_cutoff = float(cutoffs.loc[cutoffs["variant"].eq("full"), "cutoff"].iloc[0])
    replayed_full = float(cutoff_audit["full"].get("replayed_mean", full_cutoff))
    audit = {
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "fixed_variants": list(VARIANTS),
        "fixed_external_cohorts": list(FIXED_COHORTS),
        "training_samples": {
            "sample_ids_sha256": stable_hash(train_ids),
            "hash_canonicalization": "sorted unique sample_id JSON, compact ASCII",
            "n": len(train_ids), "external_sample_id_intersection_n": 0,
        },
        "cutoffs": cutoff_audit,
        "expected_full_cutoff": expected_full_cutoff,
        "observed_full_cutoff": replayed_full,
        "authoritative_full_cutoff_used": full_cutoff,
        "full_cutoff_exact_match": (
            None if expected_full_cutoff is None else replayed_full == expected_full_cutoff
        ),
        "full_cutoff_within_tolerance": (
            None if expected_full_cutoff is None else bool(np.isclose(
                replayed_full, expected_full_cutoff, rtol=0.0, atol=1e-6))
        ),
        "environment": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv], "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "lifelines": __import__("lifelines").__version__,
        },
        "outputs": list(OUTPUT_NAMES),
    }
    (output_dir / OUTPUT_NAMES[4]).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    """Run validation, statistics, cutoff checking, and output writing."""
    args = parse_args()
    try:
        input_path = args.input_csv.expanduser().resolve()
        data = load_and_validate(input_path)
        cutoff_table, cutoff_audit = compute_cutoffs(data)
        cutoff_map = dict(zip(cutoff_table["variant"], cutoff_table["cutoff"]))
        observed_full = float(cutoff_map["full"])
        if args.expected_full_cutoff is not None and not np.isclose(
            observed_full, args.expected_full_cutoff, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"Full cutoff mismatch: expected={args.expected_full_cutoff!r}, "
                f"observed={observed_full!r}"
            )
        if args.expected_full_cutoff is not None:
            cutoff_audit["full"]["replayed_mean"] = observed_full
            cutoff_audit["full"]["authoritative_frozen_mean"] = args.expected_full_cutoff
            cutoff_audit["full"]["replay_delta"] = observed_full - args.expected_full_cutoff
            full_row = cutoff_table["variant"].eq("full")
            cutoff_table.loc[full_row, "cutoff"] = args.expected_full_cutoff
            cutoff_map["full"] = args.expected_full_cutoff
        main_cox = compute_main_cox(data, cutoff_map)
        logrank = compute_logrank_table(data, cutoff_map)
        sensitivity = compute_sensitivity(data, cutoff_map, logrank)
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None else input_path.parent
        )
        write_outputs(
            output_dir, input_path, data, cutoff_table, cutoff_audit,
            main_cox, logrank, sensitivity, args.expected_full_cutoff,
        )
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote fixed-eight-cohort statistics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
