from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mimic_icu_eicu_transport_recalibration as icu  # noqa: E402
import nhanes_mimic_oneyear_mortality_transport as nm  # noqa: E402
import sensitivity_unweighted_logistic as sens  # noqa: E402


OUT_DIR = ROOT / "outputs" / "current_submission" / "tables"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.fillna("NA").astype(str)
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(row[col] for col in display.columns) + " |")
    return "\n".join(lines)


def fmt(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def build_source_internal_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    nhanes = nm.load_nhanes()
    nh_train, nh_test = nm.temporal_nhanes_split(nhanes)
    nh_features = nm.COMMON_BASE
    nh_model = sens.make_unweighted_nhanes_model(nh_features)
    nh_train = nh_train.dropna(subset=["outcome"]).copy()
    nh_test = nh_test.dropna(subset=["outcome"]).copy()
    nh_model.fit(nh_train[nh_features], nh_train["outcome"].astype(int))
    nh_pred = nh_model.predict_proba(nh_test[nh_features])[:, 1]
    nh_metrics = sens.point_metrics(nh_test["outcome"].astype(int).to_numpy(), nh_pred)
    rows.append(
        {
            "Source": "NHANES",
            "Internal evaluation": "NHANES 2013-2014 temporal holdout",
            "Endpoint": "1-year mortality",
            "Feature set": "NHANES-compatible base",
            **nh_metrics,
        }
    )

    sources = {
        "MIMIC-IV ICU": icu.load_source("MIMIC-IV ICU"),
        "eICU": icu.load_source("eICU"),
    }
    splits: dict[str, dict[str, pd.DataFrame]] = {}
    for i, source in enumerate(["MIMIC-IV ICU", "eICU"]):
        development, holdout = icu.split_by_patient(sources[source], icu.RANDOM_SEED + i)
        splits[source] = {"development": development, "holdout": holdout}

    spec = icu.FEATURE_SETS["icu_native_primary"]
    numeric = spec["numeric"]
    binary = spec["binary"]
    icu_features = numeric + binary
    for source in ["MIMIC-IV ICU", "eICU"]:
        model = sens.make_unweighted_icu_model(numeric, binary)
        train = splits[source]["development"]
        test = splits[source]["holdout"]
        model.fit(train[icu_features], train["outcome"].astype(int))
        pred = model.predict_proba(test[icu_features])[:, 1]
        metrics = sens.point_metrics(test["outcome"].astype(int).to_numpy(), pred)
        rows.append(
            {
                "Source": source,
                "Internal evaluation": f"{source} patient-level holdout",
                "Endpoint": "Hospital mortality",
                "Feature set": "ICU-native primary",
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    numeric = build_source_internal_metrics()
    numeric_path = OUT_DIR / "Supplementary_Table_source_internal_calibration_numeric.csv"
    numeric.to_csv(numeric_path, index=False)

    rows = []
    for _, row in numeric.iterrows():
        rows.append(
            {
                "Source": row["Source"],
                "Internal evaluation": row["Internal evaluation"],
                "Endpoint": row["Endpoint"],
                "Feature set": row["Feature set"],
                "N": int(row["n"]),
                "Events": int(row["events"]),
                "Event rate": fmt(row["event_rate"]),
                "Mean predicted risk": fmt(row["mean_prediction"]),
                "AUC": fmt(row["roc_auc"]),
                "PR AUC": fmt(row["pr_auc"]),
                "Brier score": fmt(row["brier_score"]),
                "ECE": fmt(row["ece_10bin"]),
                "Equal-width ECE": fmt(row["ece_10bin_equal_width"]),
                "Calibration slope": fmt(row["calibration_slope"]),
                "Calibration intercept": fmt(row["calibration_intercept"]),
            }
        )
    formatted = pd.DataFrame(rows)
    formatted_path = OUT_DIR / "Supplementary_Table_source_internal_calibration.csv"
    formatted.to_csv(formatted_path, index=False, encoding="utf-8-sig")

    notes = [
        "This supplementary table reports unweighted logistic model performance on the source dataset's internal holdout or temporal holdout.",
        "The ICU source models were close to calibrated on their source holdouts, supporting interpretation of ICU-to-ICU slope distortion as a transport phenomenon.",
        "The NHANES temporal holdout contained few one-year mortality events, so its internal calibration slope is imprecise and the NHANES-to-MIMIC analysis remains a cross-setting stress test.",
    ]
    md = "# Supplementary_Table_source_internal_calibration\n\n"
    md += markdown_table(formatted)
    md += "\n\n## Notes\n\n"
    md += "\n".join(f"- {note}" for note in notes)
    md += "\n"
    (OUT_DIR / "Supplementary_Table_source_internal_calibration.md").write_text(md, encoding="utf-8")

    print(f"Wrote {formatted_path}")
    print(f"Wrote {numeric_path}")


if __name__ == "__main__":
    main()
