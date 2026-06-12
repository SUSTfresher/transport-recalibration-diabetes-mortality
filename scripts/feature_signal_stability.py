from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "nhanes_mimic_common_transport_table.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "feature_signal_stability"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures_feasibility"

FEATURES = [
    "age",
    "female",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "hypertension_history",
    "ckd_history",
    "cvd_history",
]

MAX_PERMUTATION_N = 5000
PERMUTATION_REPEATS = 20

FEATURE_LABELS = {
    "age": "Age",
    "female": "Female",
    "bmi": "BMI",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "hypertension_history": "Hypertension",
    "ckd_history": "CKD",
    "cvd_history": "CVD",
}


def make_logistic() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ]
    )
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def make_hgb() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                FEATURES,
            )
        ]
    )
    return Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=20260608,
                ),
            ),
        ]
    )


def load_data() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["known_outcome"].astype(str).str.lower().isin(["true", "1"])].copy()
    out = {}
    for source in ["NHANES", "MIMIC-IV"]:
        source_df = df[df["source"].eq(source)].dropna(subset=["outcome"]).copy()
        source_df["outcome"] = pd.to_numeric(source_df["outcome"], errors="coerce").astype(int)
        out[source] = source_df
    return out


def permutation_eval_sample(x: pd.DataFrame, y: pd.Series, source: str) -> tuple[pd.DataFrame, pd.Series]:
    if len(x) <= MAX_PERMUTATION_N:
        return x, y
    _, x_sample, _, y_sample = train_test_split(
        x,
        y,
        test_size=MAX_PERMUTATION_N,
        stratify=y,
        random_state=20260608 if source == "NHANES" else 20260609,
    )
    return x_sample, y_sample


def fit_models(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coef_rows = []
    perm_rows = []
    perf_rows = []
    for source, df in data.items():
        train, test = train_test_split(df, test_size=0.25, stratify=df["outcome"], random_state=20260608)
        x_train, y_train = train[FEATURES], train["outcome"]
        x_test, y_test = test[FEATURES], test["outcome"]

        log_model = make_logistic()
        log_model.fit(x_train, y_train)
        coefs = log_model.named_steps["model"].coef_[0]
        for feature, coef in zip(FEATURES, coefs):
            coef_rows.append(
                {
                    "source": source,
                    "model": "logistic_regression",
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "coefficient": float(coef),
                    "direction": "positive" if coef > 0 else "negative" if coef < 0 else "zero",
                }
            )

        for model_name, model in [("logistic_regression", log_model), ("hist_gradient_boosting", make_hgb())]:
            if model_name == "hist_gradient_boosting":
                model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            perf_rows.append({"source": source, "model": model_name, "accuracy": float(score), "test_n": int(len(test)), "test_events": int(y_test.sum())})
            x_perm, y_perm = permutation_eval_sample(x_test, y_test, source)
            perm = permutation_importance(
                model,
                x_perm,
                y_perm,
                n_repeats=PERMUTATION_REPEATS,
                random_state=20260608,
                scoring="roc_auc",
                n_jobs=1,
            )
            for feature, mean, std in zip(FEATURES, perm.importances_mean, perm.importances_std):
                perm_rows.append(
                    {
                        "source": source,
                        "model": model_name,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        "importance_mean": float(mean),
                        "importance_std": float(std),
                        "permutation_eval_n": int(len(x_perm)),
                        "permutation_eval_events": int(y_perm.sum()),
                    }
                )
    coef_df = pd.DataFrame(coef_rows)
    perm_df = pd.DataFrame(perm_rows)
    perf_df = pd.DataFrame(perf_rows)
    perm_df["importance_rank"] = perm_df.groupby(["source", "model"])["importance_mean"].rank(ascending=False, method="average")
    return coef_df, perm_df, perf_df


def stability_summary(coef: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    coef_wide = coef.pivot_table(index="feature", columns="source", values="coefficient", aggfunc="first")
    coef_wide["same_direction"] = np.sign(coef_wide["NHANES"]) == np.sign(coef_wide["MIMIC-IV"])
    for feature, row in coef_wide.iterrows():
        rows.append(
            {
                "analysis": "logistic_coefficient_direction",
                "model": "logistic_regression",
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "nhanes_value": row["NHANES"],
                "mimic_value": row["MIMIC-IV"],
                "same_direction": bool(row["same_direction"]),
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
            }
        )
    for model, group in perm.groupby("model"):
        wide = group.pivot_table(index="feature", columns="source", values="importance_rank", aggfunc="first")
        rho, p = spearmanr(wide["NHANES"], wide["MIMIC-IV"])
        for feature, row in wide.iterrows():
            rows.append(
                {
                    "analysis": "permutation_importance_rank",
                    "model": model,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "nhanes_value": row["NHANES"],
                    "mimic_value": row["MIMIC-IV"],
                    "same_direction": np.nan,
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                }
            )
    return pd.DataFrame(rows)


def plot_feature_stability(coef: pd.DataFrame, perm: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    coef_wide = coef.pivot_table(index="feature_label", columns="source", values="coefficient", aggfunc="first").reset_index()
    coef_wide = coef_wide.sort_values("MIMIC-IV").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.axvline(0, color="#555555", linewidth=0.8)
    y = np.arange(len(coef_wide))
    ax.scatter(coef_wide["NHANES"], y, label="NHANES", color="#1f78b4")
    ax.scatter(coef_wide["MIMIC-IV"], y, label="MIMIC-IV", color="#b15928")
    for i, row in coef_wide.iterrows():
        ax.plot([row["NHANES"], row["MIMIC-IV"]], [i, i], color="#999999", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(coef_wide["feature_label"])
    ax.set_xlabel("Standardized logistic coefficient")
    ax.set_title("Feature signal direction across data sources")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_11_feature_signal_coefficients.png", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "figure_11_feature_signal_coefficients.svg", bbox_inches="tight")
    plt.close(fig)

    perm_focus = perm[perm["model"].eq("hist_gradient_boosting")].copy()
    wide = perm_focus.pivot_table(index="feature_label", columns="source", values="importance_rank", aggfunc="first").reset_index()
    rho, p = spearmanr(wide["NHANES"], wide["MIMIC-IV"])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(wide["NHANES"], wide["MIMIC-IV"], color="#2f6f9f")
    for _, row in wide.iterrows():
        ax.text(row["NHANES"] + 0.05, row["MIMIC-IV"] + 0.05, row["feature_label"], fontsize=8)
    ax.plot([1, len(FEATURES)], [1, len(FEATURES)], color="#777777", linestyle="--", linewidth=0.9)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("NHANES importance rank")
    ax.set_ylabel("MIMIC-IV importance rank")
    ax.set_title(f"HGB permutation-importance rank stability\nSpearman rho={rho:.2f}, p={p:.3f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_12_feature_importance_rank_stability.png", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "figure_12_feature_importance_rank_stability.svg", bbox_inches="tight")
    plt.close(fig)


def write_readme(coef: pd.DataFrame, perm: pd.DataFrame, summary: pd.DataFrame) -> None:
    same = summary[summary["analysis"].eq("logistic_coefficient_direction")]["same_direction"].sum()
    total = summary[summary["analysis"].eq("logistic_coefficient_direction")].shape[0]
    rank_rows = summary[summary["analysis"].eq("permutation_importance_rank")][["model", "spearman_rho", "spearman_p"]].drop_duplicates()
    rank_text = "\n".join([f"- {r.model}: Spearman rho {r.spearman_rho:.3f}, p={r.spearman_p:.3f}" for r in rank_rows.itertuples()])
    text = f"""# Feature Signal Stability

Generated on 2026-06-08.

## Purpose

This analysis evaluates whether common predictors carry broadly stable signals across NHANES and MIMIC-IV despite strong source separability.

It is intended to address the review concern that a high source-classifier AUC might make transportability analysis meaningless. The key question is whether core prediction signals remain partly stable even when baseline risk and measurement processes differ.

## Methods

- Train source-specific base models in NHANES and MIMIC-IV.
- Compare standardized logistic coefficient directions.
- Compare permutation-importance ranks using Spearman rank correlation.

## Key Results

- Logistic coefficient directions matched for {int(same)} of {int(total)} common predictors.
- Permutation-importance rank stability:
{rank_text}

## Outputs

```text
outputs\\feature_signal_stability\\logistic_coefficient_stability.csv
outputs\\feature_signal_stability\\permutation_importance_stability.csv
outputs\\feature_signal_stability\\feature_signal_stability_summary.csv
outputs\\feature_signal_stability\\feature_signal_stability_metadata.json
outputs\\figures_feasibility\\figure_11_feature_signal_coefficients.png
outputs\\figures_feasibility\\figure_12_feature_importance_rank_stability.png
```

## Interpretation

If feature directions or ranks are only partly stable, the manuscript should avoid claiming that the transported model generalizes mechanistically. The stronger claim is that discrimination partly survives transport, while calibration and clinical net benefit require local target-site recalibration.
"""
    (OUT_DIR / "README_feature_signal_stability.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    coef, perm, perf = fit_models(data)
    summary = stability_summary(coef, perm)
    coef.to_csv(OUT_DIR / "logistic_coefficient_stability.csv", index=False)
    perm.to_csv(OUT_DIR / "permutation_importance_stability.csv", index=False)
    perf.to_csv(OUT_DIR / "feature_signal_model_performance.csv", index=False)
    summary.to_csv(OUT_DIR / "feature_signal_stability_summary.csv", index=False)
    metadata = {
        "features": FEATURES,
        "max_permutation_n": MAX_PERMUTATION_N,
        "permutation_repeats": PERMUTATION_REPEATS,
        "data_path": str(DATA_PATH),
        "purpose": "Assess cross-source stability of common predictor signals for dual-database transportability manuscript.",
        "notes": [
            "This is an explanatory analysis, not causal interpretation.",
            "Permutation importance is computed on within-source held-out splits.",
        ],
    }
    (OUT_DIR / "feature_signal_stability_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_feature_stability(coef, perm)
    write_readme(coef, perm, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
