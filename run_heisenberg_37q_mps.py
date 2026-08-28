#!/usr/bin/env python3
"""Standalone nested-CV runner for Heisenberg 37q/PCA-36 on CPU/MPS.

The script reproduces only the final n=2000 experiment from
``QIMED_testes-quantum_nested_pca.ipynb``. Quantum transforms are split into
small batches and written atomically so an interrupted process can resume.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pqfmlib import HeisenbergProjectiveQFM


RANDOM_STATE = 42
SAMPLE_SIZE = 2_000
OUTER_SPLITS = 10
INNER_SPLITS = 3
N_QUBITS = 37
N_PCA_COMPONENTS = 36
N_QUANTUM_FEATURES = 3 * N_QUBITS
HEISENBERG_R = 2
HEISENBERG_ALPHA = 0.1
MPS_TRUNCATION_THRESHOLD = 1e-16
SCORING = "average_precision"

METRICS = (
    "ROC-AUC",
    "PR-AUC",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
)

CLASSIFIER_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}

TARGET_COL = "readmitted_30d"
FEATURE_COLS = [
    "age_at_enc", "gender", "race", "deceased", "marital_status",
    "class_code", "enc_type_grp", "has_reason",
    "n_enc_total", "n_enc_30d", "n_enc_90d", "n_enc_365d",
    "days_since_last", "had_emer_90d", "had_imp_90d",
    "month", "quarter", "day_of_week", "is_weekend",
    "has_renal_disease", "has_diabetes", "has_hypertension", "has_mental_health",
    "n_conditions_total", "n_conditions_active",
    "last_hba1c", "last_egfr", "last_systolic_bp", "last_diastolic_bp", "last_bmi",
    "n_labs_90d", "n_vitals_90d",
    "n_procedures_90d", "n_procedures_365d", "had_surgical_90d", "had_dialysis_90d",
]
NUM_COLS = [
    "age_at_enc", "n_enc_total", "n_enc_30d", "n_enc_90d", "n_enc_365d",
    "days_since_last", "n_conditions_total", "n_conditions_active",
    "n_procedures_90d", "n_procedures_365d", "n_labs_90d", "n_vitals_90d",
]
LAB_COLS = [
    "last_hba1c", "last_egfr", "last_systolic_bp", "last_diastolic_bp", "last_bmi",
]
CAT_COLS = [
    "gender", "race", "marital_status", "class_code", "enc_type_grp",
    "month", "day_of_week", "quarter",
]
BIN_COLS = [
    "deceased", "has_reason", "had_emer_90d", "had_imp_90d",
    "has_renal_disease", "has_diabetes", "has_hypertension", "has_mental_health",
    "had_surgical_90d", "had_dialysis_90d", "is_weekend",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the n=2000 nested temporal Heisenberg 37q/PCA-36 CPU-MPS "
            "experiment with resumable quantum microbatches."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/dataset_modelo_readmissao.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/heisenberg_37q_pca36_n2000_cpu_mps_standalone"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Rows per PQFM.transform call. Lower this if memory remains high.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Aer MPS/OpenMP threads. One is the lowest-memory setting.",
    )
    parser.add_argument(
        "--max-bond-dimension",
        type=int,
        default=None,
        help="Optional MPS bond limit. Omit for no limit.",
    )
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument(
        "--max-outer-fold",
        type=int,
        default=10,
        help="Stop after this outer fold; use 1 for a pilot run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print all transform sizes without running MPS.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.shots < 1:
        parser.error("--shots must be at least 1")
    if not 1 <= args.max_outer_fold <= OUTER_SPLITS:
        parser.error(f"--max-outer-fold must be between 1 and {OUTER_SPLITS}")
    if args.max_bond_dimension is not None and args.max_bond_dimension < 1:
        parser.error("--max-bond-dimension must be positive")
    return args


def stratified_temporal_sample(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) < SAMPLE_SIZE:
        raise ValueError(
            f"Dataset has {len(frame)} rows; at least {SAMPLE_SIZE} are required."
        )
    ordered = (
        frame.sort_values("period_start", kind="mergesort")
        .reset_index()
        .rename(columns={"index": "source_index"})
    )
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )
    sample_idx, _ = next(splitter.split(ordered, ordered[TARGET_COL]))
    return (
        ordered.iloc[sample_idx]
        .sort_values("period_start", kind="mergesort")
        .reset_index(drop=True)
    )


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUM_COLS),
            ("labs", Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]), LAB_COLS),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CAT_COLS),
            ("bin", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ]), BIN_COLS),
        ]
    )


def dense_frame(values, columns) -> pd.DataFrame:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return pd.DataFrame(
        np.asarray(values, dtype=float),
        columns=list(columns),
    ).reset_index(drop=True)


def fit_preprocessing(X_train: pd.DataFrame):
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X_train)
    columns = list(preprocessor.get_feature_names_out())
    frame = dense_frame(transformed, columns)
    final_scaler = StandardScaler()
    scaled = pd.DataFrame(
        final_scaler.fit_transform(frame),
        columns=columns,
    )
    return {
        "preprocessor": preprocessor,
        "final_scaler": final_scaler,
        "columns": columns,
    }, scaled


def transform_preprocessing(fitted, X: pd.DataFrame) -> pd.DataFrame:
    transformed = fitted["preprocessor"].transform(X)
    frame = dense_frame(transformed, fitted["columns"])
    return pd.DataFrame(
        fitted["final_scaler"].transform(frame),
        columns=fitted["columns"],
    )


def fit_representations(
    X_train_raw: pd.DataFrame,
    X_validation_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fitted_preprocessing, X_train_all = fit_preprocessing(X_train_raw)
    X_validation_all = transform_preprocessing(
        fitted_preprocessing,
        X_validation_raw,
    )
    max_rank = min(X_train_all.shape[1], X_train_all.shape[0] - 1)
    if N_PCA_COMPONENTS > max_rank:
        raise ValueError(
            f"PCA-{N_PCA_COMPONENTS} requires centered rank >= "
            f"{N_PCA_COMPONENTS}, but this train split permits {max_rank}."
        )
    pca = PCA(n_components=N_PCA_COMPONENTS, svd_solver="full")
    columns = [f"PC{i:03d}" for i in range(1, N_PCA_COMPONENTS + 1)]
    X_train_pca = pd.DataFrame(
        pca.fit_transform(X_train_all),
        columns=columns,
    )
    X_validation_pca = pd.DataFrame(
        pca.transform(X_validation_all),
        columns=columns,
    )
    return (
        X_train_all.reset_index(drop=True),
        X_validation_all.reset_index(drop=True),
        X_train_pca.reset_index(drop=True),
        X_validation_pca.reset_index(drop=True),
    )


def candidate_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def fit_and_score(
    X_train,
    y_train,
    X_validation,
    y_validation,
    params: dict,
) -> float:
    classifier = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        **params,
    )
    classifier.fit(X_train, y_train)
    probabilities = classifier.predict_proba(X_validation)[:, 1]
    return float(average_precision_score(y_validation, probabilities))


def compute_metrics(y_true, prediction, probability) -> dict[str, float]:
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)
    probability = np.asarray(probability)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()
    return {
        "ROC-AUC": float(roc_auc_score(y_true, probability)),
        "PR-AUC": float(average_precision_score(y_true, probability)),
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Precision": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
    }


def memory_text() -> str:
    try:
        import psutil

        process = psutil.Process()
        rss_gib = process.memory_info().rss / (1024**3)
        available_gib = psutil.virtual_memory().available / (1024**3)
        return f"RSS={rss_gib:.2f} GiB | available={available_gib:.2f} GiB"
    except ImportError:
        return "memory=n/a (install psutil for live RSS)"


def atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def transform_in_batches(
    qfm: HeisenbergProjectiveQFM,
    X: pd.DataFrame,
    stage_dir: Path,
    *,
    batch_size: int,
    call_number: int,
    call_label: str,
) -> np.ndarray:
    stage_dir.mkdir(parents=True, exist_ok=True)
    complete_path = stage_dir / "complete.npy"
    if complete_path.exists():
        values = np.load(complete_path, allow_pickle=False)
        expected = (len(X), N_QUANTUM_FEATURES)
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError(
                f"Invalid completed cache {complete_path}: "
                f"shape={values.shape}, expected={expected}."
            )
        print(
            f"[{call_number:02d}/40] {call_label} | complete cache hit | "
            f"rows={len(X)}",
            flush=True,
        )
        return values

    batches: list[np.ndarray] = []
    total_batches = int(np.ceil(len(X) / batch_size))
    for batch_number, start in enumerate(range(0, len(X), batch_size), 1):
        end = min(start + batch_size, len(X))
        batch_path = stage_dir / f"batch_{start:06d}_{end:06d}.npy"
        expected = (end - start, N_QUANTUM_FEATURES)
        if batch_path.exists():
            quantum = np.load(batch_path, allow_pickle=False)
            if quantum.shape != expected or not np.isfinite(quantum).all():
                raise RuntimeError(
                    f"Invalid batch cache {batch_path}: "
                    f"shape={quantum.shape}, expected={expected}."
                )
            elapsed = 0.0
            status = "cache"
        else:
            started = time.perf_counter()
            quantum = np.asarray(
                qfm.transform(X.iloc[start:end]),
                dtype=float,
            )
            elapsed = time.perf_counter() - started
            if quantum.shape != expected or not np.isfinite(quantum).all():
                raise RuntimeError(
                    f"PQFM returned shape={quantum.shape}, expected={expected}."
                )
            atomic_save_array(batch_path, quantum)
            status = "computed"

        batches.append(quantum)
        print(
            f"[{call_number:02d}/40] {call_label} | "
            f"batch {batch_number}/{total_batches} rows={start}:{end} | "
            f"{status} {elapsed:.1f}s | {memory_text()}",
            flush=True,
        )
        gc.collect()

    all_values = np.vstack(batches)
    atomic_save_array(complete_path, all_values)
    print(
        f"[{call_number:02d}/40] {call_label} complete | "
        f"shape={all_values.shape} | {memory_text()}",
        flush=True,
    )
    return all_values


def build_manifest(args: argparse.Namespace, data_path: Path) -> dict:
    stat = data_path.stat()
    return {
        "schema_version": 1,
        "data_path": str(data_path.resolve()),
        "data_size_bytes": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "sample_size": SAMPLE_SIZE,
        "random_state": RANDOM_STATE,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "scoring": SCORING,
        "classifier_grid": CLASSIFIER_GRID,
        "n_qubits": N_QUBITS,
        "n_pca_components": N_PCA_COMPONENTS,
        "heisenberg_R": HEISENBERG_R,
        "heisenberg_alpha": HEISENBERG_ALPHA,
        "shots": args.shots,
        "mps_max_bond_dimension": args.max_bond_dimension,
        "mps_truncation_threshold": MPS_TRUNCATION_THRESHOLD,
        "use_tanh_scaling": True,
        "measure_2local_diagonal": False,
    }


def ensure_manifest(output_dir: Path, manifest: dict) -> None:
    path = output_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                "Output manifest does not match this run configuration. "
                "Use a different --output-dir to avoid mixing caches.\n"
                f"Existing: {json.dumps(existing, indent=2, sort_keys=True)}\n"
                f"Requested: {json.dumps(manifest, indent=2, sort_keys=True)}"
            )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"{output_dir} is non-empty but has no manifest.json. "
            "Choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, manifest)


def dry_run_report(outer_folds, max_outer_fold: int) -> None:
    total_rows = 0
    print("Transform calls that would be executed:")
    for outer_fold, (train_idx, test_idx) in enumerate(outer_folds, 1):
        if outer_fold > max_outer_fold:
            break
        inner_cv = TimeSeriesSplit(n_splits=INNER_SPLITS)
        for inner_fold, (inner_train_idx, inner_validation_idx) in enumerate(
            inner_cv.split(train_idx),
            1,
        ):
            rows = len(inner_train_idx) + len(inner_validation_idx)
            total_rows += rows
            call_number = (outer_fold - 1) * 4 + inner_fold
            print(
                f"  [{call_number:02d}/40] outer={outer_fold} "
                f"inner={inner_fold}: {rows} rows"
            )
        rows = len(train_idx) + len(test_idx)
        total_rows += rows
        call_number = (outer_fold - 1) * 4 + 4
        print(f"  [{call_number:02d}/40] outer={outer_fold}: {rows} rows")
    print(f"Total row-transforms through outer {max_outer_fold}: {total_rows}")


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, scenario), group in fold_results.groupby(
        ["config", "scenario"],
        sort=False,
    ):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            n = len(values)
            mean = float(np.mean(values)) if n else float("nan")
            std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
            if n > 1:
                critical = student_t.ppf(0.975, df=n - 1)
                margin = critical * std / np.sqrt(n)
                ci_low, ci_high = mean - margin, mean + margin
            else:
                ci_low, ci_high = float("nan"), float("nan")
            rows.append({
                "config": config,
                "scenario": scenario,
                "metric": metric,
                "n_folds": n,
                "mean": mean,
                "std": std,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })
    return pd.DataFrame(rows)


def comparison_status(delta_mean: float, p_value: float) -> str:
    significant = bool(p_value < 0.05) if np.isfinite(p_value) else False
    if np.isclose(delta_mean, 0):
        return "Sem alteração"
    if significant and delta_mean > 0:
        return "MELHORA SIGNIFICATIVA"
    if significant and delta_mean < 0:
        return "PIORA SIGNIFICATIVA"
    return "Melhora (não sig.)" if delta_mean > 0 else "Piora (não sig.)"


def paired_statistics(fold_results: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    comparisons = [
        ("classica", "quantum_only", "Clássico vs Quantum-Only"),
        ("classica", "hibrida", "Clássico vs Híbrido"),
    ]
    rows = []
    for metric in METRICS:
        paired = fold_results.pivot(
            index="outer_fold",
            columns="scenario",
            values=metric,
        )
        for baseline, challenger, label in comparisons:
            if baseline not in paired or challenger not in paired:
                continue
            pair = paired[[baseline, challenger]].dropna()
            base = pair[baseline].to_numpy(dtype=float)
            case = pair[challenger].to_numpy(dtype=float)
            delta = case - base
            if len(delta) < 2:
                statistic, p_value = float("nan"), float("nan")
                ci_low, ci_high = float("nan"), float("nan")
            else:
                if np.allclose(delta, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = wilcoxon(
                        case,
                        base,
                        alternative="two-sided",
                    )
                indices = rng.integers(0, len(delta), size=(2_000, len(delta)))
                means = delta[indices].mean(axis=1)
                ci_low, ci_high = np.percentile(means, [2.5, 97.5])
            delta_mean = float(np.mean(delta)) if len(delta) else float("nan")
            rows.append({
                "comparison": label,
                "metric": metric,
                "folds": len(delta),
                "baseline_mean": float(np.mean(base)) if len(base) else float("nan"),
                "case_mean": float(np.mean(case)) if len(case) else float("nan"),
                "delta_mean": delta_mean,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "wilcoxon_W": float(statistic),
                "p_value": float(p_value),
                "status": comparison_status(delta_mean, p_value),
            })
    return pd.DataFrame(rows)


def load_completed_fold_payloads(output_dir: Path, max_outer_fold: int):
    fold_rows = []
    tuning_rows = []
    for outer_fold in range(1, max_outer_fold + 1):
        path = output_dir / "fold_results" / f"outer_{outer_fold:02d}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_rows.extend(payload["fold_results"])
        tuning_rows.extend(payload["tuning_results"])
    return pd.DataFrame(fold_rows), pd.DataFrame(tuning_rows)


def write_aggregate_reports(output_dir: Path, max_outer_fold: int) -> None:
    fold_results, tuning_results = load_completed_fold_payloads(
        output_dir,
        max_outer_fold,
    )
    if fold_results.empty:
        return
    summary = summarize_results(fold_results)
    statistics = paired_statistics(fold_results)
    reports = output_dir / "reports"
    atomic_write_csv(reports / "fold_results.csv", fold_results)
    atomic_write_csv(reports / "tuning_results.csv", tuning_results)
    atomic_write_csv(reports / "summary.csv", summary)
    atomic_write_csv(reports / "paired_wilcoxon.csv", statistics)
    print("\nCurrent summary:")
    print(summary.to_string(index=False))
    if not statistics.empty:
        print("\nCurrent paired comparisons:")
        print(statistics.to_string(index=False))


def run_experiment(args: argparse.Namespace) -> int:
    data_path = args.data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    frame = pd.read_parquet(data_path)
    required = set(FEATURE_COLS + [TARGET_COL, "period_start"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    sampled = stratified_temporal_sample(frame)
    X = sampled[FEATURE_COLS].copy()
    y = sampled[TARGET_COL].astype(int).reset_index(drop=True)
    outer_folds = [
        (train.copy(), test.copy())
        for train, test in TimeSeriesSplit(n_splits=OUTER_SPLITS).split(X)
    ]
    print(
        f"Sample: {X.shape} | positive rate={y.mean():.4f} | "
        f"period={sampled['period_start'].min()} -> {sampled['period_start'].max()}"
    )
    if args.dry_run:
        dry_run_report(outer_folds, args.max_outer_fold)
        return 0

    output_dir = args.output_dir.resolve()
    manifest = build_manifest(args, data_path)
    ensure_manifest(output_dir, manifest)

    run_label = "heisenberg_37q_pca36_n2000_cpu_mps"
    if args.max_bond_dimension is not None:
        run_label += f"_bond_{args.max_bond_dimension}"

    # Explicit PQFM configuration for this standalone experiment.
    qfm = HeisenbergProjectiveQFM(
        name_file=run_label,
        seed=RANDOM_STATE,
        ideal=True,
        simulation=True,
        fakebackend=False,
        shots=args.shots,
        q_enc=N_QUBITS,
        R=HEISENBERG_R,
        alpha=HEISENBERG_ALPHA,
        use_tanh_scaling=True,
        measure_2local_diagonal=False,
        mps=True,
        use_gpu_statevector=False,
        mps_max_bond_dimension=args.max_bond_dimension,
        mps_truncation_threshold=MPS_TRUNCATION_THRESHOLD,
        output_root=str(output_dir / "pqfm"),
    )
    print("Preparing the Heisenberg 37q MPS circuit once...", flush=True)
    qfm.fit(np.zeros((1, N_PCA_COMPONENTS), dtype=float))
    qfm.backend.set_options(
        mps_log_data=False,
        max_parallel_threads=args.threads,
        max_parallel_experiments=1,
        max_parallel_shots=1,
        mps_omp_threads=args.threads,
    )
    assert qfm.backend.options.method == "matrix_product_state"
    assert qfm.backend.options.device == "CPU"
    assert qfm.theta_info["features_per_block"] == 36
    assert qfm.theta_info["num_blocks"] == 1
    assert qfm.theta_info["total_slots"] == 36
    print(
        f"Backend={qfm.backend.options.method}/{qfm.backend.options.device} | "
        f"batch_size={args.batch_size} | threads={args.threads} | "
        f"bond_limit={args.max_bond_dimension} | {memory_text()}",
        flush=True,
    )

    candidates = list(ParameterGrid(CLASSIFIER_GRID))
    for outer_fold, (train_idx, test_idx) in enumerate(outer_folds, 1):
        if outer_fold > args.max_outer_fold:
            break
        result_path = (
            output_dir / "fold_results" / f"outer_{outer_fold:02d}.json"
        )
        if result_path.exists():
            print(
                f"Outer fold {outer_fold}/{OUTER_SPLITS}: result checkpoint hit; skipping.",
                flush=True,
            )
            continue

        print(f"\nOUTER FOLD {outer_fold}/{OUTER_SPLITS}", flush=True)
        X_outer_train_raw = X.iloc[train_idx].copy()
        X_outer_test_raw = X.iloc[test_idx].copy()
        y_outer_train = y.iloc[train_idx].reset_index(drop=True)
        y_outer_test = y.iloc[test_idx].reset_index(drop=True)
        scores = {
            scenario: {candidate_key(params): [] for params in candidates}
            for scenario in ("classica", "quantum_only", "hibrida")
        }

        inner_cv = TimeSeriesSplit(n_splits=INNER_SPLITS)
        for inner_fold, (inner_train_idx, inner_validation_idx) in enumerate(
            inner_cv.split(X_outer_train_raw),
            1,
        ):
            X_inner_train_raw = X_outer_train_raw.iloc[inner_train_idx].copy()
            X_inner_validation_raw = X_outer_train_raw.iloc[
                inner_validation_idx
            ].copy()
            y_inner_train = y_outer_train.iloc[inner_train_idx].reset_index(drop=True)
            y_inner_validation = y_outer_train.iloc[
                inner_validation_idx
            ].reset_index(drop=True)
            if y_inner_train.nunique() < 2 or y_inner_validation.nunique() < 2:
                raise ValueError(
                    f"outer={outer_fold}, inner={inner_fold} does not contain "
                    "both target classes in train and validation."
                )

            (
                X_inner_train_all,
                X_inner_validation_all,
                X_inner_train_pca,
                X_inner_validation_pca,
            ) = fit_representations(
                X_inner_train_raw,
                X_inner_validation_raw,
            )
            n_inner_train = len(X_inner_train_pca)
            X_inner_pca_all = pd.concat(
                [X_inner_train_pca, X_inner_validation_pca],
                ignore_index=True,
            )
            call_number = (outer_fold - 1) * 4 + inner_fold
            Xq_inner_all = transform_in_batches(
                qfm,
                X_inner_pca_all,
                output_dir / "quantum_cache" / f"outer_{outer_fold:02d}" / f"inner_{inner_fold:02d}",
                batch_size=args.batch_size,
                call_number=call_number,
                call_label=f"outer={outer_fold}/10 inner={inner_fold}/3",
            )
            Xq_inner_train = pd.DataFrame(
                Xq_inner_all[:n_inner_train],
                columns=[f"heisenberg_q_{i}" for i in range(N_QUANTUM_FEATURES)],
            )
            Xq_inner_validation = pd.DataFrame(
                Xq_inner_all[n_inner_train:],
                columns=Xq_inner_train.columns,
            )
            X_inner_train_hybrid = pd.concat(
                [X_inner_train_all, Xq_inner_train],
                axis=1,
            )
            X_inner_validation_hybrid = pd.concat(
                [X_inner_validation_all, Xq_inner_validation],
                axis=1,
            )
            representations = {
                "classica": (X_inner_train_all, X_inner_validation_all),
                "quantum_only": (Xq_inner_train, Xq_inner_validation),
                "hibrida": (X_inner_train_hybrid, X_inner_validation_hybrid),
            }
            for scenario, (X_train_rep, X_validation_rep) in representations.items():
                for params in candidates:
                    scores[scenario][candidate_key(params)].append(
                        fit_and_score(
                            X_train_rep,
                            y_inner_train,
                            X_validation_rep,
                            y_inner_validation,
                            params,
                        )
                    )
            del Xq_inner_all, representations
            gc.collect()

        best_params = {}
        tuning_rows = []
        for scenario in ("classica", "quantum_only", "hibrida"):
            means = {
                key: float(np.mean(values))
                for key, values in scores[scenario].items()
            }
            best_key = max(means, key=means.get)
            best_params[scenario] = json.loads(best_key)
            for key, values in scores[scenario].items():
                tuning_rows.append({
                    "config": run_label,
                    "outer_fold": outer_fold,
                    "scenario": scenario,
                    "params": key,
                    "mean_inner_average_precision": float(np.mean(values)),
                    "std_inner_average_precision": float(np.std(values, ddof=1)),
                })

        (
            X_outer_train_all,
            X_outer_test_all,
            X_outer_train_pca,
            X_outer_test_pca,
        ) = fit_representations(
            X_outer_train_raw,
            X_outer_test_raw,
        )
        n_outer_train = len(X_outer_train_pca)
        X_outer_pca_all = pd.concat(
            [X_outer_train_pca, X_outer_test_pca],
            ignore_index=True,
        )
        call_number = (outer_fold - 1) * 4 + 4
        Xq_outer_all = transform_in_batches(
            qfm,
            X_outer_pca_all,
            output_dir / "quantum_cache" / f"outer_{outer_fold:02d}" / "outer",
            batch_size=args.batch_size,
            call_number=call_number,
            call_label=f"outer={outer_fold}/10 OUTER",
        )
        Xq_outer_train = pd.DataFrame(
            Xq_outer_all[:n_outer_train],
            columns=[f"heisenberg_q_{i}" for i in range(N_QUANTUM_FEATURES)],
        )
        Xq_outer_test = pd.DataFrame(
            Xq_outer_all[n_outer_train:],
            columns=Xq_outer_train.columns,
        )
        X_outer_train_hybrid = pd.concat(
            [X_outer_train_all, Xq_outer_train],
            axis=1,
        )
        X_outer_test_hybrid = pd.concat(
            [X_outer_test_all, Xq_outer_test],
            axis=1,
        )
        outer_representations = {
            "classica": (X_outer_train_all, X_outer_test_all),
            "quantum_only": (Xq_outer_train, Xq_outer_test),
            "hibrida": (X_outer_train_hybrid, X_outer_test_hybrid),
        }
        fold_rows = []
        for scenario, (X_train_rep, X_test_rep) in outer_representations.items():
            classifier = GradientBoostingClassifier(
                random_state=RANDOM_STATE,
                **best_params[scenario],
            )
            classifier.fit(X_train_rep, y_outer_train)
            prediction = classifier.predict(X_test_rep)
            probability = classifier.predict_proba(X_test_rep)[:, 1]
            fold_rows.append({
                "config": run_label,
                "scenario": scenario,
                "outer_fold": outer_fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_start_position": int(test_idx[0]),
                "test_end_position": int(test_idx[-1]),
                "best_params": candidate_key(best_params[scenario]),
                **compute_metrics(y_outer_test, prediction, probability),
            })

        atomic_write_json(
            result_path,
            {
                "config": run_label,
                "fold_results": fold_rows,
                "tuning_results": tuning_rows,
            },
        )
        print(
            f"Outer fold {outer_fold} completed and saved to {result_path}",
            flush=True,
        )
        write_aggregate_reports(output_dir, args.max_outer_fold)
        del Xq_outer_all, outer_representations
        gc.collect()

    write_aggregate_reports(output_dir, args.max_outer_fold)
    print(f"\nRun finished through outer fold {args.max_outer_fold}.")
    print(f"Outputs: {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_experiment(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed microbatches remain cached for resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
