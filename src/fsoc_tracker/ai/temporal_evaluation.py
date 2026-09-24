"""Temporal predictor evaluation with baselines.

Compares:
1. Last-position baseline (no motion)
2. Constant-velocity baseline (velocity * horizon)
3. Kalman filter prediction
4. Learned GRU predictor

Reports RMSE, MAE, P95 error per horizon.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fsoc_tracker.ai.temporal_dataset import FEATURE_DIM, HORIZONS_S


def last_position_baseline(
    features: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Baseline: predict zero displacement (target stays at current position).
    
    Args:
        features: (N, T, FEATURE_DIM) 
        mask: (N, T)
    Returns:
        predictions: (N, T, H, 2) - all zeros
    """
    N, T, _ = features.shape
    H = len(HORIZONS_S)
    return np.zeros((N, T, H, 2), dtype=np.float64)


def constant_velocity_baseline(
    features: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Baseline: predict velocity * horizon for each horizon.
    
    Args:
        features: (N, T, FEATURE_DIM)
        mask: (N, T)
    Returns:
        predictions: (N, T, H, 2)
    """
    N, T, F = features.shape
    H = len(HORIZONS_S)
    preds = np.zeros((N, T, H, 2), dtype=np.float64)
    
    vel_x = features[:, :, 6]  # velocity_x_px_s
    vel_y = features[:, :, 7]  # velocity_y_px_s
    
    for h_idx, horizon in enumerate(HORIZONS_S):
        preds[:, :, h_idx, 0] = vel_x * horizon
        preds[:, :, h_idx, 1] = vel_y * horizon
    
    return preds


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    method_name: str = "unknown",
) -> dict[str, Any]:
    """Evaluate predictions against ground truth.
    
    Args:
        predictions: (N, T, H, 2) predicted displacements
        targets: (N, T, H, 2) ground truth displacements
        mask: (N, T) valid timestep mask
        method_name: name for reporting
    
    Returns:
        Dictionary with per-horizon and aggregate metrics
    """
    len(HORIZONS_S)
    mask_exp = mask[:, :, np.newaxis, np.newaxis]  # (N, T, 1, 1)
    
    errors = (predictions - targets) * mask_exp
    euclidean_errors = np.sqrt(np.sum(errors ** 2, axis=-1))  # (N, T, H)
    
    valid_count = mask_exp.sum()
    
    results = {
        "method": method_name,
        "aggregate": {},
        "per_horizon": [],
    }
    
    total_mse = float(np.sum(errors ** 2) / max(valid_count, 1.0))
    total_rmse = float(np.sqrt(total_mse))
    total_mae = float(np.sum(np.abs(errors)) / max(valid_count, 1.0))
    valid_euclidean = euclidean_errors * mask[:, :, np.newaxis]
    total_p95 = float(np.percentile(valid_euclidean[valid_euclidean > 0], 95)) if np.any(valid_euclidean > 0) else 0.0
    
    results["aggregate"] = {
        "rmse_px": round(total_rmse, 4),
        "mae_px": round(total_mae, 4),
        "p95_euclidean_px": round(total_p95, 4),
        "total_samples": int(valid_count),
    }
    
    for h_idx, horizon in enumerate(HORIZONS_S):
        h_errors = errors[:, :, h_idx, :]  # (N, T, 2)
        h_euclidean = euclidean_errors[:, :, h_idx]  # (N, T)
        h_mask = mask  # (N, T)
        
        h_mse = float(np.sum(h_errors ** 2 * h_mask[:, :, np.newaxis]) / max(h_mask.sum(), 1.0))
        h_rmse = float(np.sqrt(h_mse))
        h_mae = float(np.sum(np.abs(h_errors) * h_mask[:, :, np.newaxis]) / max(h_mask.sum(), 1.0))
        h_p95 = float(np.percentile(h_euclidean[h_mask > 0], 95)) if np.any(h_mask > 0) else 0.0
        
        results["per_horizon"].append({
            "horizon_s": horizon,
            "rmse_px": round(h_rmse, 4),
            "mae_px": round(h_mae, 4),
            "p95_euclidean_px": round(h_p95, 4),
        })
    
    return results


def load_dataset_split(dataset_dir: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features, displacements, mask from a temporal dataset split."""
    base = Path(dataset_dir) / split
    features = np.load(base / "features.npy").astype(np.float64)
    displacements = np.load(base / "future_displacements.npy").astype(np.float64)
    mask = np.load(base / "mask.npy").astype(np.float64)
    return features, displacements, mask


def run_baseline_evaluation(
    dataset_dir: str = "artifacts/datasets/temporal-v1",
    split: str = "test",
    eval_timestep: int = 150,
) -> dict[str, Any]:
    """Run all baselines on a dataset split at a specific timestep."""
    features, targets, mask = load_dataset_split(dataset_dir, split)
    
    # Evaluate at a mid-sequence timestep (not the last frame)
    eval_t = min(eval_timestep, features.shape[1] - 16)
    features_t = features[:, eval_t:eval_t+1, :]  # (N, 1, F) - single timestep
    targets_t = targets[:, eval_t:eval_t+1, :, :]  # (N, 1, H, 2)
    mask_t = mask[:, eval_t:eval_t+1]  # (N, 1)
    
    print(f"[EVAL] Split: {split}, timestep: {eval_t}, features: {features_t.shape}, targets: {targets_t.shape}")
    
    results = {}
    
    lp = last_position_baseline(features_t, mask_t)
    results["last_position"] = evaluate_predictions(lp, targets_t, mask_t, "last_position")
    print(f"  Last-position: RMSE={results['last_position']['aggregate']['rmse_px']:.4f}px")
    
    cv = constant_velocity_baseline(features_t, mask_t)
    results["constant_velocity"] = evaluate_predictions(cv, targets_t, mask_t, "constant_velocity")
    print(f"  Constant-velocity: RMSE={results['constant_velocity']['aggregate']['rmse_px']:.4f}px")
    
    return results


def run_gru_evaluation(
    dataset_dir: str = "artifacts/datasets/temporal-v1",
    model_path: str = "artifacts/models/temporal-v1/temporal_gru.pt",
    split: str = "test",
    eval_timestep: int = 150,
) -> dict[str, Any]:
    """Run GRU predictor on a dataset split.
    
    GRU outputs (N, H, 2) from the full sequence. We evaluate at a fixed
    mid-sequence timestep (not the last frame, which has zero displacement).
    
    Args:
        eval_timestep: Which timestep index to evaluate at (default 150 of 300).
    """
    import torch

    from fsoc_tracker.ai.neural import build_temporal_predictor
    
    features, targets, mask = load_dataset_split(dataset_dir, split)
    N, T, H, _ = targets.shape
    
    # Ensure eval_timestep is valid
    eval_t = min(eval_timestep, T - 16)  # need room for longest horizon
    
    model = build_temporal_predictor(
        feature_dim=FEATURE_DIM,
        hidden_dim=32,
        horizons_s=tuple(HORIZONS_S),
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    with torch.no_grad():
        x = torch.from_numpy(features).float()
        output = model(x)
        gru_preds = output["displacement"].numpy().astype(np.float64)  # (N, H, 2)
    
    # Evaluate at the specified mid-sequence timestep
    targets_at_t = targets[:, eval_t]  # (N, H, 2)
    mask_at_t = mask[:, eval_t]  # (N,)
    
    gru_preds_4d = gru_preds[:, np.newaxis, :, :]  # (N, 1, H, 2)
    targets_4d = targets_at_t[:, np.newaxis, :, :]  # (N, 1, H, 2)
    mask_2d = mask_at_t[:, np.newaxis]  # (N, 1)
    
    result = evaluate_predictions(gru_preds_4d, targets_4d, mask_2d, "gru_predictor")
    print(f"  GRU predictor (t={eval_t}): RMSE={result['aggregate']['rmse_px']:.4f}px")
    
    return result


def run_full_evaluation(
    dataset_dir: str = "artifacts/datasets/temporal-v1",
    model_path: str = "artifacts/models/temporal-v1/temporal_gru.pt",
    output_dir: str = "runs/temporal-eval",
) -> dict[str, Any]:
    """Run complete evaluation with all methods on all splits."""
    all_results = {}
    
    for split in ["train", "validation", "test", "hard_test"]:
        print(f"\n=== {split.upper()} ===")
        split_results = run_baseline_evaluation(dataset_dir, split)
        
        try:
            gru_result = run_gru_evaluation(dataset_dir, model_path, split)
            split_results["gru_predictor"] = gru_result
        except Exception as e:
            print(f"  GRU evaluation failed: {e}")
            split_results["gru_predictor"] = {"error": str(e)}
        
        all_results[split] = split_results
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_report.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    
    print("\n=== SUMMARY ===")
    for split_name, split_data in all_results.items():
        if isinstance(split_data, dict) and "aggregate" in split_data.get("constant_velocity", {}):
            cv_rmse = split_data["constant_velocity"]["aggregate"]["rmse_px"]
            gru_data = split_data.get("gru_predictor", {})
            gru_rmse = gru_data.get("aggregate", {}).get("rmse_px", "N/A")
            print(f"  {split_name}: CV={cv_rmse:.4f}px, GRU={gru_rmse}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artifacts/datasets/temporal-v1")
    parser.add_argument("--model", default="artifacts/models/temporal-v1/temporal_gru.pt")
    parser.add_argument("--output", default="runs/temporal-eval")
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    
    if args.split:
        if args.split in ("test", "train", "validation", "hard_test"):
            baselines = run_baseline_evaluation(args.dataset, args.split)
            try:
                gru = run_gru_evaluation(args.dataset, args.model, args.split)
                baselines["gru_predictor"] = gru
            except Exception as e:
                print(f"GRU failed: {e}")
            print(json.dumps(baselines, indent=2))
    else:
        run_full_evaluation(args.dataset, args.model, args.output)
