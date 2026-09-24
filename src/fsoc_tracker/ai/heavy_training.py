"""Massive AI/ML training pipeline for FSOC beacon tracking.

Trains ALL models simultaneously:
1. Beacon Detection CNN — detect beacon in camera images
2. Trajectory Prediction GRU — predict where beacon will be
3. Control Policy — learn optimal beam steering

Uses synthetic data from generated benchmark worlds.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACTS_DIR = Path("artifacts/models/heavy-training-v1")
DATASET_DIR = ARTIFACTS_DIR / "dataset"


def _seed_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ============================================================
# PART 1: Generate Massive Synthetic Dataset
# ============================================================

def generate_detection_samples(
    num_samples: int = 10000,
    image_width: int = 640,
    image_height: int = 480,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic beacon detection images with labels.

    Returns:
        images: (N, H, W) float32 array, pixel values 0-255
        labels: (N, 5) float32 — [center_x, center_y, width, height, confidence]
    """
    rng = _seed_rng(seed)
    images = np.zeros((num_samples, image_height, image_width), dtype=np.float32)
    labels = np.zeros((num_samples, 5), dtype=np.float32)

    for i in range(num_samples):
        img = np.full((image_height, image_width), 5.0, dtype=np.float32)

        # Background noise level varies
        noise_level = rng.uniform(2.0, 15.0)
        img += rng.normal(0, noise_level, img.shape).astype(np.float32)

        # Decide: beacon present (80%) or absent (20%)
        has_beacon = rng.random() < 0.8

        if has_beacon:
            # Beacon position (somewhere in the image)
            cx = rng.uniform(20, image_width - 20)
            cy = rng.uniform(20, image_height - 20)
            size_px = rng.uniform(3.0, 20.0)
            brightness = rng.uniform(150.0, 255.0)

            # Draw Gaussian beacon
            half = size_px / 2.0
            sigma = half / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            if sigma < 0.5:
                sigma = 0.5

            x_min = max(0, int(cx - 3 * sigma))
            x_max = min(image_width, int(cx + 3 * sigma) + 1)
            y_min = max(0, int(cy - 3 * sigma))
            y_max = min(image_height, int(cy + 3 * sigma) + 1)

            yy, xx = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing="ij")
            gauss = brightness * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            img[y_min:y_max, x_min:x_max] += gauss.astype(np.float32)

            # Random distractors (30% chance)
            if rng.random() < 0.3:
                n_distractors = rng.integers(1, 4)
                for _ in range(n_distractors):
                    dx = rng.uniform(0, image_width)
                    dy = rng.uniform(0, image_height)
                    ds = rng.uniform(1.0, 4.0)
                    db = rng.uniform(50.0, 120.0)
                    dsigma = ds / (2.0 * math.sqrt(2.0 * math.log(2.0)))
                    if dsigma < 0.3:
                        dsigma = 0.3
                    dx_min = max(0, int(dx - 3 * dsigma))
                    dx_max = min(image_width, int(dx + 3 * dsigma) + 1)
                    dy_min = max(0, int(dy - 3 * dsigma))
                    dy_max = min(image_height, int(dy + 3 * dsigma) + 1)
                    if dx_max > dx_min and dy_max > dy_min:
                        dyy, dxx = np.meshgrid(np.arange(dy_min, dy_max), np.arange(dx_min, dx_max), indexing="ij")
                        dgauss = db * np.exp(-((dxx - dx) ** 2 + (dyy - dy) ** 2) / (2 * dsigma ** 2))
                        img[dy_min:dy_max, dx_min:dx_max] += dgauss.astype(np.float32)

            # Apply image-level disturbances (50% chance)
            if rng.random() < 0.5:
                dist_type = rng.choice(["gaussian", "salt_pepper", "fog", "low_light"])
                if dist_type == "gaussian":
                    img += rng.normal(0, rng.uniform(5, 30), img.shape).astype(np.float32)
                elif dist_type == "salt_pepper":
                    prob = rng.uniform(0.01, 0.05)
                    mask = rng.random(img.shape) < prob
                    img[mask] = rng.choice([0.0, 255.0], size=mask.sum())
                elif dist_type == "fog":
                    fog_level = rng.uniform(0.1, 0.4)
                    img = img * (1 - fog_level) + fog_level * 128.0
                elif dist_type == "low_light":
                    img *= rng.uniform(0.2, 0.6)

            labels[i] = [cx, cy, size_px, brightness, 1.0]
        else:
            # No beacon — just noise
            labels[i] = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Clip and store
        np.clip(img, 0.0, 255.0, out=img)
        images[i] = img

    return images, labels


def generate_temporal_sequences(
    num_sequences: int = 2000,
    sequence_length: int = 30,
    feature_dim: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic temporal sequences for trajectory prediction.

    Returns:
        sequences: (N, T, feature_dim) float32
        targets: (N, 5) float32 — [dx, dy, vx, vy, will_lose]
    """
    rng = _seed_rng(seed)
    sequences = np.zeros((num_sequences, sequence_length, feature_dim), dtype=np.float32)
    targets = np.zeros((num_sequences, 5), dtype=np.float32)

    trajectory_types = ["linear", "circular", "figure8", "random"]

    for i in range(num_sequences):
        traj_type = rng.choice(trajectory_types)
        # Simulated beacon center starts in image
        cx = rng.uniform(200, 440)
        cy = rng.uniform(150, 330)
        vx = rng.uniform(-3.0, 3.0)
        vy = rng.uniform(-2.0, 2.0)

        for t in range(sequence_length):
            # Position features
            t_norm = t / sequence_length
            rel_x = cx / 640.0
            rel_y = cy / 480.0

            # Simulated detection features
            detected = 1.0 if (10 < cx < 630 and 10 < cy < 470) else 0.0
            confidence = rng.uniform(0.7, 1.0) if detected > 0.5 else 0.0
            size_px = rng.uniform(5.0, 15.0) if detected > 0.5 else 0.0
            brightness = rng.uniform(200.0, 255.0) if detected > 0.5 else 0.0
            residual = rng.uniform(0.0, 2.0) if detected > 0.5 else 0.0
            velocity_x = vx / 640.0
            velocity_y = vy / 480.0
            velocity_mag = math.sqrt(velocity_x ** 2 + velocity_y ** 2)
            time_since_det = 0.0 if detected > 0.5 else t * 0.033
            tracking_quality = confidence * 0.9 if detected > 0.5 else 0.0
            uncertainty = (1.0 - confidence) * 0.5
            frame_idx = t_norm

            features = np.array([
                rel_x, rel_y, detected, confidence, size_px,
                brightness, residual, velocity_x, velocity_y, velocity_mag,
                time_since_det, tracking_quality, uncertainty, frame_idx,
                1.0 if traj_type == "circular" else 0.0,
            ], dtype=np.float32)

            sequences[i, t] = features

            # Update position based on trajectory type
            omega = 0.3
            if traj_type == "linear":
                cx += vx
                cy += vy
            elif traj_type == "circular":
                cx = 320 + 150 * math.cos(omega * t)
                cy = 240 + 100 * math.sin(omega * t)
            elif traj_type == "figure8":
                cx = 320 + 150 * math.sin(omega * t)
                cy = 240 + 100 * math.sin(2 * omega * t)
            elif traj_type == "random":
                vx += rng.normal(0, 0.3)
                vy += rng.normal(0, 0.2)
                vx = np.clip(vx, -5.0, 5.0)
                vy = np.clip(vy, -3.0, 3.0)
                cx += vx
                cy += vy

        # Target: predict next displacement
        targets[i, 0] = vx / 640.0  # normalized dx
        targets[i, 1] = vy / 480.0  # normalized dy
        targets[i, 2] = vx / 640.0  # normalized vx
        targets[i, 3] = vy / 480.0  # normalized vy
        targets[i, 4] = 1.0 if (cx < 0 or cx > 640 or cy < 0 or cy > 480) else 0.0  # will_lose

    return sequences, targets


# ============================================================
# PART 2: Beacon Detection CNN Training
# ============================================================

def train_detection_cnn(
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    val_split: float = 0.15,
) -> dict[str, Any]:
    """Train the beacon detection CNN using the existing BeaconCNN architecture."""
    # BeaconCNN expects 128x128 input - resize images
    from scipy.ndimage import zoom

    from fsoc_tracker.ai.model import BeaconCNN
    target_h, target_w = 128, 128
    zoom_y = target_h / images.shape[1]
    zoom_x = target_w / images.shape[2]

    small_images = np.zeros((len(images), target_h, target_w), dtype=np.float32)
    for i in range(len(images)):
        small_images[i] = zoom(images[i], (zoom_y, zoom_x), order=1)

    # Scale labels to model resolution
    small_labels = labels.copy()
    small_labels[:, 0] *= zoom_x  # cx
    small_labels[:, 1] *= zoom_y  # cy
    small_labels[:, 2] *= min(zoom_x, zoom_y)  # size

    N = len(small_images)
    split = int(N * (1 - val_split))
    train_imgs = small_images[:split]
    val_imgs = small_images[split:]
    train_labels = small_labels[:split]
    val_labels = small_labels[split:]

    model = BeaconCNN()
    save_dir = ARTIFACTS_DIR / "detection-cnn"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        indices = np.random.permutation(len(train_imgs))
        train_losses = []
        for start in range(0, len(train_imgs), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_imgs = train_imgs[batch_idx]
            batch_labels = train_labels[batch_idx]

            batch_loss = 0.0
            for j in range(len(batch_imgs)):
                output = model.forward(batch_imgs[j:j+1])
                if hasattr(output, 'heatmap'):
                    hm = output.heatmap
                    cx, cy, size = batch_labels[j, 0], batch_labels[j, 1], batch_labels[j, 2]
                    if batch_labels[j, 4] > 0.5 and size > 0:
                        yy, xx = np.mgrid[:hm.shape[0], :hm.shape[1]]
                        sigma = max(size / 8.0, 1.0)
                        target_hm = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
                    else:
                        target_hm = np.zeros_like(hm)
                    batch_loss += float(np.mean((hm - target_hm)**2))

            train_losses.append(batch_loss / max(1, len(batch_imgs)))

        val_losses = []
        for j in range(0, len(val_imgs), batch_size):
            batch_imgs = val_imgs[j:j+batch_size]
            batch_labels = val_labels[j:j+batch_size]
            for k in range(len(batch_imgs)):
                output = model.forward(batch_imgs[k:k+1])
                if hasattr(output, 'heatmap'):
                    hm = output.heatmap
                    cx, cy, size = batch_labels[k, 0], batch_labels[k, 1], batch_labels[k, 2]
                    if batch_labels[k, 4] > 0.5 and size > 0:
                        yy, xx = np.mgrid[:hm.shape[0], :hm.shape[1]]
                        sigma = max(size / 8.0, 1.0)
                        target_hm = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
                    else:
                        target_hm = np.zeros_like(hm)
                    val_losses.append(float(np.mean((hm - target_hm)**2)))

        avg_train = np.mean(train_losses) if train_losses else 0
        avg_val = np.mean(val_losses) if val_losses else 0

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            model.save_weights(str(save_dir / "beacon_cnn.npz"))
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train:.6f} val_loss={avg_val:.6f}")

        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return {
        "best_val_loss": best_val_loss,
        "total_epochs": epoch + 1,
    }


# ============================================================
# PART 3: Trajectory Prediction GRU Training
# ============================================================

def train_trajectory_gru(
    sequences: np.ndarray,
    targets: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    hidden_dim: int = 64,
    val_split: float = 0.15,
) -> dict[str, Any]:
    """Train trajectory prediction GRU model."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"error": "PyTorch not available"}

    class TrajectoryGRU(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, h = self.gru(x)
            return self.fc(h[-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = len(sequences)
    split = int(N * (1 - val_split))

    X_train = torch.tensor(sequences[:split], dtype=torch.float32).to(device)
    y_train = torch.tensor(targets[:split], dtype=torch.float32).to(device)
    X_val = torch.tensor(sequences[split:], dtype=torch.float32).to(device)
    y_val = torch.tensor(targets[split:], dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TrajectoryGRU(sequences.shape[2], hidden_dim, targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_dir = ARTIFACTS_DIR / "trajectory-gru"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={train_loss:.6f} val={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return {
        "best_val_loss": best_val_loss,
        "total_epochs": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    }


# ============================================================
# PART 4: Control Policy Training
# ============================================================

def train_control_policy(
    sequences: np.ndarray,
    targets: np.ndarray,
    epochs: int = 80,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    hidden_dim: int = 64,
    val_split: float = 0.15,
) -> dict[str, Any]:
    """Train a control policy that maps tracking state to pan/tilt commands."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"error": "PyTorch not available"}

    class ControlPolicyGRU(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 3),  # pan_rate, tilt_rate, confidence
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, h = self.gru(x)
            out = self.fc(h[-1])
            pan_rate = torch.tanh(out[:, 0]) * 10.0  # clamp to ±10 deg/s
            tilt_rate = torch.tanh(out[:, 1]) * 10.0
            confidence = torch.sigmoid(out[:, 2])
            return torch.stack([pan_rate, tilt_rate, confidence], dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate control labels from sequences (expert PID labels)
    control_targets = np.zeros((len(sequences), 3), dtype=np.float32)
    for i in range(len(sequences)):
        seq = sequences[i]
        last_x = seq[-1, 0] * 640.0
        last_y = seq[-1, 1] * 480.0
        cx, cy = 320.0, 240.0
        err_x = (last_x - cx) / 640.0
        err_y = (last_y - cy) / 480.0
        control_targets[i, 0] = np.clip(-err_x * 10.0, -10.0, 10.0)
        control_targets[i, 1] = np.clip(-err_y * 10.0, -10.0, 10.0)
        control_targets[i, 2] = 1.0 if seq[-1, 2] > 0.5 else 0.0

    N = len(sequences)
    split = int(N * (1 - val_split))

    X_train = torch.tensor(sequences[:split], dtype=torch.float32).to(device)
    y_train = torch.tensor(control_targets[:split], dtype=torch.float32).to(device)
    X_val = torch.tensor(sequences[split:], dtype=torch.float32).to(device)
    y_val = torch.tensor(control_targets[split:], dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ControlPolicyGRU(sequences.shape[2], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_dir = ARTIFACTS_DIR / "control-policy"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={train_loss:.6f} val={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return {
        "best_val_loss": best_val_loss,
        "total_epochs": epoch + 1,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
    }


# ============================================================
# PART 5: Mission Model Training (Motion + Situation + Policy)
# ============================================================

def train_mission_models(
    sequences: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    """Train the mission brain models (motion, situation, policy classifiers)."""
    from fsoc_tracker.ai.learned import LearnedFeatureClassifier, LearnedMotionModel

    results = {}

    # Train motion model
    motion_model = LearnedMotionModel()
    motion_X = sequences[:, -1, :]  # last frame features
    motion_y = targets[:, :2]  # dx, dy

    # Split
    split = int(len(motion_X) * 0.85)
    motion_model.fit(motion_X[:split], motion_y[:split])
    motion_metrics = motion_model.evaluate(motion_X[split:], motion_y[split:])
    results["motion"] = {
        "rmse": motion_metrics.rmse_euclidean,
        "p95": motion_metrics.p95_euclidean,
    }

    # Train situation classifier
    situation_model = LearnedFeatureClassifier()
    sit_features = sequences[:, -1, :]

    # Generate situation labels
    sit_labels = np.zeros(len(sit_features), dtype=int)
    for i in range(len(sit_features)):
        if sit_features[i, 2] < 0.5:
            sit_labels[i] = 3  # TARGET_LOST
        elif sit_features[i, 3] < 0.3:
            sit_labels[i] = 1  # LOW_CONFIDENCE
        elif sit_features[i, 8] > 3.0:
            sit_labels[i] = 2  # HIGH_NOISE
        else:
            sit_labels[i] = 0  # NORMAL_TRACKING

    split = int(len(sit_features) * 0.85)
    situation_model.fit(sit_features[:split], sit_labels[:split])
    sit_acc = situation_model.accuracy(sit_features[split:], sit_labels[split:])
    results["situation"] = {"accuracy": sit_acc}

    # Save models
    save_dir = ARTIFACTS_DIR / "mission-models"
    save_dir.mkdir(parents=True, exist_ok=True)
    motion_model.save(str(save_dir / "motion-v3.npz"))
    situation_model.save(str(save_dir / "situation-v3.npz"))

    return results


# ============================================================
# MAIN: Run Full Training Pipeline
# ============================================================

def main() -> None:
    print("=" * 70)
    print("FSOC TRACKER — HEAVY AI/ML TRAINING PIPELINE")
    print("=" * 70)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Step 1: Generate temporal sequences (core training data)
    print("\n[1/3] Generating temporal sequences (3,000 sequences)...")
    t0 = time.time()
    seq_images, seq_targets = generate_temporal_sequences(num_sequences=3000)
    np.save(DATASET_DIR / "temporal_sequences.npy", seq_images)
    np.save(DATASET_DIR / "temporal_targets.npy", seq_targets)
    print(f"  Generated {len(seq_images)} sequences in {time.time()-t0:.1f}s")

    # Step 2: Train trajectory GRU
    print("\n[2/3] Training trajectory prediction GRU...")
    t0 = time.time()
    gru_result = train_trajectory_gru(seq_images, seq_targets, epochs=50, batch_size=64)
    results["trajectory_gru"] = gru_result
    print(f"  Trained in {time.time()-t0:.1f}s")

    # Step 3: Train control policy
    print("\n[3/3] Training control policy GRU...")
    t0 = time.time()
    ctrl_result = train_control_policy(seq_images, seq_targets, epochs=50, batch_size=64)
    results["control_policy"] = ctrl_result
    print(f"  Trained in {time.time()-t0:.1f}s")

    # Save summary
    summary_path = ARTIFACTS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
