"""Train the NumPy BeaconCNN architecture with PyTorch, export to .npz.

The runtime inference model (ai/model.py:BeaconCNN) is pure NumPy with
the exact layer layout below, so trained torch weights copy over
without transposition. Produces versioned artifacts consumable by
AIBeaconDetector.load_weights, with a parity check (torch vs NumPy
heatmaps must agree) and a detection sanity check baked in.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fsoc_tracker.ai.config import DatasetConfig
from fsoc_tracker.ai.dataset import generate_split


def build_torch_model():
    import torch.nn as nn

    class BeaconCNNTorch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 16 * 16)

        def forward(self, x):
            import torch.nn.functional as F
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.max_pool2d(F.relu(self.conv3(x)), 2)
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            return self.fc2(x).reshape(-1, 1, 16, 16)

    return BeaconCNNTorch()


def _gaussian_target(cx, cy, size=16, sigma=1.0):
    yy, xx = np.mgrid[0:size, 0:size]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def _prep_samples(split, size=128):
    """Resize images to model input, build heatmap targets."""
    import cv2
    xs, ys = [], []
    for s in split.samples:
        img = s.image
        if (img.shape[1], img.shape[0]) != (size, size):
            img = cv2.resize(img, (size, size))
        xs.append((img.astype(np.float32) / 255.0)[None, :, :])
        if s.label.visible:
            cx = s.label.center_x / s.image.shape[1] * (size // 8)
            cy = s.label.center_y / s.image.shape[0] * (size // 8)
            ys.append(_gaussian_target(cx, cy))
        else:
            ys.append(np.zeros((size // 8, size // 8), dtype=np.float32))
    return np.stack(xs), np.stack(ys)[:, None, :, :]


def train_numpy_cnn(
    num_samples: int = 2000,
    val_samples: int = 400,
    epochs: int = 15,
    batch_size: int = 32,
    seed: int = 42,
    output_dir: str = "artifacts/models/beacon-numpy-v1",
    noise_sigma_max: float = 15.0,
    salt_pepper_max: float = 0.1,
    poisson_prob: float = 0.0,
) -> dict:
    import torch
    import torch.nn as nn

    t0 = time.time()
    torch.manual_seed(seed)
    rng_cfg = DatasetConfig(noise_sigma_max=noise_sigma_max,
                            salt_pepper_max=salt_pepper_max,
                            poisson_prob=poisson_prob)
    train_split = generate_split("train", num_samples, rng_cfg, seed,
                                 include_disturbances=True)
    val_split = generate_split("val", val_samples, rng_cfg, seed + 1000,
                               include_disturbances=True)
    x_train, y_train = _prep_samples(train_split)
    x_val, y_val = _prep_samples(val_split)

    model = build_torch_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Positives cover ~1/25 of heatmap cells; without pos_weight the net
    # learns the all-zero prior (low loss, no detections).
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]))
    n = len(x_train)
    best_val, best_state, history = float("inf"), None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(x_train[idx])
            yb = torch.from_numpy(y_train[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(
                model(torch.from_numpy(x_val)),
                torch.from_numpy(y_val)))
        history.append({"epoch": ep + 1, "train_loss": tot / n,
                        "val_loss": vl})
        print(f"  epoch {ep + 1}/{epochs} train={tot / n:.4f} val={vl:.4f}",
              flush=True)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = {}
    for k, v in best_state.items():
        arr = v.numpy()
        if k == "conv1.weight":
            weights["conv1_w"] = arr
        elif k == "conv1.bias":
            weights["conv1_b"] = arr
        elif k == "conv2.weight":
            weights["conv2_w"] = arr
        elif k == "conv2.bias":
            weights["conv2_b"] = arr
        elif k == "conv3.weight":
            weights["conv3_w"] = arr
        elif k == "conv3.bias":
            weights["conv3_b"] = arr
        elif k == "fc1.weight":
            weights["fc1_w"] = arr
        elif k == "fc1.bias":
            weights["fc1_b"] = arr
        elif k == "fc2.weight":
            weights["fc2_w"] = arr
        elif k == "fc2.bias":
            weights["fc2_b"] = arr
    np.savez(out / "beacon_cnn.npz", **weights)

    # Parity: torch vs NumPy inference must agree closely. Compared on
    # the raw 16x16 logits (the upscalers differ cosmetically: scipy
    # zoom vs torch interpolate on sharp peaks).
    from fsoc_tracker.ai import model as _m
    probe = x_val[:4]
    with torch.no_grad():
        torch_16 = model(torch.from_numpy(probe)).numpy()
    max_diff = 0.0
    for img in probe[:, 0, :, :]:
        im = (img.astype(np.float32))
        if im.ndim == 2:
            im = im[None, :, :]
        x = _m._relu(_m._conv2d(im, weights["conv1_w"], weights["conv1_b"], pad=1))
        x = _m._maxpool2d(x)
        x = _m._relu(_m._conv2d(x, weights["conv2_w"], weights["conv2_b"], pad=1))
        x = _m._maxpool2d(x)
        x = _m._relu(_m._conv2d(x, weights["conv3_w"], weights["conv3_b"], pad=1))
        x = _m._maxpool2d(x)
        x = _m._relu(_m._fc(x.flatten(), weights["fc1_w"], weights["fc1_b"]))
        n16 = _m._fc(x, weights["fc2_w"], weights["fc2_b"]).reshape(1, 16, 16)
        break
    max_diff = float(np.abs(torch_16[0] - n16).max())
    # Detection sanity on synthetic beacons via the runtime detector.
    from fsoc_tracker.ai.inference import AIBeaconDetector
    det = AIBeaconDetector()
    det.load_weights(str(out / "beacon_cnn.npz"))
    hits = 0
    tried = min(40, len(x_val))
    for img in (x_val[:tried, 0, :] * 255).astype(np.uint8):
        rr = det.detect(img, 0.0, 0)
        if rr.primary_detection is not None and rr.primary_detection.detected:
            hits += 1

    receipt = {
        "model": "BeaconCNN-numpy-v1",
        "train_samples": num_samples,
        "val_samples": val_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "best_val_loss": best_val,
        "history": history,
        "parity_max_abs_diff": max_diff,
        "detection_sanity_hits": f"{hits}/{tried}",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "history"}, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BeaconCNN, export .npz")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/models/beacon-numpy-v1")
    parser.add_argument("--noise-sigma-max", type=float, default=15.0)
    parser.add_argument("--salt-max", type=float, default=0.1)
    parser.add_argument("--poisson-prob", type=float, default=0.0)
    args = parser.parse_args()
    train_numpy_cnn(num_samples=args.samples, val_samples=args.val_samples,
                    epochs=args.epochs, batch_size=args.batch_size,
                    seed=args.seed, output_dir=args.output,
                    noise_sigma_max=args.noise_sigma_max,
                    salt_pepper_max=args.salt_max,
                    poisson_prob=args.poisson_prob)


if __name__ == "__main__":
    main()
