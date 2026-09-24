"""FSOC Disturbance-Aware AI Training Pipeline.

Trains models to handle ALL atmospheric disturbance types:
- Clear sky (baseline)
- Fog (reduced visibility, scattering)
- Rain (droplet scattering, attenuation)
- Turbulence (beam wander, scintillation)
- Dust (particulate scattering)
- Smoke (absorption, scattering)
- Snow (large particle scattering)

Models trained:
1. DisturbanceClassifier — identifies disturbance type from sensor data
2. AdaptiveBeamController — compensates for disturbance effects
3. AtmosphericPredictor — predicts atmospheric state evolution
"""

from __future__ import annotations

import random
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# DATA GENERATION
# ============================================================

class DisturbanceType:
    CLEAR = 0
    FOG = 1
    RAIN = 2
    TURBULENCE = 3
    DUST = 4
    SMOKE = 5
    SNOW = 6

N_TYPES = 7

def generate_disturbance_features(disturbance_type: int, rng: random.Random) -> list[float]:
    """Generate realistic sensor features for a disturbance type.
    
    Features:
    [0] brightness_mean — mean image brightness (0-255)
    [1] brightness_std — brightness variance
    [2] contrast — edge strength measure
    [3] scintillation_index — intensity fluctuation
    [4] beam_wander_px — beam displacement
    [5] attenuation_db — signal loss in dB
    [6] visibility_km — meteorological visibility
    [7] cn2_turbulence — turbulence strength (log scale)
    [8] wind_speed_ms — wind speed
    [9] humidity_percent — relative humidity
    [10] temperature_c — ambient temperature
    [11] particle_density — particle concentration
    """
    if disturbance_type == DisturbanceType.CLEAR:
        return [
            rng.uniform(180, 220),  # brightness_mean
            rng.uniform(15, 30),    # brightness_std
            rng.uniform(40, 60),    # contrast
            rng.uniform(0.0, 0.1),  # scintillation_index
            rng.uniform(0.0, 0.5),  # beam_wander_px
            rng.uniform(0.0, 1.0),  # attenuation_db
            rng.uniform(8.0, 15.0), # visibility_km
            rng.uniform(-16, -14),  # cn2_turbulence (log10)
            rng.uniform(0.0, 5.0),  # wind_speed_ms
            rng.uniform(40, 60),    # humidity_percent
            rng.uniform(15, 30),    # temperature_c
            rng.uniform(0.0, 10.0), # particle_density
        ]
    elif disturbance_type == DisturbanceType.FOG:
        return [
            rng.uniform(20, 80),    # brightness_mean
            rng.uniform(5, 15),     # brightness_std
            rng.uniform(5, 20),     # contrast
            rng.uniform(0.0, 0.3),  # scintillation_index
            rng.uniform(0.5, 3.0),  # beam_wander_px
            rng.uniform(5.0, 30.0), # attenuation_db
            rng.uniform(0.1, 2.0),  # visibility_km
            rng.uniform(-15, -13),  # cn2_turbulence
            rng.uniform(0.0, 3.0),  # wind_speed_ms
            rng.uniform(90, 100),   # humidity_percent
            rng.uniform(5, 15),     # temperature_c
            rng.uniform(50, 200),   # particle_density
        ]
    elif disturbance_type == DisturbanceType.RAIN:
        return [
            rng.uniform(40, 120),   # brightness_mean
            rng.uniform(20, 50),    # brightness_std
            rng.uniform(10, 30),    # contrast
            rng.uniform(0.1, 0.5),  # scintillation_index
            rng.uniform(1.0, 5.0),  # beam_wander_px
            rng.uniform(3.0, 20.0), # attenuation_db
            rng.uniform(1.0, 5.0),  # visibility_km
            rng.uniform(-14, -12),  # cn2_turbulence
            rng.uniform(5.0, 20.0), # wind_speed_ms
            rng.uniform(70, 100),   # humidity_percent
            rng.uniform(5, 20),     # temperature_c
            rng.uniform(100, 500),  # particle_density
        ]
    elif disturbance_type == DisturbanceType.TURBULENCE:
        return [
            rng.uniform(150, 200),  # brightness_mean
            rng.uniform(30, 80),    # brightness_std (high variance!)
            rng.uniform(20, 50),    # contrast
            rng.uniform(0.3, 1.0),  # scintillation_index (high!)
            rng.uniform(2.0, 10.0), # beam_wander_px (high!)
            rng.uniform(1.0, 10.0), # attenuation_db
            rng.uniform(5.0, 10.0), # visibility_km
            rng.uniform(-13, -11),  # cn2_turbulence (strong!)
            rng.uniform(10.0, 30.0),# wind_speed_ms
            rng.uniform(30, 60),    # humidity_percent
            rng.uniform(10, 35),    # temperature_c
            rng.uniform(0.0, 50.0), # particle_density
        ]
    elif disturbance_type == DisturbanceType.DUST:
        return [
            rng.uniform(80, 150),   # brightness_mean
            rng.uniform(15, 40),    # brightness_std
            rng.uniform(10, 25),    # contrast
            rng.uniform(0.1, 0.4),  # scintillation_index
            rng.uniform(1.0, 4.0),  # beam_wander_px
            rng.uniform(4.0, 15.0), # attenuation_db
            rng.uniform(1.0, 4.0),  # visibility_km
            rng.uniform(-14, -12),  # cn2_turbulence
            rng.uniform(5.0, 25.0), # wind_speed_ms
            rng.uniform(20, 50),    # humidity_percent
            rng.uniform(25, 45),    # temperature_c
            rng.uniform(200, 800),  # particle_density
        ]
    elif disturbance_type == DisturbanceType.SMOKE:
        return [
            rng.uniform(30, 100),   # brightness_mean
            rng.uniform(10, 30),    # brightness_std
            rng.uniform(5, 15),     # contrast
            rng.uniform(0.0, 0.2),  # scintillation_index
            rng.uniform(0.5, 2.0),  # beam_wander_px
            rng.uniform(5.0, 25.0), # attenuation_db
            rng.uniform(0.5, 3.0),  # visibility_km
            rng.uniform(-15, -13),  # cn2_turbulence
            rng.uniform(1.0, 8.0),  # wind_speed_ms
            rng.uniform(40, 70),    # humidity_percent
            rng.uniform(20, 40),    # temperature_c
            rng.uniform(100, 600),  # particle_density
        ]
    elif disturbance_type == DisturbanceType.SNOW:
        return [
            rng.uniform(50, 150),   # brightness_mean
            rng.uniform(20, 60),    # brightness_std
            rng.uniform(8, 25),     # contrast
            rng.uniform(0.2, 0.6),  # scintillation_index
            rng.uniform(1.5, 6.0),  # beam_wander_px
            rng.uniform(3.0, 18.0), # attenuation_db
            rng.uniform(0.5, 3.0),  # visibility_km
            rng.uniform(-14, -12),  # cn2_turbulence
            rng.uniform(3.0, 15.0), # wind_speed_ms
            rng.uniform(60, 90),    # humidity_percent
            rng.uniform(-5, 5),     # temperature_c
            rng.uniform(50, 300),   # particle_density
        ]
    else:
        return [0.0] * 12


def generate_compensation_params(disturbance_type: int, rng: random.Random) -> list[float]:
    """Generate optimal compensation parameters for a disturbance type.
    
    Output:
    [0] gain_boost — controller gain multiplier
    [1] damping_factor — additional damping
    [2] filter_cutoff_hz — low-pass filter cutoff
    [3] prediction_horizon_s — how far ahead to predict
    [4] integration_limit — anti-windup limit
    [5] feedforward_gain — feedforward control gain
    """
    if disturbance_type == DisturbanceType.CLEAR:
        return [1.0, 1.0, 10.0, 0.1, 1.0, 0.0]
    elif disturbance_type == DisturbanceType.FOG:
        return [1.3, 0.9, 8.0, 0.15, 1.2, 0.1]
    elif disturbance_type == DisturbanceType.RAIN:
        return [1.2, 0.8, 7.0, 0.12, 1.1, 0.15]
    elif disturbance_type == DisturbanceType.TURBULENCE:
        return [1.5, 0.6, 12.0, 0.08, 0.8, 0.3]
    elif disturbance_type == DisturbanceType.DUST:
        return [1.2, 0.85, 8.0, 0.12, 1.0, 0.1]
    elif disturbance_type == DisturbanceType.SMOKE:
        return [1.3, 0.9, 7.0, 0.15, 1.1, 0.1]
    elif disturbance_type == DisturbanceType.SNOW:
        return [1.4, 0.7, 6.0, 0.18, 1.2, 0.2]
    else:
        return [1.0, 1.0, 10.0, 0.1, 1.0, 0.0]


def generate_temporal_sequence(disturbance_type: int, seq_len: int, rng: random.Random) -> tuple[list[list[float]], list[float]]:
    """Generate a temporal sequence of disturbance features with time evolution."""
    features = []
    base = generate_disturbance_features(disturbance_type, rng)
    
    for t in range(seq_len):
        # Add temporal evolution
        noise = [rng.gauss(0, 0.02) for _ in range(len(base))]
        evolved = [base[i] + noise[i] * (t / seq_len) for i in range(len(base))]
        features.append(evolved)
    
    comp = generate_compensation_params(disturbance_type, rng)
    return features, comp


# ============================================================
# PYTORCH MODELS
# ============================================================

if HAS_TORCH:
    class DisturbanceClassifier(nn.Module):
        """Classifies atmospheric disturbance type from sensor features."""

        def __init__(self, input_dim: int = 12, hidden_dim: int = 64, n_classes: int = N_TYPES):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    class AdaptiveBeamController(nn.Module):
        """Learns optimal compensation parameters for each disturbance type.
        
        Input: 12 sensor features + 7 disturbance type probabilities
        Output: 6 compensation parameters
        """

        def __init__(self, input_dim: int = 12 + N_TYPES, hidden_dim: int = 64, output_dim: int = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim),
                nn.Softplus(),  # All compensation params must be positive
            )

        def forward(self, x):
            return self.net(x)

    class AtmosphericPredictor(nn.Module):
        """GRU-based model that predicts atmospheric state evolution.
        
        Input: sequence of sensor features
        Output: predicted features at next timestep
        """

        def __init__(self, input_dim: int = 12, hidden_dim: int = 64, num_layers: int = 2):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            gru_out, _ = self.gru(x)
            return self.fc(gru_out)


class DisturbanceDataset(Dataset):
    """Dataset for disturbance classification and compensation."""

    def __init__(self, n_samples: int = 5000, seq_len: int = 10, rng_seed: int = 42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = random.Random(rng_seed)

        self.features = []
        self.labels = []
        self.compensations = []
        self.sequences = []

        for _ in range(n_samples):
            dtype = self.rng.randint(0, N_TYPES - 1)
            feat = generate_disturbance_features(dtype, self.rng)
            comp = generate_compensation_params(dtype, self.rng)
            seq, _ = generate_temporal_sequence(dtype, seq_len, self.rng)

            self.features.append(feat)
            self.labels.append(dtype)
            self.compensations.append(comp)
            self.sequences.append(seq)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.compensations[idx], dtype=torch.float32),
            torch.tensor(self.sequences[idx], dtype=torch.float32),
        )


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_disturbance_models(output_dir: str = "artifacts/models/disturbance-v1", n_samples: int = 5000, epochs: int = 50):
    """Train all disturbance-aware models."""
    if not HAS_TORCH:
        print("PyTorch not available — skipping training")
        return

    print("=" * 70)
    print("FSOC DISTURBANCE-AWARE AI TRAINING")
    print("=" * 70)
    print(f"Samples: {n_samples} | Epochs: {epochs} | Disturbance types: {N_TYPES}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    print("\n[1/4] Generating disturbance dataset...")
    t0 = time.time()
    dataset = DisturbanceDataset(n_samples=n_samples, seq_len=10, rng_seed=42)
    print(f"  Generated {n_samples} samples in {time.time()-t0:.1f}s")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # ---- Model 1: Disturbance Classifier ----
    print("\n[2/4] Training Disturbance Classifier...")
    classifier = DisturbanceClassifier()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        classifier.train()
        train_loss = 0.0
        for features, labels, _, _ in train_loader:
            optimizer.zero_grad()
            output = classifier(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels, _, _ in val_loader:
                output = classifier(features)
                val_loss += criterion(output, labels).item()
                pred = output.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        val_loss /= max(len(val_loader), 1)
        accuracy = correct / max(total, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), out / "disturbance_classifier.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss/len(train_loader):.4f} val_loss={val_loss:.4f} accuracy={accuracy:.3f}")

    print(f"  Best classifier val_loss: {best_val_loss:.4f}")

    # ---- Model 2: Adaptive Beam Controller ----
    print("\n[3/4] Training Adaptive Beam Controller...")
    controller = AdaptiveBeamController()
    optimizer = optim.Adam(controller.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        controller.train()
        train_loss = 0.0
        for features, labels, compensations, _ in train_loader:
            # Concatenate features with one-hot disturbance type
            type_onehot = torch.zeros(labels.size(0), N_TYPES)
            type_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
            inp = torch.cat([features, type_onehot], dim=1)

            optimizer.zero_grad()
            output = controller(inp)
            loss = criterion(output, compensations)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        controller.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, compensations, _ in val_loader:
                type_onehot = torch.zeros(labels.size(0), N_TYPES)
                type_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
                inp = torch.cat([features, type_onehot], dim=1)
                output = controller(inp)
                val_loss += criterion(output, compensations).item()

        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(controller.state_dict(), out / "adaptive_beam_controller.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss/len(train_loader):.4f} val_loss={val_loss:.4f}")

    print(f"  Best controller val_loss: {best_val_loss:.4f}")

    # ---- Model 3: Atmospheric Predictor (GRU) ----
    print("\n[4/4] Training Atmospheric Predictor (GRU)...")
    predictor = AtmosphericPredictor()
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        predictor.train()
        train_loss = 0.0
        for _, _, _, sequences in train_loader:
            # Input: all but last timestep, Target: all but first timestep
            inp = sequences[:, :-1, :]
            target = sequences[:, 1:, :]

            optimizer.zero_grad()
            output = predictor(inp)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, _, _, sequences in val_loader:
                inp = sequences[:, :-1, :]
                target = sequences[:, 1:, :]
                output = predictor(inp)
                val_loss += criterion(output, target).item()

        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(predictor.state_dict(), out / "atmospheric_predictor.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss/len(train_loader):.6f} val_loss={val_loss:.6f}")

    print(f"  Best predictor val_loss: {best_val_loss:.6f}")

    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "n_types": N_TYPES,
        "type_names": ["clear", "fog", "rain", "turbulence", "dust", "smoke", "snow"],
        "feature_dim": 12,
        "compensation_dim": 6,
        "sequence_length": 10,
        "models": [
            "disturbance_classifier.pt",
            "adaptive_beam_controller.pt",
            "atmospheric_predictor.pt",
        ],
    }
    import json
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — Models saved to {output_dir}")
    print(f"{'='*70}")

    # Verify models
    print("\nVerification:")
    classifier.eval()
    test_feat = torch.tensor(generate_disturbance_features(DisturbanceType.FOG, random.Random(99)), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = classifier(test_feat)
        pred_type = pred.argmax(dim=1).item()
        conf = torch.softmax(pred, dim=1).max().item()
    print(f"  Fog input classified as: {['CLEAR','FOG','RAIN','TURBULENCE','DUST','SMOKE','SNOW'][pred_type]} (confidence: {conf:.3f})")

    return {
        "classifier": str(out / "disturbance_classifier.pt"),
        "controller": str(out / "adaptive_beam_controller.pt"),
        "predictor": str(out / "atmospheric_predictor.pt"),
    }


if __name__ == "__main__":
    train_disturbance_models()
