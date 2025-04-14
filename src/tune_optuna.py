import optuna
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import learn2learn as l2l

# ====== CONFIGURATION ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optical_flow_path = "E:/visual_speaker_auth/data/gridcorpus/optical_flow"
train_speakers = [f"s{i}" for i in range(1, 21)]
val_speakers = [f"s{i}" for i in range(21, 26)]
test_speakers = [f"s{i}" for i in range(26, 31)]
MAX_FRAMES = 30
EPOCHS = 10
TASKS_PER_EPOCH = 50
VAL_TASKS = 20

# ====== MODEL DEFINITION ======
class OpticalFlowModel(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

# ====== TASK GENERATOR ======
class TaskGenerator:
    def __init__(self, base_path, speakers, shots):
        self.base_path = base_path
        self.speakers = speakers
        self.shots = shots

    def _load(self, path):
        try:
            flow = np.load(path, allow_pickle=True)
            flow = torch.from_numpy(flow).float().permute(3, 0, 1, 2)
            flow = flow[:, :MAX_FRAMES] if flow.size(1) >= MAX_FRAMES else F.pad(flow, (0, 0, 0, 0, 0, MAX_FRAMES - flow.size(1)))
            return flow + 0.01 * torch.randn_like(flow)
        except Exception as e:
            print(f"⚠️ Skipping file {path}: {e}")
            return None

    def create_task(self):
        for _ in range(10):
            s1, s2 = random.sample(self.speakers, 2)

            def sample_paths(speaker, label):
                path = os.path.join(self.base_path, label, speaker)
                if not os.path.exists(path): return []
                files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')]
                return random.sample(files, min(len(files), self.shots))

            sr, sf = sample_paths(s1, 'real'), sample_paths(s1, 'fake')
            qr, qf = sample_paths(s2, 'real'), sample_paths(s2, 'fake')

            if min(len(sr), len(sf), len(qr), len(qf)) < self.shots:
                continue

            s_data = [(p, 0) for p in sr] + [(p, 1) for p in sf]
            q_data = [(p, 0) for p in qr] + [(p, 1) for p in qf]

            try:
                s_x = torch.stack([self._load(p) for p, _ in s_data if self._load(p) is not None]).to(device)
                s_y = torch.tensor([lbl for p, lbl in s_data if self._load(p) is not None]).to(device)
                q_x = torch.stack([self._load(p) for p, _ in q_data if self._load(p) is not None]).to(device)
                q_y = torch.tensor([lbl for p, lbl in q_data if self._load(p) is not None]).to(device)
                return s_x, s_y, q_x, q_y
            except:
                continue
        return None, None, None, None

# ====== OPTUNA OBJECTIVE FUNCTION ======
def objective(trial):
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    inner_lr = trial.suggest_float("inner_lr", 0.001, 0.03)
    meta_lr = trial.suggest_float("meta_lr", 1e-5, 1e-3)
    shots = trial.suggest_int("shots", 3, 8)

    model = OpticalFlowModel(dropout).to(device)
    maml = l2l.algorithms.MAML(model, lr=inner_lr)
    optimizer = optim.Adam(maml.parameters(), lr=meta_lr)

    train_gen = TaskGenerator(optical_flow_path, train_speakers, shots)
    val_gen = TaskGenerator(optical_flow_path, val_speakers, shots)

    # Train for a few epochs
    for _ in range(EPOCHS):
        for _ in range(TASKS_PER_EPOCH):
            s_x, s_y, q_x, q_y = train_gen.create_task()
            if s_x is None: continue
            learner = maml.clone()
            learner.adapt(F.cross_entropy(learner(s_x), s_y))
            loss = F.cross_entropy(learner(q_x), q_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation accuracy
    val_accs = []
    for _ in range(VAL_TASKS):
        s_x, s_y, q_x, q_y = val_gen.create_task()
        if s_x is None: continue
        learner = maml.clone()
        learner.adapt(F.cross_entropy(learner(s_x), s_y))
        preds = learner(q_x).argmax(dim=1)
        acc = (preds == q_y).float().mean().item()
        val_accs.append(acc)

    return np.mean(val_accs)

# ====== RUN STUDY ======
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_trial = study.best_trial
    print("\n✅ BEST HYPERPARAMETERS:")
    for key, val in best_trial.params.items():
        print(f"{key}: {val}")
