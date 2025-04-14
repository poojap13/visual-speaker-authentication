##  `README.md` — *Enhancing Visual Speaker Authentication Using Dynamic Lip Movement and Meta-Learning*  
🗓️ Last Updated: 2025-04-14


---

### 🎯 Project Overview

This project proposes a **Visual Speaker Authentication (VSA)** system that leverages **dynamic lip movements** for secure and spoof-resistant identity verification. It addresses challenges in conventional speaker authentication systems such as:
- High registration effort
- Susceptibility to DeepFake attacks
- Poor generalization to unseen users

We adopt a **few-shot meta-learning approach (MAML)** using optical flow inputs, enabling:
- **Fast adaptation to new users with minimal samples**
- **Strong spoof detection against AI-generated fakes**
- **Evaluation on speaker-disjoint splits**

---

### ❓ Research Questions

**RQ1:** Can few-shot learning (MAML) reduce speaker registration requirements compared to traditional CNN-based models?

**RQ2:** Are dynamically extracted visual features (optical flow of lip movements) effective for detecting DeepFake spoofing?

---

### 🧪 Dataset

- 📦 **GRID Corpus**: 34 speakers, frontal face videos with matching audio
- 🎭 **Fake Generation**: Created using [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), mixing real faces with mismatched audio to produce DeepFakes
- 🌀 **Optical Flow Extraction**: Lip regions are isolated using Dlib and processed with Farneback's method to capture temporal motion

---

### 🗃️ Project Structure

```
visual-speaker-authentication/
├── data/                       # Dataset instructions
├── notebooks/                 # Jupyter notebooks (preprocessing, training, eval)
├── results/                   # Metrics, confusion matrix, ROC curves
├── saved_models/              # Trained model checkpoints (.pth)
├── src/                       # Modular Python scripts
├── main.py                    # End-to-end pipeline script
├── requirements.txt           # Dependencies
├── .gitignore                 # Ignore configs, models, cache
└── README.md                  # This documentation
```

---

### 📔 Notebooks

| File                                       | Purpose                                                                 |
|--------------------------------------------|-------------------------------------------------------------------------|
| `Fakedataset_generation.ipynb`             | Generate DeepFake videos using Wav2Lip                                 |
| `OpticalFlowFakeSpeakers.ipynb`            | Extract lip optical flow features                                      |
| `VSA_FINALMAML.ipynb`                      | Full MAML training and testing                                         |
| `VSA_hyperparametertunning&20tasks.ipynb`  | Optuna tuning + reduced-task generalization setup                     |

---

### 🧠 Core Scripts (in `src/`)

| Script                 | Description                                                             |
|------------------------|-------------------------------------------------------------------------|
| `generate_fakes.py`    | Batch generates fake videos using Wav2Lip for all speakers              |
| `extract_optical_flow.py` | Extracts lip optical flow from real/fake videos (saves `.npy` files) |
| `train_maml.py`        | Final MAML model training with Optuna-best hyperparameters              |
| `tune_optuna.py`       | Hyperparameter tuning (dropout, inner_lr, meta_lr, shots) using Optuna |

---

### 🔧 How to Run

#### 📦 1. Clone and Install
```bash
git clone https://github.com/poojap13/visual-speaker-authentication.git
cd visual-speaker-authentication
pip install -r requirements.txt
```

#### 📥 2. Download GRID Dataset
```bash
cd data
# See: DownloadGridDataset.ipynb or README.md for script
```

#### 🧪 3. Generate Fake Videos
```bash
python src/generate_fakes.py
```

#### 🎞️ 4. Extract Optical Flow
```bash
python src/extract_optical_flow.py
```

#### 🧠 5. Tune Hyperparameters (Optional)
```bash
python src/tune_optuna.py
```

#### 🧬 6. Train Final MAML Model
```bash
python src/train_maml.py
```

#### ✅ 7. Full Pipeline
```bash
python main.py
```

---

### 🧾 Results Summary (on speaker-disjoint setup)

| Model           | Accuracy | AUC   | EER   | HTER  | F1    |
|----------------|----------|-------|-------|-------|-------|
| MAML (Final)   | 99.67%   | 0.997 | 0.000 | 0.003 | 0.996 |

📍 See: `results/` folder for confusion matrix, ROC curves, and logs

---

### 🧬 Final Hyperparameters (Tuned via Optuna)

```yaml
dropout:       0.3766
inner_lr:      0.0204
meta_lr:       0.0006
shots:         3
epochs:        15
tasks/epoch:   50
```

---

### 📌 Reproducibility Checklist (v2.0 ✅)

- ✅ Dataset statistics, splits, preprocessing steps
- ✅ Hyperparameter tuning details (via Optuna)
- ✅ All training & evaluation code provided
- ✅ Pre-trained model: `saved_models/best_model3.pth`
- ✅ Confusion matrix, ROC, metrics (`results/`)
- ✅ Training logs and curves included

---

### 🔗 Links

- 📄 Overleaf Report: [Overleaf Final Report](https://www.overleaf.com/project/67e76f407a248c43d6fd131d)
- 📦 Wav2Lip GitHub: [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- 🗃 GRID Corpus: [https://spandh.dcs.shef.ac.uk/gridcorpus/](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- 🔗 GitHub Repo: [https://github.com/poojap13/visual-speaker-authentication](https://github.com/poojap13/visual-speaker-authentication)

---

### 👩‍💻 Author

**Pooja Pathare**  
MSc in Computer Science, Lakehead University  
Supervisor: Dr. Garima Bajwa

---

### 📜 License

This project is intended for **academic research and non-commercial use only**.  
Please cite the original repositories (Wav2Lip, Learn2Learn, GRID Corpus) if used.

