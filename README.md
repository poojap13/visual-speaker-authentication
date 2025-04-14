
# Enhancing Visual Speaker Authentication Using Dynamic Lip Movement Analysis and Meta-Learning

## Overview

This project presents a visual speaker authentication (VSA) system that leverages dynamic lip movement features extracted through optical flow and a meta-learning-based few-shot learning framework to improve spoof resistance and reduce the amount of enrollment data required for new users. The goal is to enhance the security and practicality of speaker authentication systems, especially against DeepFake attacks.

## Research Questions

**RQ1:** Can few-shot learning approaches, specifically Model-Agnostic Meta-Learning (MAML), significantly reduce the user registration data requirements in visual speaker authentication systems compared to traditional CNN methods?

**RQ2:** Can dynamically extracted visual features (optical flow of lip movements) effectively distinguish genuine speakers from sophisticated DeepFake spoofing attacks?

## Methodology

- **Dataset:** GRID corpus (downloaded from https://spandh.dcs.shef.ac.uk/gridcorpus/), consisting of audiovisual speech recordings from 34 speakers.
- **Fake Video Generation:** Wav2Lip is used to create realistic DeepFake videos by altering the lip movements of the original speaker based on unrelated audio input.
- **Preprocessing:** Lip regions are extracted using Dlib landmarks, and dense optical flow is applied using Farnebäck’s method to capture dynamic lip movement between video frames.
- **Feature Representation:** Optical flow tensors of shape (75, 64, 64, 2) per video are generated to preserve spatial and temporal motion.
- **Models:**
  - **Baseline CNN:** Trained on real/fake videos for seen speakers using standard cross-entropy.
  - **MAML-based 3D CNN:** Trained with few-shot tasks to enable rapid adaptation to new speakers with limited samples.
- **Speaker Splits:**
  - Training Speakers: s1–s22
  - Validation Speakers: s23–s25
  - Test Speakers: s26–s30

## Project Structure

- **notebooks/** – Jupyter notebooks for each major step (fake video generation, optical flow extraction, CNN and MAML training).
- **src/** – Modular Python code:
  - `fake_video_generator.py` – Wrapper for Wav2Lip
  - `optical_flow.py` – Optical flow extraction functions
  - `cnn_model.py`, `maml_model.py` – Model definitions
  - `train_utils.py` – Training and evaluation functions
- **data/** – Instructions for downloading and organizing the GRID dataset
- **results/** – Plots, metrics, confusion matrices
- **saved_models/** – Final trained models (.pth files under 100MB)
- **README.md** – This project documentation
- **requirements.txt** – Python package dependencies

## How to Run

1. **Download GRID Dataset**  
   Open `notebooks/1_generate_fakes.ipynb` or `data/download_grid.ipynb`.  
   Enter speaker range (e.g., 1–30) and whether to extract files.

2. **Generate DeepFake Videos**  
   Use Wav2Lip via the provided script to create fake videos with mismatched audio.

3. **Extract Optical Flow**  
   Run the notebook `2_optical_flow_extraction.ipynb` or use the function from `src/optical_flow.py`.

4. **Train CNN Baseline**  
   Run `3_cnn_baseline_training.ipynb` to train and evaluate the baseline model.

5. **Train MAML Model**  
   Run `4_maml_training.ipynb`. The model is trained using few-shot tasks and evaluated on disjoint speakers.

## Results Summary

| Model                 | AUC   | HTER   | Accuracy (Disjoint) |
|----------------------|-------|--------|----------------------|
| CNN (Seen Speakers)  | 0.9997| 0.23%  | 99.7%               |
| CNN (Disjoint)       | 0.500 | 50.0%  | 50.0%               |
| MAML (Validation)    | 0.990+| <1%    | 98.9%               |
| MAML (Disjoint)      | 0.920 | 16.5%  | 68.6%               |

## Hyperparameters (Optuna Tuned)

- Dropout: 0.44
- Inner loop learning rate (α): 0.015
- Outer loop learning rate (β): 0.00098
- Number of shots: 7
- Epochs: 30
- Tasks per episode: 100

## Reproducibility Notes (Checklist Highlights)

- Dataset preprocessing, splits, and feature extraction are fully described.
- Source code includes training and evaluation scripts for both CNN and MAML models.
- Optuna was used for hyperparameter optimization.
- Evaluation metrics include Accuracy, AUC, Precision, Recall, F1-score, FAR, FRR, and HTER.
- Models are tested on unseen speakers (s26–s30) to validate generalization.
- All results are backed by saved visualizations and metrics in the `results/` directory.
- Code is modularized and reusable, with clear function definitions and parameters.

## Links

- GitHub Repository: https://github.com/yourusername/visual-speaker-authentication  
- Overleaf Report: https://www.overleaf.com/read/your-overleaf-link-here  

## Author

Pooja Pathare  
MSc in Computer Science, Lakehead University  
Supervised by Dr. Garima Bajwa

## License

This project is for academic, non-commercial research use only. Please cite appropriately if reused or referenced.

---
