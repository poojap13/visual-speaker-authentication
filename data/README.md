# GRID Corpus Setup & Preprocessing Guide

This folder contains the instructions and code to download and extract the GRID audio-visual speech corpus, used as the foundation for all experiments in this Visual Speaker Authentication (VSA) project.

---

## 📦 Dataset Used: GRID Corpus

- 🔗 [GRID Official Site](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- 🗣️ 34 speakers, each with 1000+ utterances
- 🎥 Audio and video pairs with fixed sentence structure

---

## 🧠 What This Script Does

The `download_grid.ipynb` notebook:
- Downloads audio (`.tar`) and video (`.zip`) files for speakers `s1` to `s30`
- Organizes them into a structured directory tree
- Optionally extracts files automatically after download

It uses Python's `os`, `zipfile`, `tarfile`, and `curl` via `subprocess`.

---

## 📂 Folder Structure Created

