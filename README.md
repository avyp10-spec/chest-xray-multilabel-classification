# chest-xray-multilabel-classification
Multi-label chest X-ray disease classification with PyTorch, transfer learning (ResNet-18), and Grad-CAM interpretability.
# Chest X-ray Disease Classification (Multi-Label)

A PyTorch transfer-learning project that classifies chest X-rays for 14 thoracic diseases (multi-label) using a pretrained ResNet-18, with Grad-CAM interpretability.

> **Status: work in progress — full writeup coming soon.** Core pipeline, training, evaluation, and Grad-CAM are complete and working; this README is a placeholder to be filled in.

## Overview

- **Dataset:** [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/sample) (5,606-image sample subset)
- **Task:** Multi-label classification — 14 diseases + "No Finding," predicted independently per image
- **Model:** ResNet-18 (ImageNet-pretrained), final layer replaced for 15 outputs
- **Approaches compared:** Feature extraction (frozen backbone) vs. fine-tuning (unfrozen `layer4`)
- **Interpretability:** Grad-CAM heatmaps on `layer4`

## Results (validation set, threshold = 0.4)

| Approach | Notes |
|---|---|
| Feature extraction | Stable baseline, AUC ~0.60–0.83 across classes |
| Fine-tuning (10 epochs) | Overfit — training loss → 0.02, but precision/recall collapsed to 0 on 8/15 classes |
| Fine-tuning (4 epochs) | Best result — improved AUC on most classes vs. feature extraction, better calibrated than the 10-epoch run |

Full per-class precision/recall/F1/AUC table: *TODO — paste from notebook.*

## Grad-CAM

Fine-tuning noticeably shifted model attention from image borders/non-anatomical regions toward actual lung fields on several examples. Side-by-side comparison images: *TODO — add `gradcam_example.png` vs `gradcam_finetuned_example.png`.*

## Key learnings / honest limitations

- Trained on the 5,606-image NIH sample subset, not the full 112k-image dataset (time constraint) — full dataset would likely improve rare-class performance (e.g. Hernia, Pneumonia, Fibrosis had very few examples).
- Severe class imbalance addressed via `pos_weight` in `BCEWithLogitsLoss`, capped at 20x to avoid destabilizing training.
- Patient-level train/val/test split used throughout to avoid leakage.
- Fine-tuning overfit quickly on this small dataset — caught via validation metrics, not just training loss, and fixed by reducing epochs.
- Some Grad-CAM outputs show attention to image borders/markers rather than lung tissue, especially in the feature-extraction model — a known risk with small medical imaging datasets.

## How to run

Open `chest_xray_classifier.ipynb` in Google Colab (GPU runtime recommended), add your Kaggle API credentials to Colab Secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`), and run all cells top to bottom.

## TODO (finishing tomorrow)

- [ ] Paste final metrics tables
- [ ] Add Grad-CAM comparison images
- [ ] Add screenshots
- [ ] Resume bullet points
- [ ] Clean up notebook comments/cell order
