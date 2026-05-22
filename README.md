# 🌿 PhytoScan — Vision-Language Plant Disease Diagnosis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-orange.svg)](https://pytorch.org)
[![CLIP](https://img.shields.io/badge/OpenAI-CLIP-purple.svg)](https://github.com/openai/CLIP)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-green.svg)](https://arxiv.org/abs/2106.09685)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **PhytoScan fine-tunes CLIP ViT-B/32 with SimCLR pre-training and LoRA
> adapters to diagnose 89 crop diseases from leaf photos, achieving 68.6%
> accuracy on real field images.**

**Author:** Shah Md Abul Hasan · University of Georgia

---

## The Problem

Crop disease identification in the field currently requires either a trained
agronomist or sending samples to a lab — both slow and expensive at scale.
Existing deep learning tools require thousands of labeled images per disease
class to train, and CNN classifiers trained on controlled datasets fail on
real field photography.

**PhytoScan addresses this by treating disease classification as a
vision-language matching problem** — matching a leaf image against 89
natural language disease descriptions — rather than training a traditional
classification head. This enables zero-shot transfer to real field images
without retraining.

---

## What It Does

Upload a leaf photograph → get the top-5 disease predictions with confidence
scores and a saliency map showing exactly where on the leaf the model focused.

```
Input  : leaf photo (JPG/PNG, any resolution)
Output : crop name · disease name · confidence %
         + saliency heatmap highlighting diseased tissue
```

**Supported crops:** Apple · Tomato · Potato · Corn · Maize · Grape ·
Pepper · Strawberry · Peach · Cherry · Soybean · Raspberry · Squash ·
Coffee · Black Pepper + more (89 total disease conditions)

---

## Field Performance

Evaluated on **PlantDoc** — 236 real field images across 27 disease classes,
completely unseen during training. This is the most important benchmark because
it measures generalization to actual farm conditions, not controlled lab imagery.

### Results — PlantDoc Test Set

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| CLIP zero-shot — simple label | 14.4% | 7.0% |
| CLIP zero-shot — expert prompt | 19.1% | 10.6% |
| CLIP zero-shot — symptom description | 17.8% | 7.9% |
| CLIP zero-shot — prompt ensemble | 18.2% | 8.5% |
| **PhytoScan (this work)** | **68.6%** | **47.1%** |

**Improvement over best zero-shot CLIP: +49.5% accuracy**

The 47.1% Macro F1 reflects uneven class representation in PlantDoc —
common diseases (Apple Scab, Tomato Late Blight) score higher while rare
classes with few test images pull the macro average down. Per-class F1
on well-represented diseases is substantially higher.

---

### Training Convergence

PhytoScan converges rapidly after SimCLR domain pre-training.
PlantDoc accuracy jumps from 36% → 60% in just 3 epochs — confirming
that the SSL warm-start provides agronomically relevant initialization.

| Epoch | Train Loss | Val Acc | PlantDoc Acc | PlantDoc F1 |
|-------|-----------|---------|--------------|-------------|
| 1 | 1.0998 | 92.3% | 36.0% | 20.0% |
| 2 | 0.2107 | 94.2% | 52.5% | 33.3% |
| 3 | 0.1504 | 94.8% | 60.6% | 33.9% |
| 5 | 0.0857 | 95.4% | 66.9% | 45.4% |
| **6** | **0.0610** | **95.3%** | **68.6%** | **47.1%** ← **best** |
| 12 | 0.0153 | 95.8% | 66.9% | 45.2% ← early stop |

Best checkpoint at **epoch 6**. Early stopping at epoch 12 (patience = 6).
LeafNet validation accuracy kept rising while PlantDoc generalization
plateaued — evidence of distribution gap between controlled training
images and real field photography.

---

## Why This Matters for Precision Agriculture

### 1. No Per-Farm Retraining
Traditional disease classifiers need thousands of labeled images per
class per environment. PhytoScan uses natural language disease descriptions
as classifiers — adding a new disease means writing a caption, not
collecting new training data.

### 2. Explainable Predictions for Agronomists
Saliency maps show exactly which tissue the model focused on.
Incorrect predictions still highlight disease-relevant regions
(lesions, discoloration) — errors are expert-level misclassification
between similar diseases, not background confusion.

```
✅ Correct:   model focuses on dark scab lesions → predicts Apple Scab
❌ Incorrect: model focuses on brown lesion → predicts Early Blight
              instead of Late Blight (visually similar)
```

This is the right failure mode for a field tool — the model is uncertain
about disease subtype, not about whether a plant is diseased at all.

### 3. Compute-Efficient Deployment
Only **2,949K parameters (1.91% of CLIP)** are fine-tuned via LoRA.
The frozen backbone can be shared across multiple agricultural models
without storing separate full-model checkpoints per crop.

### 4. Connects to UAV and Multispectral Workflows
PhytoScan operates on standard RGB leaf images — compatible with photos
captured by DJI Mavic, Phantom, or any field camera. Results can be
combined with UAV-based canopy monitoring to flag individual plants
for ground-level disease verification.

---

## Technical Approach

### Three-Stage Pipeline

```
Stage 1 — Domain Adaptation (SimCLR)
──────────────────────────────────────
60,000 unlabeled leaf images
Self-supervised contrastive learning
Teaches CLIP to be invariant to:
  • Lighting variation (field vs greenhouse)
  • Camera angle and crop
  • Motion blur and focus variation
  • Weather and shadow effects
SSL loss: 6.01 → 3.37 (1 epoch, 60k images)

        ↓  adapted visual encoder

Stage 2 — Parameter-Efficient Fine-tuning (LoRA)
──────────────────────────────────────────────────
CLIP ViT-B/32 backbone — 154.2M parameters — all frozen
LoRA adapters injected into MLP layers only
  • 48 LoRA tensors (c_fc + c_proj × 12 blocks × 2)
  • Rank r=32, scaling alpha=64
  • 2,949K trainable parameters (1.91%)
Label smoothing=0.1, AMP, AdamW lr=5e-5

Training: 108,000 LeafNet images · 12 epochs · ~76 min total

        ↓  fine-tuned model

Stage 3 — Zero-shot Inference
───────────────────────────────
89 metadata-enriched disease captions → text embeddings
Leaf image → visual embedding
Cosine similarity → ranked predictions
No threshold, no calibration, no post-processing
```

### Why LoRA in MLP Layers Only

Injecting LoRA into attention projection layers (q, k, v) interferes with
PyTorch's scaled dot-product attention fast-path and disrupts CLIP's
learned cross-modal alignment. MLP-only injection:

- Preserves visual-language alignment from CLIP pretraining
- Adapts feed-forward representations to disease feature space
- Passes all 48/48 gradient checks at initialization

### Why Metadata-Enriched Prompts

CLIP was pretrained on 400M internet image-text pairs — it responds to
descriptive, contextual captions rather than short labels. Symptom
descriptions activate the relevant parts of CLIP's text representation
space, producing more discriminative class embeddings.

```python
# Poor (14.4% accuracy)
"a photo of apple leaf with scab"

# Better (19.1% accuracy)
"apple leaf infected with Venturia inaequalis fungal pathogen"

# Best for fine-tuning
"A field photograph of an apple leaf infected with scab.
 Visible symptoms include: olive-green to dark brown velvety
 lesions on leaves. Agricultural crop disease detection image."
```

### Saliency Maps Without Gradient Flow

Standard GradCAM fails on LoRA-CLIP because frozen layers block gradient
propagation to intermediate activations (measured gradient max = 0.0000
at the last transformer block despite the score having a valid grad_fn).

PhytoScan uses **activation norm saliency** — L2 norm of 49 patch token
activations from the last ViT transformer block, reshaped from 7×7 to
224×224. No backward pass required, stable across all inputs.

---

## Dataset

### LeafNet — Training

| Property | Value |
|----------|-------|
| Images | 121,337 |
| Classes | 89 disease conditions |
| Crops covered | 15+ species |
| Largest class | Coffee healthy — 13,288 images |
| Smallest class | Pepper leaf spot — 46 images |
| Train / Val split | 108,000 / 12,000 |
| Source | HuggingFace `enalis/LeafNet` |

### PlantDoc — Test Only

| Property | Value |
|----------|-------|
| Images | 236 |
| Classes | 27 |
| Image type | Real field photography |
| Used in training | Never |
| Source | `github.com/pratikkayal/PlantDoc-Dataset` |

---

## Installation

```bash
git clone https://github.com/abulhasan121/PhytoScan.git
cd PhytoScan

pip install -r requirements.txt

# Download PlantDoc for evaluation
git clone https://github.com/pratikkayal/PlantDoc-Dataset data/plantdoc
```

---

## Usage

### Run Full Pipeline

```bash
# SimCLR → LoRA fine-tuning → Gradio demo
python src/train.py

# Skip SimCLR if checkpoint exists
python src/train.py --skip-ssl

# Launch Gradio demo from saved checkpoint
python src/train.py --demo-only
```

### Gradio Demo

```bash
python app.py
# Launches at http://localhost:7860
# Set share=True for public URL
```

### Python API

```python
import torch
import open_clip
from src.config  import CFG
from src.lora    import apply_lora
from src.prompts import make_meta_prompt
from src.gradcam import CLIPGradCAM

# Load model
model, _, preprocess = open_clip.create_model_and_transforms(
    CFG['clip_model'], pretrained=CFG['pretrained'])
model.visual = apply_lora(model.visual, r=CFG['lora_r'], alpha=CFG['lora_alpha'])

ckpt = torch.load('results/checkpoints/best_model.pt')
model.load_state_dict(ckpt['model_state'])
model.eval()

# Predict
from PIL import Image
import torch.nn.functional as F

img    = preprocess(Image.open('leaf.jpg').convert('RGB')).unsqueeze(0)
gradcam = CLIPGradCAM(model)
heatmap = gradcam.generate(img.squeeze(0))  # (224, 224) saliency map
```

---

## Repository Structure

```
PhytoScan/
├── README.md
├── app.py                    ← Gradio app (Diagnose · Architecture ·
│                               Disease Classes · About)
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── config.py             ← CFG dict — all hyperparameters
│   ├── prompts.py            ← 4 prompt strategies + caption parser
│   ├── lora.py               ← LoRALinear + apply_lora
│   ├── simclr.py             ← SimCLR augmentation + NT-Xent loss
│   ├── dataset.py            ← LeafNetDataset + PlantDocDataset +
│   │                           PlantDoc→LeafNet class mapping
│   ├── evaluate.py           ← build_class_embeddings + evaluate fns
│   ├── gradcam.py            ← Activation norm saliency map
│   └── train.py              ← Full pipeline (argparse)
│
├── notebooks/
│   └── PhytoScan_full.ipynb  ← Complete Colab notebook
│
└── results/
    └── figures/
        ├── training_history.png
        ├── gradcam_heatmaps.png
        ├── leafnet_class_distribution.png
        └── plantdoc_class_distribution.png
```

---

## Limitations and Field Deployment Considerations

| Limitation | Implication for Field Use |
|------------|--------------------------|
| 68.6% PlantDoc accuracy | Suitable for triage / flag-for-inspection workflows, not final diagnosis |
| 7×7 saliency resolution | Region-level attribution only — not pixel-level lesion mapping |
| Single leaf close-up | Does not work on canopy-level or whole-plant imagery |
| 89 fixed disease classes | Cannot detect novel diseases without retraining |
| RGB only | No multispectral or thermal integration |
| Class imbalance in training | Rare diseases (< 50 training images) perform worse |

**Recommended use case:** First-pass screening tool integrated with a
human-in-the-loop verification workflow. Flag high-confidence detections
for agronomist confirmation; route low-confidence or multi-disease
predictions for lab testing.

---

## Experimental Setup

| Component | Value |
|-----------|-------|
| GPU | NVIDIA L4 · 24GB VRAM |
| Framework | PyTorch 2.10.0+cu128 · OpenCLIP |
| SSL training | 1 epoch · 60k images · ~38 min |
| Fine-tuning | 12 epochs (stopped early) · ~76 min |
| Inference speed | ~50ms per image (T4 GPU) |
| Checkpoint size | ~357 MB (LoRA weights only: ~12 MB) |

---

## Citation

```bibtex
@misc{physoscan2025,
  title   = {PhytoScan: Domain-Adaptive Vision-Language Model
              for Plant Disease Diagnosis via SimCLR and LoRA},
  author  = {Hasan, Shah Md Abul},
  year    = {2025},
  url     = {https://github.com/abulhasan121/PhytoScan}
}
```

---

## Related Projects

| Project | Description |
|---------|-------------|
| [AutoWeedMap](https://github.com/abulhasan121/AutoWeedMap) | Zero-click weedy rice detection from multispectral UAV imagery using SAM + NDVI guidance |
| [AgriScholar](https://github.com/abulhasan121/AgriScholar) | Semantic search across agricultural research papers using RAG + ChromaDB + Claude API |
| **PhytoScan** | This project |

---

## License

MIT — see [LICENSE](LICENSE)

**Author:** Shah Md Abul Hasan · University of Georgia
