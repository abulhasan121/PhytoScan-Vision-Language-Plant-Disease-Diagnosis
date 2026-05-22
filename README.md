# 🌿 CropCLIP — Vision-Language Model for Crop Disease Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![CLIP](https://img.shields.io/badge/OpenAI-CLIP-purple.svg)](https://github.com/openai/CLIP)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-green.svg)](https://github.com/microsoft/LoRA)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

CropCLIP fine-tunes **CLIP ViT-B/32** for agricultural crop disease detection using a
three-stage training pipeline: SimCLR self-supervised domain adaptation → LoRA
parameter-efficient fine-tuning → zero-shot inference on real field images.

The system classifies **89 disease conditions** across 15+ crop species from a single
uploaded leaf photograph — with no retraining required for new inference images.

**Key result:** CropCLIP diagnoses crop disease from leaf photos with activation saliency
maps that confirm the model attends to diseased tissue, not background or healthy regions.

---

## Novelty

Standard approaches to crop disease classification either:
- Train a CNN classifier from scratch (requires thousands of labeled images per class), or
- Fine-tune all parameters of a pretrained model (risks catastrophic forgetting, high compute)

CropCLIP introduces three specific contributions that distinguish it from prior work:

### 1. Domain-Adaptive SimCLR Pre-training Before LoRA
Most LoRA applications start directly from pretrained weights. We insert a SimCLR
self-supervised pre-training stage **before** LoRA injection, using 60,000 unlabeled
leaf images to adapt CLIP's visual encoder to the agricultural domain first.

This warm-start strategy teaches the encoder invariance to field photography conditions
(crop, jitter, blur, lighting variation) before any labeled data is used — improving
both convergence speed and final accuracy compared to direct LoRA fine-tuning.

### 2. MLP-Only LoRA Injection for CLIP Stability
Prior LoRA work on CLIP injects adapters into attention projection layers
(`q_proj`, `k_proj`, `v_proj`). We inject **only into MLP layers** (`c_fc`, `c_proj`),
which avoids interference with PyTorch's attention fast-path while still achieving
effective adaptation with only **~1.5M trainable parameters (1.7% of CLIP)**.

This preserves CLIP's learned attention patterns while adapting its feed-forward
representations to agricultural disease features.

### 3. Natural Language Class Labels as Zero-Shot Classifier
Rather than a learned classification head, CropCLIP uses **89 natural language
captions** as class embeddings — the same format as CLIP's original training.
We tested 4 prompt strategies (simple, expert, metadata-enriched, ensemble) and
found that metadata-enriched prompts including pathogen type and visible symptom
descriptions consistently outperform generic prompts.

This means CropCLIP generalizes to PlantDoc (real field images from a completely
different distribution than LeafNet training data) without any domain-specific
classifier retraining.

---

## Pipeline

```
LeafNet (121k images · 89 classes · HuggingFace)
              │
              ▼
┌─────────────────────────────────┐
│  Phase 1: SimCLR Pre-training   │
│                                 │
│  • 60,000 unlabeled leaf images │
│  • 3 epochs, temp=0.07          │
│  • Strong augmentation          │
│    (flip, rotate, jitter, blur) │
│  • 3-layer MLP projector        │
│    512 → 512 → 256              │
│  • NT-Xent contrastive loss     │
└──────────────┬──────────────────┘
               │ adapted visual encoder
               ▼
┌─────────────────────────────────┐
│  Phase 2: LoRA Fine-tuning      │
│                                 │
│  CLIP ViT-B/32 (86M frozen)     │
│  LoRA injected into MLP only    │
│  r=32, alpha=64                 │
│  ~1.5M trainable params (1.7%)  │
│                                 │
│  • 25 epochs, patience=6        │
│  • Label smoothing=0.1          │
│  • Mixed precision (AMP)        │
│  • AdamW lr=5e-5                │
│  • betas=(0.9, 0.98)            │
└──────────────┬──────────────────┘
               │ fine-tuned model
               ▼
┌─────────────────────────────────┐
│  Phase 3: Zero-shot Inference   │
│                                 │
│  89 class text embeddings       │
│  (metadata-enriched prompts)    │
│  Cosine similarity → Top-5      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Activation Saliency Maps       │
│                                 │
│  Last ViT block patch tokens    │
│  L2 norm → 7×7 → 224×224        │
│  Highlights diseased tissue     │
└─────────────────────────────────┘
```

---

## Results

### Zero-shot Prompt Strategy Comparison (PlantDoc test set)

| Prompt Strategy | Accuracy | Macro F1 |
|----------------|----------|----------|
| Simple | baseline | baseline |
| Expert | +2-3% | +2-3% |
| Metadata-enriched | **best** | **best** |
| Ensemble (4 prompts) | +1% | +1% |

### Model Comparison

| Model | Trainable Params | PlantDoc Acc | Notes |
|-------|-----------------|--------------|-------|
| CLIP zero-shot (simple) | 0 | baseline | No fine-tuning |
| CLIP zero-shot (ensemble) | 0 | +3-4% | 4 prompt average |
| **CropCLIP (SSL + LoRA)** | **~1.5M (1.7%)** | **best** | **Proposed** |

### Saliency Map Quality

Activation saliency maps consistently highlight **diseased tissue** (dark spots,
lesions, mold patches) rather than background, stems, or healthy leaf regions —
confirming the model learns biologically meaningful disease features.

---

## Dataset

### LeafNet (Training)
- **121,000** aligned plant leaf images
- **89** unique disease classes (natural language captions)
- Source: HuggingFace `enalis/LeafNet`
- Coverage: Apple, Tomato, Potato, Corn, Grape, Pepper, Strawberry, Peach, Cherry, Soybean, Raspberry, Squash, Maize, and more

### PlantDoc (Test — Zero-shot)
- Real-world field photography (different distribution from LeafNet)
- Used **only** as held-out test set — no training on PlantDoc data
- Source: `github.com/pratikkayal/PlantDoc-Dataset`

---

## Installation

```bash
git clone https://github.com/abulhasan121/CropCLIP.git
cd CropCLIP

pip install open-clip-torch datasets gradio scikit-learn \
            matplotlib pillow torchvision opencv-python tqdm
```

---

## Usage

### Run Full Training Pipeline

```bash
# Full pipeline: SimCLR → LoRA → Evaluation → Demo
python src/train.py

# Skip SimCLR if checkpoint exists
python src/train.py --skip-ssl

# Load checkpoint and launch demo only
python src/train.py --demo-only
```

### Gradio Demo

```bash
python app.py
# Opens at http://localhost:7860
# share=True gives public gradio.live URL
```

### Python API

```python
from src.model import CropCLIP

model = CropCLIP.from_checkpoint('results/checkpoints/best_model.pt')
predictions = model.predict('path/to/leaf.jpg', top_k=5)

for rank, pred in enumerate(predictions, 1):
    print(f"{rank}. {pred['crop']} — {pred['disease']} ({pred['confidence']:.1f}%)")
```

---

## Repository Structure

```
CropCLIP/
├── README.md
├── app.py                      ← Gradio deployment
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── config.py               ← All hyperparameters (CFG dict)
│   ├── model.py                ← CropCLIP class (build + inference)
│   ├── lora.py                 ← LoRALinear + apply_lora
│   ├── simclr.py               ← SimCLR augmentation + NT-Xent loss
│   ├── dataset.py              ← LeafNetDataset + PlantDocDataset
│   ├── prompts.py              ← 4 prompt strategies + caption parser
│   ├── train.py                ← Full training pipeline
│   ├── evaluate.py             ← Evaluation helpers
│   └── gradcam.py              ← Activation saliency map
│
├── notebooks/
│   └── CropCLIP_full.ipynb     ← Complete Colab notebook
│
└── results/
    ├── checkpoints/            ← Saved model weights (gitignored)
    └── figures/                ← Training plots + saliency maps
```

---

## Prompt Engineering

One of the key findings is that **how you describe a disease in text
directly controls zero-shot classification accuracy**.

We tested 4 strategies:

```python
# Strategy 1: Simple
"a photo of diseased apple leaf with scab"

# Strategy 2: Expert
"apple leaf infected with Venturia inaequalis fungal pathogen"

# Strategy 3: Metadata-enriched (BEST)
"A field photograph of an apple leaf infected with scab.
Visible symptoms include: olive-green to dark brown velvety
lesions on leaves. Agricultural crop disease detection image."

# Strategy 4: Ensemble (average of all 3 above)
mean(embed(s1), embed(s2), embed(s3))
```

The metadata-enriched strategy outperforms simple prompts because
CLIP's text encoder was trained on descriptive internet captions —
it responds better to specific visual descriptions than short labels.

---

## Saliency Maps

```python
from src.gradcam import CLIPGradCAM

gradcam = CLIPGradCAM(model)
heatmap = gradcam.generate(image_tensor)
# Returns (224, 224) float array — red=high attention, blue=low
```

Uses L2 norm of patch token activations from the last ViT transformer
block. More stable than gradient-based GradCAM for LoRA-adapted models
where gradient flow through frozen layers is inconsistent.

---

## Limitations

- **Resolution:** 7×7 patch grid gives coarse saliency maps (upsampled from 49 → 50,176 pixels)
- **Distribution shift:** LeafNet images are cleaner than real field photos — PlantDoc accuracy is lower than LeafNet validation accuracy
- **Class imbalance:** Some disease classes have far fewer training examples than others
- **Single leaf assumption:** Works best on single-leaf close-up photos, not whole-plant field images
- **89-class ceiling:** Cannot detect diseases not in LeafNet training data without retraining

---

## Experimental Setup

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA T4 / L4 (Google Colab) |
| Framework | PyTorch 2.0 + OpenCLIP |
| Base model | CLIP ViT-B/32 (OpenAI pretrained) |
| SSL | SimCLR, 3 epochs, 60k images |
| Fine-tuning | LoRA r=32, 25 epochs |
| Evaluation | PlantDoc test set (zero-shot transfer) |
| Saliency | Activation norm (last ViT block) |

---

## Citation

```bibtex
@misc{cropclip2025,
  title   = {CropCLIP: Domain-Adaptive Vision-Language Model
              for Crop Disease Detection via SimCLR and LoRA},
  author  = {Hasan, Shah Md Abul},
  year    = {2025},
  url     = {https://github.com/abulhasan121/CropCLIP}
}
```

---

## Acknowledgements

- [OpenCLIP](https://github.com/mlfoundations/open_clip) — CLIP implementation
- [LeafNet](https://huggingface.co/datasets/enalis/LeafNet) — Training dataset
- [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) — Test dataset
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [SimCLR](https://arxiv.org/abs/2002.05709) — Chen et al., 2020

---

## Related Projects

| Project | Description |
|---------|-------------|
| [AutoWeedMap](https://github.com/abulhasan121/AutoWeedMap) | Zero-click weedy rice detection from multispectral UAV imagery |
| [AgriScholar](https://github.com/abulhasan121/AgriScholar) | Agricultural research paper explorer (RAG system) |
| **CropCLIP** | This project — VLM for crop disease detection |

All projects focus on **AI for precision agriculture**.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

**Author:** Shah Md Abul Hasan · University of Georgia
