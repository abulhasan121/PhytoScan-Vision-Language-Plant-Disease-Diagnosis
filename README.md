# PhytoScan — Vision-Language Plant Disease Diagnosis

**Developed by Shah Md Abul Hasan · University of Georgia**

PhytoScan is a vision-language plant disease diagnosis system built on top of OpenAI CLIP, SimCLR self-supervised learning, and LoRA parameter-efficient fine-tuning. The model diagnoses 89 crop disease conditions directly from leaf photographs while remaining lightweight enough for practical deployment.

Unlike traditional CNN classifiers trained as closed-set classification systems, PhytoScan treats disease diagnosis as a semantic vision-language matching problem. Instead of predicting fixed numeric classes, the system compares a leaf image against descriptive disease captions written in natural language.

The model achieves:
- 68.6% accuracy on PlantDoc real-field images
- 47.1% Macro F1 across unseen field conditions
- 49.5% improvement over standard zero-shot CLIP

---

# Project Motivation

Accurate disease diagnosis is a major challenge in precision agriculture because disease pressure can vary significantly across fields, cultivars, environments, and growth stages. Early and reliable disease detection is essential for site-specific fungicide application, yield protection, and reducing unnecessary chemical inputs. However, field-scale disease scouting still relies heavily on manual inspection by trained agronomists, which is labor-intensive, time-consuming, and difficult to scale across large production systems.

Most existing deep learning approaches for plant disease diagnosis are based on traditional CNN classifiers trained on controlled greenhouse datasets. While these models often report very high accuracy, they frequently fail under real agricultural field conditions where images contain:

Variable lighting and shadows
Soil and canopy background clutter
Multiple overlapping leaves
Motion blur and focus variation
Different cameras, angles, and environmental conditions
Large differences among cultivars and growth stages

A key limitation of conventional CNN classifiers is that they treat disease diagnosis as a fixed closed-set classification problem. Each disease is represented only as a numeric label, forcing the model to memorize visual patterns from large labeled datasets. This creates major scalability challenges for precision agriculture because:

New diseases require retraining the classifier
Rare diseases suffer from insufficient labeled data
Models transfer poorly across crops and production environments
Farm-specific retraining is often necessary

PhytoScan addresses these limitations by using a vision-language model (VLM) architecture instead of a traditional CNN classifier. Rather than predicting fixed numeric classes, the system maps both leaf images and natural language disease descriptions into a shared semantic embedding space. Disease diagnosis becomes a similarity-matching problem between visual symptoms and descriptive agronomic text.

This vision-language formulation is particularly valuable for precision agriculture because it enables:

Stronger generalization across diverse field environments
Reduced dependence on massive labeled agricultural datasets
Easier adaptation to new crops and disease conditions
More interpretable predictions linked to biological symptom descriptions
Integration with multimodal agricultural sensing systems

The novelty of PhytoScan lies in adapting a large-scale vision-language foundation model (CLIP) specifically for agricultural disease diagnosis using:

SimCLR self-supervised domain adaptation on unlabeled leaf imagery
Parameter-efficient LoRA fine-tuning
Metadata-enriched disease prompt engineering
Explainable saliency-based visualization
Real-field evaluation on unseen PlantDoc imagery

Unlike traditional plant disease classifiers optimized only for laboratory datasets, PhytoScan was designed with real precision agriculture deployment in mind. The model operates on standard RGB imagery captured from smartphones, handheld field cameras, or UAV platforms, making it compatible with modern precision agriculture workflows. For example, UAV-based canopy stress monitoring can first identify suspicious field regions, after which PhytoScan can provide plant-level disease diagnosis from close-range imagery for targeted verification and management decisions.

By combining foundation vision-language models with agricultural domain adaptation, PhytoScan demonstrates a scalable framework for next-generation AI-assisted crop health monitoring systems capable of supporting precision scouting, disease surveillance, and site-specific crop protection strategies under real production conditions.
---

# System Overview

The system operates as a retrieval-style vision-language pipeline.

Instead of learning a traditional classification head, PhytoScan:
1. Encodes disease descriptions into text embeddings
2. Encodes leaf images into visual embeddings
3. Computes cosine similarity between them
4. Returns the highest matching disease conditions

This enables:
- Flexible disease descriptions
- Natural language extensibility
- Improved generalization
- Explainable predictions through saliency maps

---

# Inference Pipeline

```text
Leaf Image
      ↓
CLIP Visual Encoder
      ↓
Visual Embedding
      ↓
Cosine Similarity
      ↓
Disease Caption Embeddings
      ↓
Ranked Predictions
      ↓
Top-5 Diseases + Confidence Scores + Saliency Map
```

---

# Supported Crops

PhytoScan supports disease diagnosis across multiple crop species including:

- Apple
- Tomato
- Potato
- Corn
- Maize
- Grape
- Pepper
- Strawberry
- Peach
- Cherry
- Soybean
- Raspberry
- Squash
- Coffee
- Black Pepper

and additional disease conditions for a total of 89 crop-disease combinations.

---

# Field Performance

The model was evaluated on PlantDoc, a challenging real-field benchmark containing images never seen during training.

Unlike controlled datasets, PlantDoc includes:
- Natural lighting variation
- Background clutter
- Occlusion
- Real field disease presentation

This benchmark measures actual field generalization rather than laboratory performance.

## PlantDoc Test Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| CLIP zero-shot — simple labels | 14.4% | 7.0% |
| CLIP zero-shot — expert prompts | 19.1% | 10.6% |
| CLIP zero-shot — symptom prompts | 17.8% | 7.9% |
| CLIP zero-shot — prompt ensemble | 18.2% | 8.5% |
| **PhytoScan** | **68.6%** | **47.1%** |

### Key Observation

The strongest improvement came from:
- Domain adaptation via SimCLR
- Agricultural prompt engineering
- LoRA fine-tuning
- Metadata-enriched disease descriptions

The model achieved:
- +49.5% accuracy improvement over standard zero-shot CLIP
- Strong transfer to unseen field environments
- Robustness to real agricultural image conditions

---

# Training Dynamics

PhytoScan converged rapidly after self-supervised domain adaptation.

PlantDoc accuracy increased from:
- 36.0% → 60.6% within 3 epochs
- Final best accuracy reached at epoch 6

This confirms that SimCLR pretraining provided agriculturally relevant initialization before supervised fine-tuning.

## Training History

| Epoch | Train Loss | Validation Accuracy | PlantDoc Accuracy | PlantDoc F1 |
|---|---|---|---|---|
| 1 | 1.0998 | 92.3% | 36.0% | 20.0% |
| 2 | 0.2107 | 94.2% | 52.5% | 33.3% |
| 3 | 0.1504 | 94.8% | 60.6% | 33.9% |
| 5 | 0.0857 | 95.4% | 66.9% | 45.4% |
| **6** | **0.0610** | **95.3%** | **68.6%** | **47.1%** |
| 12 | 0.0153 | 95.8% | 66.9% | 45.2% |

### Early Stopping Behavior

LeafNet validation accuracy continued increasing while PlantDoc performance plateaued.

This suggests:
- Distribution shift between greenhouse and field imagery
- Overfitting risk on controlled datasets
- Importance of real-field evaluation

---

# Why This Matters for Precision Agriculture

## 1. No Per-Farm Retraining

Traditional CNN disease classifiers require:
- Thousands of labeled images
- Farm-specific retraining
- New classifier heads for new diseases

PhytoScan instead uses:
- Natural language disease descriptions
- Semantic embedding matching
- Flexible zero-shot transfer

Adding a new disease becomes:
- Writing a caption
- Not collecting a new dataset

---

## 2. Explainable Predictions

The system produces saliency maps showing:
- Which leaf regions influenced the prediction
- Which lesions or discolorations were used
- Whether the model focused on disease tissue

Correct predictions typically focus on:
- Lesions
- Chlorosis
- Necrosis
- Fungal spots

Incorrect predictions generally remain biologically meaningful:
- Confusion occurs between visually similar diseases
- Not between disease and background

This is an important property for agricultural decision support systems.

---

## 3. Parameter-Efficient Deployment

Only 2.95 million parameters are trainable through LoRA adapters.

The original CLIP backbone remains frozen.

Advantages:
- Smaller checkpoints
- Faster fine-tuning
- Lower GPU memory usage
- Shared reusable backbone

Only 1.91% of CLIP parameters were updated during training.

---

## 4. Integration with UAV Pipelines

PhytoScan operates on standard RGB imagery and can integrate with:
- DJI drone workflows
- Smartphone field scouting
- UAV canopy monitoring
- Multispectral disease detection systems

Potential agricultural workflow:
1. UAV detects canopy stress
2. Ground-level RGB image captured
3. PhytoScan diagnoses disease
4. Agronomist verifies treatment decision

---

# Technical Architecture

The system follows a three-stage training pipeline.

---

## Stage 1 — SimCLR Domain Adaptation

Unlabeled leaf imagery was used for self-supervised contrastive learning.

The objective was to adapt CLIP's visual encoder to agricultural image distributions before supervised fine-tuning.

### SimCLR Objectives

The model learns invariance to:
- Lighting variation
- Camera angle
- Motion blur
- Focus variation
- Weather conditions
- Background clutter

### SSL Training

| Property | Value |
|---|---|
| Images | 60,000 |
| Loss Reduction | 6.01 → 3.37 |
| Epochs | 1 |
| Training Time | ~38 min |

---

## Stage 2 — LoRA Fine-Tuning

The CLIP ViT-B/32 backbone remained frozen.

LoRA adapters were inserted into:
- MLP feed-forward layers only

### LoRA Configuration

| Parameter | Value |
|---|---|
| Backbone | CLIP ViT-B/32 |
| Total Parameters | 154.2M |
| Trainable Parameters | 2.95M |
| Trainable Fraction | 1.91% |
| Rank (r) | 32 |
| Alpha | 64 |

### Why MLP-Only LoRA

Injecting LoRA into attention projections disrupted:
- CLIP visual-language alignment
- PyTorch attention optimization paths

MLP-only injection:
- Preserved pretrained alignment
- Improved agricultural adaptation
- Passed all gradient integrity checks

---

## Stage 3 — Vision-Language Inference

The final system performs zero-shot style retrieval.

### Inference Process

```text
Leaf Image
    ↓
Visual Embedding
    ↓
Cosine Similarity
    ↓
Disease Caption Embeddings
    ↓
Ranked Predictions
```

No:
- Calibration
- Threshold tuning
- Post-processing
- Specialized classifier head

was required.

---

# Prompt Engineering

Prompt quality significantly influenced CLIP performance.

Simple labels underperformed because CLIP was pretrained on descriptive internet text.

## Example Prompt Evolution

### Poor Prompt

```text
"a photo of apple leaf with scab"
```

### Better Prompt

```text
"apple leaf infected with Venturia inaequalis fungal pathogen"
```

### Best Prompt Style

```text
"A field photograph of an apple leaf infected with scab.
Visible symptoms include olive-green to dark brown lesions.
Agricultural crop disease detection image."
```

Metadata-enriched prompts improved semantic separation between disease classes.

---

# Saliency Map Generation

Traditional GradCAM methods failed because:
- CLIP backbone layers remained frozen
- Intermediate gradients vanished
- Backward propagation produced near-zero activations

PhytoScan instead uses:
- Activation norm saliency

### Method

1. Extract final ViT patch token activations
2. Compute L2 norm per patch
3. Reshape from 7×7 → 224×224
4. Generate heatmap without backward propagation

Advantages:
- Stable
- No gradient dependency
- Works consistently with frozen CLIP

---

# Dataset Information

## LeafNet — Training Dataset

| Property | Value |
|---|---|
| Images | 121,337 |
| Classes | 89 |
| Crops | 15+ |
| Largest Class | Coffee healthy — 13,288 |
| Smallest Class | Pepper leaf spot — 46 |
| Train / Validation Split | 108k / 12k |
| Source | HuggingFace `enalis/LeafNet` |

---

## PlantDoc — Field Evaluation Dataset

| Property | Value |
|---|---|
| Images | 236 |
| Classes | 27 |
| Image Type | Real field photography |
| Used in Training | No |
| Source | PlantDoc Dataset |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/abulhasan121/PhytoScan.git
cd PhytoScan
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Download PlantDoc

```bash
git clone https://github.com/pratikkayal/PlantDoc-Dataset data/plantdoc
```

---

# Running the Project

## Full Training Pipeline

```bash
python src/train.py
```

## Skip SimCLR Stage

```bash
python src/train.py --skip-ssl
```

## Launch Demo Only

```bash
python src/train.py --demo-only
```

---

# Gradio Interface

Launch the web interface:

```bash
python app.py
```

Then open:

```text
http://localhost:7860
```

The interface supports:
- Leaf image upload
- Top-5 disease predictions
- Confidence visualization
- Saliency heatmaps
- Architecture overview

---

# Python API Example

```python
import torch
import open_clip

from src.config import CFG
from src.lora import apply_lora
from src.gradcam import CLIPGradCAM

model, _, preprocess = open_clip.create_model_and_transforms(
    CFG['clip_model'],
    pretrained=CFG['pretrained']
)

model.visual = apply_lora(
    model.visual,
    r=CFG['lora_r'],
    alpha=CFG['lora_alpha']
)

checkpoint = torch.load('results/checkpoints/best_model.pt')

model.load_state_dict(checkpoint['model_state'])
model.eval()
```

---

# Repository Structure

```text
PhytoScan/
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── config.py
│   ├── prompts.py
│   ├── lora.py
│   ├── simclr.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── train.py
│
├── notebooks/
│   └── PhytoScan_full.ipynb
│
└── results/
    └── figures/
        ├── training_history.png
        ├── gradcam_heatmaps.png
        ├── leafnet_class_distribution.png
        └── plantdoc_class_distribution.png
```

---

# Limitations and Deployment Considerations

| Limitation | Field Implication |
|---|---|
| 68.6% accuracy | Suitable for triage, not final diagnosis |
| 7×7 saliency resolution | Region-level explanation only |
| Single leaf input | Cannot process canopy-scale imagery |
| Fixed 89 disease classes | Cannot detect unknown diseases |
| RGB only | No multispectral integration |
| Class imbalance | Rare diseases perform worse |

### Recommended Deployment Strategy

PhytoScan is best used as:
- A first-pass field screening tool
- Agronomist decision support
- UAV follow-up diagnosis
- Human-in-the-loop disease verification

Not as:
- Fully autonomous diagnosis
- Laboratory replacement
- Regulatory decision system

---

# Experimental Setup

| Component | Value |
|---|---|
| GPU | NVIDIA L4 — 24 GB VRAM |
| Framework | PyTorch 2.10 + OpenCLIP |
| SSL Training | 1 epoch · 60k images |
| Fine-Tuning | 12 epochs |
| Inference Speed | ~50 ms/image |
| Full Checkpoint Size | ~357 MB |
| LoRA Weights Only | ~12 MB |

---

# Related Projects

| Project | Description |
|---|---|
| AutoWeedMap | SAM-guided weedy rice detection from UAV imagery |
| AgriScholar | Agricultural RAG system for scientific literature |
| PhytoScan | Vision-language plant disease diagnosis |

---

# Citation

```bibtex
@misc{phytoscan2025,
  title   = {PhytoScan: Domain-Adaptive Vision-Language Model for Plant Disease Diagnosis via SimCLR and LoRA},
  author  = {Hasan, Shah Md Abul},
  year    = {2025},
  url     = {https://github.com/abulhasan121/PhytoScan}
}
```

---

# Author

**Shah Md Abul Hasan**

Built with:
- OpenCLIP
- PyTorch
- LoRA
- SimCLR
- Gradio

---

# License

MIT License
