"""
app.py
CropCLIP Gradio Deployment — Crop Disease Detection Demo

Run:
    python app.py
    # or from train.py: python src/train.py --demo-only
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import gradio as gr
from PIL import Image

from src.config   import CFG
from src.prompts  import parse_caption, make_meta_prompt
from src.gradcam  import CLIPGradCAM
from src.evaluate import build_class_embeddings


# ══════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════

def predict_disease(image_pil,
                    model, preprocess, tokenizer,
                    class_emb_demo, gradcam_demo,
                    unique_captions, device):
    if image_pil is None:
        return "⚠️ Please upload a leaf image.", None, None

    img_resized = image_pil.convert('RGB').resize((224, 224))
    img_tensor  = preprocess(img_resized).to(device)
    img_array   = np.array(img_resized) / 255.0

    model.eval()
    with torch.no_grad():
        img_emb = F.normalize(
            model.encode_image(img_tensor.unsqueeze(0)), dim=-1
        )
        probs = F.softmax(
            img_emb @ class_emb_demo.T * 100, dim=-1
        ).squeeze()

    top5_vals, top5_idx = probs.topk(5)

    # Prediction text
    lines = []
    for rank, (conf, idx) in enumerate(zip(top5_vals, top5_idx), 1):
        p       = parse_caption(unique_captions[idx.item()])
        crop    = p['crop'].title()
        disease = '✅ Healthy' if p['is_healthy'] \
                  else f"🔴 {p['disease'].title()}"
        lines.append(f"**{rank}. {crop} — {disease}**  `{conf*100:.1f}%`")
    prediction_md = "\n\n".join(lines)

    # Saliency overlay
    heatmap    = gradcam_demo.generate(img_tensor)
    overlay    = np.clip(
        0.55 * img_array + 0.45 * plt.cm.jet(heatmap)[..., :3], 0, 1
    )
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

    # Confidence chart
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    labels, values = [], []
    for idx, val in zip(top5_idx, top5_vals):
        p = parse_caption(unique_captions[idx.item()])
        d = 'Healthy' if p['is_healthy'] else p['disease'].title()
        labels.append(f"{p['crop'].title()}\n{d}")
        values.append(val.item() * 100)

    colors = ['#52b788'] + ['#2d6a4f'] * 4
    bars   = ax.barh(labels[::-1], values[::-1],
                     color=colors[::-1], height=0.55, edgecolor='none')
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', color='white',
                fontsize=9, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.set_xlabel('Confidence (%)', color='#b7e4c7', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    chart_path = '/tmp/cropclip_chart.png'
    plt.savefig(chart_path, dpi=130, bbox_inches='tight',
                facecolor='#0f1117')
    plt.close()
    chart_pil = Image.open(chart_path)

    return prediction_md, overlay_pil, chart_pil


# ══════════════════════════════════════════════════════════
# LAUNCH FUNCTION (called from train.py)
# ══════════════════════════════════════════════════════════

def launch_demo(model, preprocess, tokenizer,
                unique_captions, caption_to_idx,
                share: bool = True):

    device         = next(model.parameters()).device
    class_emb_demo = build_class_embeddings(
        model, tokenizer, unique_captions,
        make_meta_prompt, str(device)
    ).detach()
    gradcam_demo   = CLIPGradCAM(model)
    num_classes    = len(unique_captions)

    # Wrap predict with fixed model args
    def predict(image_pil):
        return predict_disease(
            image_pil, model, preprocess, tokenizer,
            class_emb_demo, gradcam_demo,
            unique_captions, str(device)
        )

    # ── PlantDoc examples ──────────────────────────────────
    example_paths = []
    pd_path = CFG['plantdoc_path']
    if os.path.exists(pd_path):
        for cls in sorted(os.listdir(pd_path))[:3]:
            cls_dir = os.path.join(pd_path, cls)
            if os.path.isdir(cls_dir):
                imgs = [f for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.jpg','.png'))]
                if imgs:
                    example_paths.append(
                        [os.path.join(cls_dir, imgs[0])]
                    )

    # ── CSS ───────────────────────────────────────────────
    css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
body,.gradio-container{background:#0f1117 !important;font-family:'Syne',sans-serif !important;color:#e0e0e0 !important;}
.gradio-container{max-width:1350px !important;margin:0 auto !important;}
.hdr{background:linear-gradient(135deg,#1a3a2a 0%,#0d2d1e 100%);padding:2rem 2.5rem;margin-bottom:1.5rem;border-bottom:3px solid #52b788;border-radius:8px;}
.ttl{font-size:2.2rem;font-weight:800;color:#b7e4c7;letter-spacing:-0.02em;margin:0;}
.sub{font-family:'DM Mono',monospace;font-size:0.78rem;color:#52b788;margin-top:0.4rem;}
.bdg{display:inline-block;background:#2d6a4f;color:#b7e4c7;font-family:'DM Mono',monospace;font-size:.62rem;padding:.12rem .45rem;border-radius:2px;margin-right:.3rem;text-transform:uppercase;letter-spacing:.08em;}
#diag-btn{background:#52b788 !important;color:#0f1117 !important;font-weight:800 !important;font-size:1rem !important;border:none !important;border-radius:6px !important;padding:0.8rem !important;width:100% !important;}
#diag-btn:hover{background:#74c69d !important;}
.tip{background:#1a2e1a;border-left:3px solid #52b788;padding:.65rem .9rem;border-radius:0 5px 5px 0;font-family:'DM Mono',monospace;font-size:.75rem;color:#b7e4c7;margin-bottom:.8rem;}
"""

    # ── Gradio app ─────────────────────────────────────────
    with gr.Blocks(css=css, title="CropCLIP") as demo:

        gr.HTML(f"""
        <div class="hdr">
            <div class="ttl">🌿 CropCLIP</div>
            <div class="sub">
                Crop Disease Detection · CLIP ViT-B/32 +
                SimCLR Pre-training + LoRA Fine-tuning
            </div>
            <div style="margin-top:.8rem;">
                <span class="bdg">CLIP ViT-B/32</span>
                <span class="bdg">SimCLR SSL</span>
                <span class="bdg">LoRA r={CFG['lora_r']}</span>
                <span class="bdg">{num_classes} Classes</span>
                <span class="bdg">Saliency Maps</span>
                <span class="bdg">PlantDoc Tested</span>
            </div>
        </div>
        """)

        with gr.Tabs():

            # ── Diagnose ──────────────────────────────────
            with gr.Tab("🔬 Diagnose"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""<div class="tip">
                            📸 Upload a clear leaf photo.<br>
                            Works best with close-up field images.
                        </div>""")
                        img_input = gr.Image(
                            type='pil', label='Upload Leaf Image', height=280)
                        diag_btn  = gr.Button(
                            "🔍  Diagnose Disease",
                            elem_id="diag-btn", size="lg")
                        gr.HTML("""<div class="tip" style="margin-top:1rem;">
                            <b>Supported crops:</b><br>
                            Apple · Tomato · Potato · Corn · Grape<br>
                            Pepper · Strawberry · Peach · Cherry<br>
                            Soybean · Raspberry · Squash + more
                        </div>""")
                        if example_paths:
                            gr.Examples(
                                examples=example_paths,
                                inputs=[img_input],
                                label="PlantDoc Examples",
                                examples_per_page=6,
                            )

                    with gr.Column(scale=2):
                        with gr.Row():
                            overlay_out = gr.Image(
                                type='pil',
                                label='🌡️ Saliency Map',
                                height=280)
                            chart_out   = gr.Image(
                                type='pil',
                                label='📊 Confidence Scores',
                                height=280)
                        gr.HTML("""<div class="tip">
                            🔴 Red = high attention · 🔵 Blue = low attention
                        </div>""")
                        pred_out = gr.Markdown(
                            value="*Upload a leaf image and click Diagnose ↑*"
                        )

            # ── Architecture ──────────────────────────────
            with gr.Tab("🏗️ Architecture"):
                trainable = sum(p.numel() for p in model.parameters()
                                if p.requires_grad)
                frozen    = sum(p.numel() for p in model.parameters()
                                if not p.requires_grad)
                gr.Markdown(f"""
## CropCLIP Architecture

```
LeafNet (121k images · {num_classes} classes)
         │
         ▼
SimCLR Pre-training
 • {CFG['ssl_samples']:,} unlabeled images · {CFG.get('ssl_epochs',3)} epochs
 • temperature = {CFG['ssl_temp']}
 • 3-layer projector: 512→512→256
         │ adapted visual encoder
         ▼
CLIP ViT-B/32 ({frozen/1e6:.0f}M frozen params)
         │
         ▼
LoRA Injection (MLP layers only)
 • r={CFG['lora_r']}, alpha={CFG['lora_alpha']}
 • {trainable/1e3:.0f}K trainable params ({trainable/(frozen+trainable)*100:.2f}%)
         │
         ▼
Fine-tuning on LeafNet
 • {CFG['epochs']} epochs · patience={CFG['patience']}
 • Label smoothing={CFG.get('label_smoothing',0.1)}
 • AdamW lr={CFG['lr']}
         │
         ▼
Zero-shot Inference on PlantDoc
 • {num_classes} class text embeddings
 • Cosine similarity → Top-5
         │
         ▼
Activation Saliency
 • Last ViT block patch tokens
 • L2 norm → 7×7 → 224×224
```
                """)

            # ── Disease Classes ───────────────────────────
            with gr.Tab("📋 Disease Classes"):
                rows = []
                for i, cap in enumerate(unique_captions):
                    p = parse_caption(cap)
                    s = "✅ Healthy" if p['is_healthy'] \
                        else f"🔴 {p['disease'].title()}"
                    rows.append(f"| {i+1} | {p['crop'].title()} | {s} |")
                gr.Markdown(
                    f"## All {num_classes} Disease Classes\n\n"
                    "| # | Crop | Condition |\n|---|---|---|\n"
                    + "\n".join(rows)
                )

            # ── About ─────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
## CropCLIP

**Author:** Shah Md Abul Hasan · University of Georgia

**Novelty:**
1. SimCLR domain adaptation *before* LoRA (not standard)
2. MLP-only LoRA injection for CLIP stability
3. Metadata-enriched natural language class labels

| Setting | Value |
|---------|-------|
| Base model | CLIP ViT-B/32 |
| Training set | LeafNet (121k images) |
| Test set | PlantDoc (zero-shot) |
| Classes | {num_classes} |
| LoRA rank | {CFG['lora_r']} |
| Trainable | {trainable/1e3:.0f}K params ({trainable/(frozen+trainable)*100:.2f}%) |

**Related:**
[AutoWeedMap](https://github.com/abulhasan121/AutoWeedMap) ·
[AgriScholar](https://github.com/abulhasan121/AgriScholar)
                """)

        gr.HTML("""
        <div style="margin-top:2rem;padding:.8rem 2rem;
             background:#0d2d1e;color:#52b788;
             font-family:'DM Mono',monospace;font-size:.72rem;
             text-align:center;border-top:2px solid #1a3a2a;border-radius:8px;">
            CropCLIP · CLIP ViT-B/32 + SimCLR + LoRA ·
            Shah Md Abul Hasan · University of Georgia
        </div>
        """)

        diag_btn.click(
            fn=predict, inputs=[img_input],
            outputs=[pred_out, overlay_out, chart_out])
        img_input.change(
            fn=predict, inputs=[img_input],
            outputs=[pred_out, overlay_out, chart_out])

    print('\nLaunching CropCLIP demo...')
    demo.launch(share=share, show_error=True)


# ── Standalone entry point ─────────────────────────────────
if __name__ == '__main__':
    import open_clip
    from datasets import load_dataset

    print('Loading model from checkpoint...')
    ds      = load_dataset('enalis/LeafNet')
    leafnet = ds['train']
    unique_captions = sorted(set(leafnet['caption']))
    caption_to_idx  = {c: i for i, c in enumerate(unique_captions)}

    from src.lora import apply_lora
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, _, preprocess = open_clip.create_model_and_transforms(
        CFG['clip_model'], pretrained=CFG['pretrained'])
    tokenizer = open_clip.get_tokenizer(CFG['clip_model'])

    for p in model.parameters():
        p.requires_grad = False
    model.visual = apply_lora(model.visual,
                               r=CFG['lora_r'],
                               alpha=CFG['lora_alpha'],
                               dropout=CFG['lora_dropout'])
    model = model.to(DEVICE)

    ckpt = torch.load(CFG['best_checkpoint'], map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.logit_scale.data = ckpt['logit_scale']
    print(f"Loaded — PlantDoc acc: {ckpt['pd_acc']:.1f}%")

    launch_demo(model, preprocess, tokenizer,
                unique_captions, caption_to_idx)
