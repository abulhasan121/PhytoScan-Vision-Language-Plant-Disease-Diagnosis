"""
src/config.py
All hyperparameters for CropCLIP in one place.
"""

CFG = {
    # ── Model ──────────────────────────────────────────────
    'clip_model'      : 'ViT-B-32',
    'pretrained'      : 'openai',

    # ── Training ───────────────────────────────────────────
    'batch_size'      : 64,
    'epochs'          : 25,
    'lr'              : 5e-5,
    'weight_decay'    : 0.02,
    'warmup_steps'    : 500,
    'patience'        : 6,
    'seed'            : 42,
    'val_split'       : 0.1,
    'max_samples'     : 120000,
    'label_smoothing' : 0.1,
    'use_amp'         : True,

    # ── LoRA ───────────────────────────────────────────────
    'lora_r'          : 32,
    'lora_alpha'      : 64,
    'lora_dropout'    : 0.1,

    # ── SimCLR ─────────────────────────────────────────────
    'ssl_samples'     : 60000,
    'ssl_epochs'      : 3,
    'ssl_lr'          : 5e-5,
    'ssl_temp'        : 0.07,

    # ── Paths ──────────────────────────────────────────────
    'ssl_checkpoint'  : 'results/checkpoints/ssl_pretrained_visual.pt',
    'best_checkpoint' : 'results/checkpoints/best_model.pt',
    'figures_dir'     : 'results/figures',
    'plantdoc_path'   : 'data/plantdoc/test',
}
