"""
src/prompts.py
Caption parser and 4 prompt engineering strategies for CropCLIP.

Key finding: metadata-enriched prompts with symptom descriptions
outperform simple or expert prompts for zero-shot CLIP classification.
"""

import re


def parse_caption(caption):
    """
    Parse a LeafNet natural-language caption into structured fields.

    LeafNet captions follow two formats:
      Healthy  : 'a image of {Crop} healthy leaves with leaves appearing...'
      Diseased : 'a image of {Crop} leaves diseased by {Disease}
                  with symptoms of {Symptoms}'
    """
    text   = re.sub(r'^a image of\s+', '', caption.strip(), flags=re.IGNORECASE)
    result = {
        'raw'       : caption,
        'crop'      : '',
        'disease'   : '',
        'symptoms'  : '',
        'is_healthy': False
    }

    m = re.match(
        r'^(.+?)\s+leaves\s+diseased\s+by\s+(.+?)'
        r'(?:\s+with\s+symptoms?\s+of\s+(.+))?$',
        text, re.IGNORECASE
    )
    if m:
        result['crop']     = m.group(1).strip().lower()
        result['disease']  = m.group(2).strip().lower()
        result['symptoms'] = (m.group(3) or '').strip().lower()
        return result

    m = re.match(r'^(.+?)\s+healthy\s+leaves', text, re.IGNORECASE)
    if m:
        result['crop']       = m.group(1).strip().lower()
        result['disease']    = 'healthy'
        result['is_healthy'] = True
        return result

    result['crop']    = text.split()[0].lower()
    result['disease'] = text
    return result


# ── Strategy 1: Simple ────────────────────────────────────────────
def make_simple_prompt(caption):
    """Short label-style prompt. Lowest accuracy."""
    p = parse_caption(caption)
    if p['is_healthy']:
        return f"a photo of healthy {p['crop']} leaf"
    return f"a photo of {p['crop']} leaf with {p['disease']}"


# ── Strategy 2: Expert ────────────────────────────────────────────
def make_expert_prompt(caption):
    """Domain expert phrasing. Moderate accuracy."""
    p = parse_caption(caption)
    if p['is_healthy']:
        return (f"a photograph of a healthy {p['crop']} plant leaf "
                f"showing normal green coloration with no disease symptoms")
    return (f"a close-up photograph of {p['crop']} leaf "
            f"showing symptoms of {p['disease']} disease infection")


# ── Strategy 3: Metadata-enriched (BEST) ─────────────────────────
def make_meta_prompt(caption):
    """
    Metadata-enriched prompt with symptom descriptions.
    Best performing strategy — outperforms all others.

    CLIP's text encoder responds better to descriptive captions
    (matching its internet training distribution) than to short labels.
    """
    p = parse_caption(caption)
    if p['is_healthy']:
        return (f"A photograph of a healthy {p['crop']} leaf. "
                f"The leaf appears vibrant green with no visible disease symptoms. "
                f"Normal leaf texture and color throughout.")
    symptoms = p['symptoms'][:120] if p['symptoms'] else 'visible disease symptoms'
    return (f"A field photograph of a {p['crop']} leaf infected with {p['disease']}. "
            f"Visible symptoms include: {symptoms}. "
            f"Agricultural crop disease detection image.")


# ── Strategy 4: Ensemble ──────────────────────────────────────────
def make_ensemble_prompts(caption):
    """
    Returns list of 3 prompt variants for ensemble averaging.
    Ensemble = mean of all 3 embeddings before classification.
    """
    return [
        make_simple_prompt(caption),
        make_expert_prompt(caption),
        make_meta_prompt(caption),
    ]
