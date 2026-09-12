"""Bonus Module E: image-caption consistency with OpenCLIP (generic captions only, never claims about people/events).

Score = cosine similarity between CLIP image and text embeddings. We report the raw similarity and its rank
against a bank of generic distractor captions, and map it to a band with documented heuristic thresholds
(CLIP ViT-B/32 matching pairs are typically 0.25-0.35; unrelated pairs 0.10-0.18).
"""
from __future__ import annotations

import os

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain-HTTP weight download (xet stalled on some networks)

DISTRACTORS = [
    "a photo of a dog", "a city street at night", "a bowl of fresh fruit", "a mountain landscape", "a portrait of a person",
    "a red sports car", "a plate of pasta", "a wooden chair in a room", "a beach with palm trees", "a close-up of a flower",
    "a laptop on a desk", "a cat sleeping on a sofa", "an abstract painting", "a group of people at a party", "a bicycle leaning on a wall",
    "a glass of water", "a snowy forest", "a smartphone product shot", "a bird on a branch", "a bridge over a river",
]


class CaptionChecker:
    def __init__(self, device: torch.device | str = "cpu", model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            self._distractor_emb = self._text_emb(DISTRACTORS)

    @torch.no_grad()
    def _text_emb(self, texts: list[str]) -> torch.Tensor:
        t = self.tokenizer(texts).to(self.device)
        e = self.model.encode_text(t).float()
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def check(self, img: Image.Image, caption: str) -> dict:
        caption = caption.strip()[:300]
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        ie = self.model.encode_image(x).float()
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = self._text_emb([caption])
        sim = float((ie @ te.T)[0, 0])
        d_sims = (ie @ self._distractor_emb.T)[0]
        rank = int((d_sims > sim).sum().item()) + 1  # 1 = caption beats every distractor
        if sim >= 0.28:
            band, word = "consistent", "matches"
        elif sim >= 0.21:
            band, word = "weak", "only loosely matches"
        else:
            band, word = "inconsistent", "does not match"
        sentence = (f"Caption check: the text \"{caption}\" {word} the image content (CLIP similarity {sim:.2f}; "
                    f"ranked {rank} of {len(DISTRACTORS) + 1} against generic captions). "
                    + ("A mismatch between caption and content is a separate authenticity risk (misleading framing), independent of whether the pixels are synthetic." if band != "consistent" else "Consistency does not prove authenticity - it only means the caption describes what is shown."))
        return {"caption": caption, "similarity": round(sim, 4), "rank_vs_distractors": rank, "n_distractors": len(DISTRACTORS),
                "band": band, "sentence": sentence, "model": "OpenCLIP ViT-B/32 (openai weights)"}
