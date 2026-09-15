from __future__ import annotations
import csv, io
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from model import config as C

def load_manifest(split=None, sources=None):
    if not C.MANIFEST_PATH.exists():
        return []
    rows = []
    with open(C.MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if split and r.get("split") != split: continue
            if sources and r.get("source") not in sources: continue
            row = dict(r)
            row["label"] = int(row["label"])
            rows.append(row)
    return rows

def balanced_sample_weights(rows):
    labels = [int(r["label"]) for r in rows]
    n0 = labels.count(0) or 1; n1 = labels.count(1) or 1
    w = {0: 1.0/n0, 1: 1.0/n1}
    return [w[l] for l in labels]

def train_transform(img_size=224):
    return T.Compose([T.Resize((img_size+32, img_size+32)), T.RandomCrop(img_size),
        T.RandomHorizontalFlip(), T.ColorJitter(0.2,0.2,0.1,0.05), T.RandomGrayscale(p=0.05),
        T.ToTensor(), T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1,1.5))], p=0.3)])

def eval_transform(img_size=224):
    return T.Compose([T.Resize((img_size,img_size)), T.ToTensor(), T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)])

def to_input_tensor(img, img_size=224):
    return eval_transform(img_size)(img.convert("RGB")).unsqueeze(0)

def jpeg_recompress(img, quality=75):
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=quality); buf.seek(0)
    return Image.open(buf).convert("RGB")

def rescale(img, factor=0.5):
    w, h = img.size
    small = img.resize((max(1,int(w*factor)), max(1,int(h*factor))), Image.BICUBIC)
    return small.resize((w,h), Image.BICUBIC)

def gaussian_blur(img, sigma=1.0):
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def add_noise(img, std=5.0):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return Image.fromarray(np.clip(arr + np.random.normal(0,std,arr.shape),0,255).astype(np.uint8))

def screenshot(img, scale=0.8, quality=80):
    w, h = img.size
    small = img.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.BICUBIC)
    buf = io.BytesIO(); small.convert("RGB").save(buf, "JPEG", quality=quality); buf.seek(0)
    return Image.open(buf).convert("RGB").resize((w,h), Image.BICUBIC)

class ManifestDataset(Dataset):
    def __init__(self, rows, transform, degrade=None):
        self.rows = rows; self.transform = transform
        self.degrade = degrade; self.attr_index = C.ATTR_INDEX
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        if self.degrade: img = self.degrade(img)
        x = self.transform(img)
        y = torch.tensor(int(r["label"]), dtype=torch.float32)
        attr = torch.tensor(self.attr_index.get(r.get("generator",""), C.ATTR_IGNORE), dtype=torch.long)
        return x, y, attr, r["path"]
