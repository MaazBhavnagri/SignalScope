import csv, os, random
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from tqdm.auto import tqdm

# We will put data at root of repo
PROCESSED = Path('data/processed')
PROCESSED.mkdir(parents=True, exist_ok=True)
MANIFEST = Path('data/manifest.csv')
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
SEED = 1337
random.seed(SEED)

GEN_MAP = {
    'stable_diffusion_v_1_4': 'SD14', 'stable_diffusion_v_1_5': 'SD15',
    'ADM': 'ADM', 'GLIDE': 'GLIDE', 'Midjourney': 'Midjourney',
    'VQDM': 'VQDM', 'Wukong': 'Wukong', 'BigGAN': 'BigGAN',
    'real': 'Real', 'REAL': 'Real',
    0: 'Real', 1: 'ADM', 2: 'BigGAN', 3: 'GLIDE', 4: 'Midjourney',
    5: 'SD14', 6: 'SD15', 7: 'VQDM', 8: 'Wukong'
}
HELD_OUT = {'Midjourney', 'VQDM'}

rows = []

def save_img(img, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        img.convert('RGB').save(dest, 'JPEG', quality=95)

# ── CIFAKE: stream directly, no list() in memory ──
print('Loading CIFAKE (streaming)...')
CIFAKE_PER_CLASS = 10000  # reduced to save RAM - still 20k total
real_count, fake_count = 0, 0
val_real_count, val_fake_count = 0, 0

for split_tag, max_per in [('train', CIFAKE_PER_CLASS), ('test', 1000)]:
    ds = load_dataset('dragonintelligence/CIFAKE-image-dataset', split=split_tag, streaming=False)
    r_count, f_count = 0, 0
    out_split = 'train' if split_tag == 'train' else 'val'
    for i, item in enumerate(tqdm(ds, desc=f'CIFAKE {split_tag}')):
        lbl = item['label']
        if lbl == 0 and r_count >= max_per: continue
        if lbl == 1 and f_count >= max_per: continue
        gen = 'Real' if lbl == 0 else 'SD14'
        fname = PROCESSED / out_split / f'cifake_{split_tag}_{i}_{lbl}.jpg'
        save_img(item['image'], fname)
        rows.append({'path': str(fname), 'label': lbl, 'generator': gen, 'source': 'cifake', 'split': out_split})
        if lbl == 0: r_count += 1
        else: f_count += 1
        if r_count >= max_per and f_count >= max_per: break
    del ds

print(f'CIFAKE done: {len(rows)} rows')

# ── Tiny-GenImage: stream one by one ──
print('Loading Tiny-GenImage (streaming)...')
try:
    genimg = load_dataset('TheKernel01/Tiny-GenImage', split='train', streaming=False)
    for i, item in enumerate(tqdm(genimg, desc='Tiny-GenImage')):
        gen_name = GEN_MAP.get(item.get('generator', item.get('label_str', '')), None)
        if gen_name is None: continue
        lbl = 0 if gen_name == 'Real' else 1
        if gen_name in HELD_OUT:
            split = 'test'
        elif i % 10 == 0:
            split = 'test'
        elif i % 10 == 1:
            split = 'val'
        else:
            split = 'train'
        fname = PROCESSED / split / f'genimg_{i}_{lbl}.jpg'
        save_img(item['image'], fname)
        rows.append({'path': str(fname), 'label': lbl, 'generator': gen_name, 'source': 'tiny_genimage', 'split': split})
    del genimg
    print(f'Tiny-GenImage done, total rows: {len(rows)}')
except Exception as e:
    print(f'Tiny-GenImage failed ({e}), continuing with CIFAKE only')

# ── Write manifest ──
with open(MANIFEST, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['path','label','generator','source','split'])
    w.writeheader(); w.writerows(rows)

from collections import Counter
splits = Counter(r['split'] for r in rows)
print('Manifest written:', dict(splits))
