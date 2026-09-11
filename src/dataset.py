"""Multi-label Charades dataset for the two-stream network.

Reads the compact subset produced by preprocess.py:
    <root>/rgb_frames/<id>/<id>-NNNNNN.jpg
    <root>/flow_frames/<id>/<id>-NNNNNNx.jpg , <id>-NNNNNNy.jpg

Each item returns (rgb, flow, label):
    rgb   : FloatTensor [num_rgb, 3, H, W]              (0 if load_rgb=False)
    flow  : FloatTensor [2*num_flow, H, W]              (0 if load_flow=False)
    label : FloatTensor [157] multi-hot

split: 'train' | 'val' | 'test'  (meaning depends on --protocol; see annotations.py).
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import annotations


def _img_norm(t, mean, std):
    return (t - mean) / std


class CharadesTwoStream(Dataset):
    def __init__(self, root, annot_dir, split, img_size=224,
                 num_rgb=8, num_flow=10, train=True,
                 load_rgb=True, load_flow=True, protocol="train_test"):
        self.root = root
        self.img_size = img_size
        self.num_rgb = num_rgb
        self.num_flow = num_flow
        self.use_all_rgb = num_rgb <= 0
        self.use_all_flow = num_flow <= 0
        self.train = train
        self.load_rgb = load_rgb
        self.load_flow = load_flow

        ids, labels = annotations.build_labels(annot_dir, split, protocol=protocol)
        keep = [i for i, v in enumerate(ids) if self._has_data(v)]
        self.ids = [ids[i] for i in keep]
        self.labels = labels[keep]
        print(f"[{split}|{protocol}] usable videos: {len(self.ids)} / {len(ids)}", flush=True)
        if not self.ids:
            raise RuntimeError(
                f"No usable videos for split={split} under {root}. "
                "Did build_subset.sh finish and was the tar extracted here?"
            )

        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # ---- availability check ----
    def _has_data(self, vid):
        ok = True
        if self.load_rgb:
            d = os.path.join(self.root, "rgb_frames", vid)
            ok = ok and os.path.isdir(d) and bool(os.listdir(d))
        if self.load_flow:
            d = os.path.join(self.root, "flow_frames", vid)
            ok = ok and os.path.isdir(d) and bool(os.listdir(d))
        return ok

    # ---- sampling helper ----
    def _pick(self, files, n):
        t = len(files)
        if t == 0:
            return []
        if n <= 0:
            return files
        if t < n:
            files = (files * (n // t + 1))[:n]
            t = len(files)
        idx = np.linspace(0, t - 1, n, dtype=int)
        if self.train:
            idx = np.clip(idx + np.random.randint(-2, 3, size=n), 0, t - 1)
        return [files[i] for i in idx]

    # ---- geometry shared by rgb + flow so they stay aligned ----
    def _geom(self):
        s = self.img_size
        if self.train:
            scale = (s + 32) / s
            big = int(round(s * scale))
            top = np.random.randint(0, big - s + 1)
            left = np.random.randint(0, big - s + 1)
            flip = np.random.rand() < 0.5
            return big, top, left, flip
        return s, 0, 0, False

    def _load_img(self, path, mode, big, top, left, flip):
        img = Image.open(path).convert(mode).resize((big, big), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        t = t.permute(2, 0, 1) if mode == "RGB" else t.unsqueeze(0)
        t = t[:, top:top + self.img_size, left:left + self.img_size]
        if flip:
            t = torch.flip(t, dims=[2])
        return t

    def _rgb(self, vid, g):
        d = os.path.join(self.root, "rgb_frames", vid)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
        n = 0 if self.use_all_rgb else self.num_rgb
        frames = [self._load_img(os.path.join(d, f), "RGB", *g)
                  for f in self._pick(files, n)]
        x = torch.stack(frames)                       # [T,3,H,W]
        return _img_norm(x, self.rgb_mean, self.rgb_std)

    def _flow(self, vid, g):
        d = os.path.join(self.root, "flow_frames", vid)
        files = sorted(os.listdir(d))
        xs = [f for f in files if f.endswith("x.jpg")]
        ys = [f for f in files if f.endswith("y.jpg")]
        n = min(len(xs), len(ys))
        pairs = list(zip(xs[:n], ys[:n]))
        chans = []
        n = 0 if self.use_all_flow else self.num_flow
        for fx, fy in self._pick(pairs, n):
            chans.append(self._load_img(os.path.join(d, fx), "L", *g))
            chans.append(self._load_img(os.path.join(d, fy), "L", *g))
        x = torch.cat(chans, dim=0)                   # [2*num_flow,H,W]
        return (x - 0.5) / 0.5

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vid = self.ids[i]
        g = self._geom()
        label = torch.from_numpy(self.labels[i])
        try:
            rgb = self._rgb(vid, g) if self.load_rgb else torch.zeros(1)
            flow = self._flow(vid, g) if self.load_flow else torch.zeros(1)
            return rgb, flow, label
        except Exception:  # noqa: BLE001 -- never kill an epoch over one bad file
            nr = max(self.num_rgb, 1) if not self.use_all_rgb else 8
            nf = max(self.num_flow, 1) if not self.use_all_flow else 10
            rgb = torch.zeros(nr, 3, self.img_size, self.img_size) if self.load_rgb else torch.zeros(1)
            flow = torch.zeros(2 * nf, self.img_size, self.img_size) if self.load_flow else torch.zeros(1)
            return rgb, flow, label
