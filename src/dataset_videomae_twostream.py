"""Charades RGB + optical-flow clips for two-stream VideoMAE V2 fine-tuning.

Reuses the compact subset from preprocess.py (same as the CNN two-stream jobs):

    <root>/rgb_frames/<id>/<id>-NNNNNN.jpg
    <root>/flow_frames/<id>/<id>-NNNNNNx.jpg , <id>-NNNNNNy.jpg

Each item returns (rgb_clip, flow_clip, label):
    rgb_clip  : FloatTensor [3, num_frames, H, W]  (ImageNet-normalized)
    flow_clip : FloatTensor [2, num_frames, H, W]  (normalized to [-1, 1])
    label     : FloatTensor [157] multi-hot
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import annotations

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CharadesVideoMAETwoStream(Dataset):
    def __init__(self, root, annot_dir, split, num_frames=16, img_size=224,
                 train=True, protocol="train_test",
                 num_segment=1, num_crop=1):
        self.root = root
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train
        self.num_segment = max(1, int(num_segment))
        self.num_crop = max(1, int(num_crop))

        ids, labels = annotations.build_labels(annot_dir, split, protocol=protocol)
        keep = [i for i, v in enumerate(ids) if self._has_data(v)]
        self.ids = [ids[i] for i in keep]
        self.labels = labels[keep]
        print(f"[{split}|{protocol}] twostream usable videos: "
              f"{len(self.ids)} / {len(ids)}", flush=True)
        if not self.ids:
            raise RuntimeError(
                f"No usable RGB+flow videos for split={split} under {root}. "
                "Did build_subset.sh finish and was the tar extracted here?"
            )

        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def _has_data(self, vid):
        rgb_d = os.path.join(self.root, "rgb_frames", vid)
        flow_d = os.path.join(self.root, "flow_frames", vid)
        if not os.path.isdir(rgb_d) or not os.listdir(rgb_d):
            return False
        if not os.path.isdir(flow_d) or not os.listdir(flow_d):
            return False
        xs = [f for f in os.listdir(flow_d) if f.endswith("x.jpg")]
        ys = [f for f in os.listdir(flow_d) if f.endswith("y.jpg")]
        return bool(xs) and bool(ys)

    @staticmethod
    def _segment_indices(n_avail, n_want, seg_idx, num_segment):
        if n_avail <= 0:
            return []
        if n_avail < n_want or num_segment <= 1:
            base = np.linspace(0, n_avail - 1, n_want)
            return np.round(base).astype(int).clip(0, n_avail - 1).tolist()
        part = n_avail / num_segment
        lo = int(seg_idx * part)
        hi = max(lo, int((seg_idx + 1) * part) - 1)
        hi = min(hi, n_avail - 1)
        idx = np.linspace(lo, hi, n_want)
        return np.round(idx).astype(int).clip(0, n_avail - 1).tolist()

    def _pick_indices(self, n_avail, n_want, seg_idx=0):
        if n_avail == 0:
            return []
        if self.train:
            if n_avail < n_want:
                base = np.linspace(0, n_avail - 1, n_want)
                return np.round(base).astype(int).tolist()
            idx = np.linspace(0, n_avail - 1, n_want, dtype=int)
            idx = np.clip(idx + np.random.randint(-2, 3, size=n_want), 0, n_avail - 1)
            return idx.tolist()
        return self._segment_indices(n_avail, n_want, seg_idx, self.num_segment)

    def _geom(self, crop_idx=0):
        s = self.img_size
        if self.train:
            big = int(round(s * (s + 32) / s))
            top = np.random.randint(0, big - s + 1)
            left = np.random.randint(0, big - s + 1)
            flip = np.random.rand() < 0.5
            return big, top, left, flip
        if self.num_crop <= 1:
            return s, 0, 0, False
        big = int(round(s * 256 / 224))
        max_off = big - s
        lefts = [0, max_off // 2, max_off] if self.num_crop == 3 else (
            [int(round(i * max_off / (self.num_crop - 1))) for i in range(self.num_crop)]
        )
        left = lefts[crop_idx % len(lefts)]
        top = max_off // 2
        return big, top, left, False

    def _load_rgb(self, path, big, top, left, flip):
        img = Image.open(path).convert("RGB").resize((big, big), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        t = t[:, top:top + self.img_size, left:left + self.img_size]
        if flip:
            t = torch.flip(t, dims=[2])
        return (t - self.mean) / self.std

    def _load_flow_gray(self, path, big, top, left, flip):
        img = Image.open(path).convert("L").resize((big, big), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)
        t = t[:, top:top + self.img_size, left:left + self.img_size]
        if flip:
            t = torch.flip(t, dims=[2])
        return t

    def _flow_pairs(self, vid):
        d = os.path.join(self.root, "flow_frames", vid)
        files = sorted(os.listdir(d))
        xs = [f for f in files if f.endswith("x.jpg")]
        ys = [f for f in files if f.endswith("y.jpg")]
        n = min(len(xs), len(ys))
        return list(zip(xs[:n], ys[:n]))

    def _clips(self, vid, seg_idx=0, crop_idx=0):
        rgb_d = os.path.join(self.root, "rgb_frames", vid)
        rgb_files = sorted(f for f in os.listdir(rgb_d) if f.lower().endswith(".jpg"))
        flow_pairs = self._flow_pairs(vid)

        n = min(len(rgb_files), len(flow_pairs))
        if n == 0:
            raise RuntimeError(f"no aligned rgb/flow for {vid}")

        idxs = self._pick_indices(n, self.num_frames, seg_idx=seg_idx)
        g = self._geom(crop_idx)
        rgb_frames, flow_frames = [], []
        for i in idxs:
            ii = min(i, n - 1)
            rgb_frames.append(self._load_rgb(os.path.join(rgb_d, rgb_files[ii]), *g))
            fx, fy = flow_pairs[ii]
            flow_d = os.path.join(self.root, "flow_frames", vid)
            dx = self._load_flow_gray(os.path.join(flow_d, fx), *g)
            dy = self._load_flow_gray(os.path.join(flow_d, fy), *g)
            flow_frames.append(torch.cat([dx, dy], dim=0))  # [2, H, W]

        rgb = torch.stack(rgb_frames).permute(1, 0, 2, 3)    # [3, T, H, W]
        flow = torch.stack(flow_frames).permute(1, 0, 2, 3)  # [2, T, H, W]
        flow = (flow - 0.5) / 0.5                            # [-1, 1], matches CNN stream
        return rgb, flow

    def get_views(self, i):
        """All (segment × crop) (rgb, flow) pairs for video i."""
        vid = self.ids[i]
        label = torch.from_numpy(self.labels[i])
        n_seg = 1 if self.train else self.num_segment
        n_crop = 1 if self.train else self.num_crop
        views = []
        try:
            for s in range(n_seg):
                for c in range(n_crop):
                    views.append(self._clips(vid, seg_idx=s, crop_idx=c))
        except Exception:  # noqa: BLE001
            z = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
            f = torch.zeros(2, self.num_frames, self.img_size, self.img_size)
            views = [(z, f)] * (n_seg * n_crop)
        return views, label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        views, label = self.get_views(i)
        rgb, flow = views[0]
        return rgb, flow, label
