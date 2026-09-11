"""Charades RGB-clip dataset for VideoMAE V2 fine-tuning.

VideoMAE V2 needs only RGB, as a single clip tensor of shape [C, T, H, W].
Rather than download the 55 GB Charades mp4 archive, we REUSE the compact
RGB subset already produced by `preprocess.py` / `build_subset.sh`:

    <root>/rgb_frames/<id>/<id>-NNNNNN.jpg

Each item returns (clip, label):
    clip  : FloatTensor [3, num_frames, img_size, img_size]  (ImageNet-normalized)
    label : FloatTensor [157] multi-hot

Labels and split protocols are shared with the two-stream pipeline
(see annotations.py). split: 'train' | 'val' | 'test'.

Note on frame count: the default subset stores 16 frames/video, which matches
VideoMAE V2's default `num_frames=16`. If you rebuild the subset with more
frames, `num_frames` evenly-spaced frames are sampled from what is available.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import annotations

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CharadesVideoClips(Dataset):
    def __init__(self, root, annot_dir, split, num_frames=16, img_size=224,
                 train=True, protocol="train_test",
                 num_segment=1, num_crop=1):
        self.root = root
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train
        # Multi-view eval (VideoMAE / VideoGLUE style). Training ignores these.
        self.num_segment = max(1, int(num_segment))
        self.num_crop = max(1, int(num_crop))

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

        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def _has_data(self, vid):
        d = os.path.join(self.root, "rgb_frames", vid)
        return os.path.isdir(d) and bool(os.listdir(d))

    @staticmethod
    def _segment_indices(n_avail, n_want, seg_idx, num_segment):
        """Temporal multi-segment indices across available frames."""
        if n_avail <= 0:
            return []
        if n_avail < n_want or num_segment <= 1:
            idx = np.linspace(0, n_avail - 1, n_want)
            return np.round(idx).astype(int).clip(0, n_avail - 1).tolist()
        # Split the video into num_segment contiguous parts; sample n_want in each.
        part = n_avail / num_segment
        lo = int(seg_idx * part)
        hi = max(lo, int((seg_idx + 1) * part) - 1)
        hi = min(hi, n_avail - 1)
        idx = np.linspace(lo, hi, n_want)
        return np.round(idx).astype(int).clip(0, n_avail - 1).tolist()

    def _pick(self, files, n, seg_idx=0):
        """Choose n frame paths (train: jittered global; eval: per-segment)."""
        t = len(files)
        if t == 0:
            return []
        if self.train:
            if t < n:
                files = (files * (n // t + 1))[:n]
                t = len(files)
            idx = np.linspace(0, t - 1, n, dtype=int)
            idx = np.clip(idx + np.random.randint(-2, 3, size=n), 0, t - 1)
            return [files[i] for i in idx]
        idxs = self._segment_indices(t, n, seg_idx, self.num_segment)
        return [files[i] for i in idxs]

    def _geom(self, crop_idx=0):
        """Shared crop geometry for every frame in the clip (temporal alignment)."""
        s = self.img_size
        if self.train:
            big = int(round(s * (s + 32) / s))
            top = np.random.randint(0, big - s + 1)
            left = np.random.randint(0, big - s + 1)
            flip = np.random.rand() < 0.5
            return big, top, left, flip
        if self.num_crop <= 1:
            return s, 0, 0, False
        # Three horizontal crops on a 256-resized square (VideoMAE-style).
        big = int(round(s * 256 / 224))
        max_off = big - s
        lefts = [0, max_off // 2, max_off] if self.num_crop == 3 else (
            [int(round(i * max_off / (self.num_crop - 1))) for i in range(self.num_crop)]
        )
        left = lefts[crop_idx % len(lefts)]
        top = max_off // 2
        return big, top, left, False

    def _load_frame(self, path, big, top, left, flip):
        img = Image.open(path).convert("RGB").resize((big, big), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        t = t[:, top:top + self.img_size, left:left + self.img_size]
        if flip:
            t = torch.flip(t, dims=[2])
        return (t - self.mean) / self.std

    def _clip(self, vid, seg_idx=0, crop_idx=0):
        d = os.path.join(self.root, "rgb_frames", vid)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
        g = self._geom(crop_idx)
        frames = [self._load_frame(os.path.join(d, f), *g)
                  for f in self._pick(files, self.num_frames, seg_idx=seg_idx)]
        clip = torch.stack(frames)            # [T, 3, H, W]
        return clip.permute(1, 0, 2, 3)       # [3, T, H, W]

    def get_views(self, i):
        """All (segment × crop) clips for video i — used by multi-view eval."""
        vid = self.ids[i]
        label = torch.from_numpy(self.labels[i])
        views = []
        n_seg = 1 if self.train else self.num_segment
        n_crop = 1 if self.train else self.num_crop
        try:
            for s in range(n_seg):
                for c in range(n_crop):
                    views.append(self._clip(vid, seg_idx=s, crop_idx=c))
        except Exception:  # noqa: BLE001
            z = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
            views = [z] * (n_seg * n_crop)
        return views, label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        views, label = self.get_views(i)
        return views[0], label