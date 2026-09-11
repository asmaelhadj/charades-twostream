"""Params / GFLOPs / latency helpers for deployment discussions."""
import time

import torch

# Official VideoMAE V2 numbers at 16×224² (single view), from MODEL_ZOO / papers.
_GFLOPS_16 = {
    "vit_small_patch16_224": 57.0,
    "vit_base_patch16_224": 180.0,
    "vit_large_patch16_224": 597.0,
    "vit_huge_patch16_224": 1192.0,
}


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_gflops(model_name, num_frames=16, streams=1):
    """Rough GFLOPs/clip. Attention ~scales with tokens²; tokens ∝ frames."""
    base = _GFLOPS_16.get(model_name)
    if base is None:
        return None
    r = max(1, int(num_frames)) / 16.0
    # Conservative: treat dominant cost as quadratic in temporal tokens.
    return base * (r ** 2) * max(1, int(streams))


@torch.no_grad()
def benchmark_latency(model, example, device, warmup=10, reps=50):
    """Median forward latency in ms for one example batch (already on device)."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(*example) if isinstance(example, (tuple, list)) else model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(*example) if isinstance(example, (tuple, list)) else model(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "p90_ms": times[int(0.9 * (len(times) - 1))],
    }


def summarize_cost(model_name, num_frames, streams=1, params=None, latency=None):
    gflops = estimate_gflops(model_name, num_frames, streams=streams)
    out = {
        "model": model_name,
        "num_frames": num_frames,
        "streams": streams,
        "params": params,
        "params_m": None if params is None else round(params / 1e6, 2),
        "gflops_est": None if gflops is None else round(gflops, 1),
    }
    if latency is not None:
        out.update(latency)
    return out
