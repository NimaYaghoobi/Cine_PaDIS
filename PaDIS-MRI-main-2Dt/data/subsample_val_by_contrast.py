# This script exists to construct a subsampled T1+FLAIR validation set. Specifically, with 25 T1 samples and 7 FLAIR samples.

import os
import shutil
import argparse
import random
import torch


T1_NAME_PREFIXES = ("sample_t1_", "sample_t1pre_", "sample_t1post_")
FLAIR_NAME_PREFIX = "sample_flair_"

def _contrast_from_name(fname: str):
    name = fname.lower()
    if name.startswith(FLAIR_NAME_PREFIX):
        return "flair"
    if name.startswith("sample_t1pre_"):
        return "t1pre"
    if name.startswith("sample_t1post_"):
        return "t1post"
    if name.startswith("sample_t1_"):
        return "t1"
    return None

def _contrast_from_metadata(path: str):
    try:
        data = torch.load(path, map_location="cpu")
    except Exception:
        return None

    for k in ("contrast", "tag"):
        v = data.get(k, None)
        if isinstance(v, str):
            v = v.lower()
            if v in ("t1", "t1pre", "t1post", "flair"):
                return v

    meta = data.get("meta", {})
    if isinstance(meta, dict):
        v = meta.get("contrast", None)
        if isinstance(v, str):
            v = v.lower()
            if v in ("t1", "t1pre", "t1post", "flair"):
                return v

    series = data.get("series_description", None)
    if isinstance(series, str):
        s = series.upper()
        if "FLAIR" in s: return "flair"
        if "T1POST" in s or "POST" in s: return "t1post"
        if "T1PRE" in s or "PRE" in s: return "t1pre"
        if "T1" in s: return "t1"

    return None

def detect_contrast(input_dir: str, fname: str, prefer_name: bool = True, fallback_to_meta: bool = True):
    if prefer_name:
        c = _contrast_from_name(fname)
        if c is not None:
            return c
    if fallback_to_meta:
        return _contrast_from_metadata(os.path.join(input_dir, fname))
    return None

# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="Sample and rename validation set for brain MRI with contrast counts")
    p.add_argument("--input_dir", "-i", required=True, help="Directory containing .pt files")
    p.add_argument("--output_dir", "-o", default="val_t1-flair_subsamp", help="Directory to copy and rename sampled files into")
    p.add_argument("--snr", type=float, default=32.0, help="SNR level (dB) for folder naming")

    p.add_argument("--flair-count", type=int, default=7, help="Number of FLAIR samples")
    p.add_argument("--t1-count", type=int, default=25, help="Number of T1 samples (generic pool)")
    p.add_argument("--t1pre-count", type=int, default=0, help="Number of T1PRE samples")
    p.add_argument("--t1post-count", type=int, default=0, help="Number of T1POST samples")

    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--prefer-name", action="store_true", default=True, help="Prefer filename to detect contrast (default True)")
    p.add_argument("--no-fallback-meta", action="store_true", help="Disable metadata fallback (default: fallback enabled)")
    p.add_argument("--legacy-order", action="store_true",
                   help="Use unsorted filesystem order before sampling (not recommended).")

    args = p.parse_args()

    rng = random.Random(args.seed)

    all_files = [f for f in os.listdir(args.input_dir) if f.endswith(".pt") and f.startswith("sample_")]
    if not args.legacy_order:
        all_files.sort()  

    flair_pool, t1pre_pool, t1post_pool, t1_pool = [], [], [], []
    for fname in all_files:
        c = detect_contrast(
            input_dir=args.input_dir,
            fname=fname,
            prefer_name=args.prefer_name,
            fallback_to_meta=(not args.no_fallback_meta),
        )
        if c == "flair":
            flair_pool.append(fname)
        elif c == "t1pre":
            t1pre_pool.append(fname)
        elif c == "t1post":
            t1post_pool.append(fname)
        elif c == "t1":
            t1_pool.append(fname)
        else:
            # Unknown contrast; skip silently (or log if you prefer)
            pass

    need_flair = args.flair_count
    need_t1pre = args.t1pre_count
    need_t1post = args.t1post_count
    need_t1_base = args.t1_count

    if len(flair_pool) < need_flair:
        raise ValueError(f"Not enough FLAIR files: found {len(flair_pool)}, need {need_flair}")
    if len(t1pre_pool) < need_t1pre:
        raise ValueError(f"Not enough T1PRE files: found {len(t1pre_pool)}, need {need_t1pre}")
    if len(t1post_pool) < need_t1post:
        raise ValueError(f"Not enough T1POST files: found {len(t1post_pool)}, need {need_t1post}")


    sampled = []

    sampled_flair  = rng.sample(flair_pool, need_flair) if need_flair > 0 else []
    sampled += sampled_flair

    sampled_t1pre  = rng.sample(t1pre_pool, need_t1pre) if need_t1pre > 0 else []
    sampled += sampled_t1pre

    sampled_t1post = rng.sample(t1post_pool, need_t1post) if need_t1post > 0 else []
    sampled += sampled_t1post

    rem_t1pre  = [f for f in t1pre_pool  if f not in sampled_t1pre]
    rem_t1post = [f for f in t1post_pool if f not in sampled_t1post]

    sampled_t1 = []
    if need_t1_base > 0:
        if len(t1_pool) >= need_t1_base:
            sampled_t1 = rng.sample(t1_pool, need_t1_base)
        else:
            sampled_t1 = list(t1_pool)  # take all we have
            still_need = need_t1_base - len(sampled_t1)
            # fall back to remaining subtype pools (concat + sample)
            fallback_pool = rem_t1pre + rem_t1post
            if len(fallback_pool) < still_need:
                raise ValueError(
                    f"Not enough total T1 (including subtypes) to satisfy t1_count={need_t1_base}. "
                    f"Have plain T1={len(t1_pool)}, remaining subtype pool={len(fallback_pool)}."
                )
            sampled_t1 += rng.sample(fallback_pool, still_need)
    sampled += sampled_t1

    outdir = os.path.join(args.output_dir, f"{int(args.snr)}dB")
    os.makedirs(outdir, exist_ok=True)

    sampled_sorted = sorted(sampled)

    mapping_path = os.path.join(outdir, 'mapping.txt')
    with open(mapping_path, 'w') as mapf:
        for idx, fname in enumerate(sampled_sorted):
            new_name = f"sample_{idx}.pt"
            src = os.path.join(args.input_dir, fname)
            dst = os.path.join(outdir, new_name)
            shutil.copy(src, dst)
            mapf.write(f"{new_name} <- {fname}\n")
            print(f"Copied and renamed {fname} -> {new_name}")

    print(f"Done. Wrote {len(sampled_sorted)} files.")
    print(f"Mapping: {mapping_path}")

if __name__ == "__main__":
    main()
