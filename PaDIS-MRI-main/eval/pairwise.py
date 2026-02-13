#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

METRICS = ["psnr", "ssim", "nrmse"]  # keys expected in results.json

def load_per_image(path: Path, tag: str):
    """
    Load one results.json and return:
      { f"{tag}:{image_id}": {"psnr":..., "ssim":..., "nrmse":...}, ... }
    If path is None or empty, return {}.
    """
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    per_image = data.get("per_image", {})
    out = {}
    for k, v in per_image.items():
        cid = f"{tag}:{str(k)}"
        out[cid] = {m: v[m] for m in METRICS if m in v}
    return out

def combine_method(method_cfg: dict, name: str):
    """
    Merge T2 and T1 (whichever exist) for a method section from the config.
    method_cfg may contain keys 't2' and/or 't1'. At least one must exist.
    """
    if not isinstance(method_cfg, dict):
        raise ValueError(f"Config for {name} must be an object with 't2' and/or 't1'.")
    d_t2 = load_per_image(method_cfg.get("t2"), "T2") if method_cfg.get("t2") else {}
    d_t1 = load_per_image(method_cfg.get("t1"), "T1") if method_cfg.get("t1") else {}
    if not d_t1 and not d_t2:
        raise ValueError(f"{name}: neither 't1' nor 't2' provided.")
    return {**d_t2, **d_t1}

def paired_stats(A: dict, B: dict, metric: str):
    """
    Return (N, mean_diff, std_diff) for paired differences on 'metric'.
    Uses population std (divide by N). Intersection over composite IDs.
    """
    shared = [i for i in A.keys() if i in B and metric in A[i] and metric in B[i]]
    N = len(shared)
    if N == 0:
        return 0, float("nan"), float("nan")
    diffs = [A[i][metric] - B[i][metric] for i in shared]
    mean_diff = sum(diffs) / N
    var = sum((d - mean_diff) ** 2 for d in diffs) / N
    return N, mean_diff, math.sqrt(var)

def label_metric(m): return m.upper() if m != "nrmse" else "NRMSE"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Build merged dicts per method, allowing only T1 or only T2
    alg = {
        "PaDIS": combine_method(cfg.get("PaDIS", {}), "PaDIS"),
        "EDM":   combine_method(cfg.get("EDM",   {}), "EDM"),
        "PICS":  combine_method(cfg.get("PICS",  {}), "PICS"),
    }

    # Quick counts
    for name, d in alg.items():
        t2_n = sum(1 for k in d if k.startswith("T2:"))
        t1_n = sum(1 for k in d if k.startswith("T1:"))
        print(f"{name}: {len(d)} composite IDs  (T2={t2_n}, T1={t1_n})")

    pairs = [("PaDIS", "EDM"), ("PaDIS", "PICS"), ("EDM", "PICS")]

    print("\nPairwise std of paired differences (population std)")
    print("Intersection over available IDs per pair; also printing mean_diff.\n")

    for a, b in pairs:
        print(f"=== {a} vs {b} ===")
        for m in METRICS:
            N, mean_diff, std_diff = paired_stats(alg[a], alg[b], m)
            print(f"{label_metric(m):<6}  N={N:<3d}  mean_diff={mean_diff:>9.4f}  std_diff={std_diff:>9.4f}")
        print()

    # Shared across all three (informational)
    all3 = set(alg["PaDIS"]) & set(alg["EDM"]) & set(alg["PICS"])
    print(f"Composite IDs shared across ALL THREE methods: {len(all3)}")

if __name__ == "__main__":
    main()
