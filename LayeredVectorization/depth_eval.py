"""Depth-estimator evaluation for SAM-segment back-to-front layer ordering.

The eval's purpose is utility-driven: confirm the cheapest depth model that
gives stable, well-separated per-segment medians. Depth is run on the
*original* RGB; SAM masks sample from it; per-mask median = layer-order key.

Built incrementally. This slice handles SAM mask caching only — run it,
inspect mask counts, then we layer in depth inference next.
"""

import argparse
import csv
import glob
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_REGISTRY = {
    "dpt_base":  {"hf_id": "Intel/dpt-beit-base-384",                   "larger_is_closer": True},
    "da_base":   {"hf_id": "LiheYoung/depth-anything-base-hf",          "larger_is_closer": True},
    "dav2_base": {"hf_id": "depth-anything/Depth-Anything-V2-Base-hf",  "larger_is_closer": True},
    "glpn":      {"hf_id": "vinvino02/glpn-nyu",                        "larger_is_closer": False},
    "zoeDepth":  {"hf_id": "Intel/zoedepth-nyu-kitti",                  "larger_is_closer": False},
}

MIN_AREA_FRAC = 0.002  # drop SAM masks smaller than 0.2% of image area
VIS_MAX_MASKS = 12     # plot-only cap; metrics use every mask above MIN_AREA_FRAC


def load_sam_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["sam"]


def init_sam(device, sam_conf):
    sam = sam_model_registry[sam_conf["model_type"]](checkpoint=sam_conf["sam_checkpoint"])
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=sam_conf["points_per_side"],
        pred_iou_thresh=sam_conf["pred_iou_thresh"],
        stability_score_thresh=sam_conf["stability_score_thresh"],
        crop_n_layers=sam_conf["crop_n_layers"],
        crop_n_points_downscale_factor=sam_conf["crop_n_points_downscale_factor"],
        min_mask_region_area=sam_conf["min_mask_region_area"],
        box_nms_thresh=sam_conf["box_nms_thresh"],
    )


def run_sam(mask_generator, image_rgb):
    raw = mask_generator.generate(image_rgb)
    return [m["segmentation"].astype(bool) for m in raw]


def filter_by_area(masks, image_area, min_frac):
    min_pixels = int(image_area * min_frac)
    keep = [m for m in masks if int(m.sum()) >= min_pixels]
    keep.sort(key=lambda m: int(m.sum()), reverse=True)
    return keep


def load_cached_masks(cache_path):
    data = np.load(cache_path)
    return [data[k].astype(bool) for k in data.files]


def save_cached_masks(cache_path, masks):
    np.savez_compressed(cache_path, *[m.astype(np.uint8) for m in masks])


def load_depth_model(hf_id, device):
    processor = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModelForDepthEstimation.from_pretrained(hf_id).to(device).eval()
    return processor, model


def infer_depth(image_pil, processor, model, device, target_hw, larger_is_closer):
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth  # [1, H', W']
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)  # [1, 1, H', W']
    upsampled = F.interpolate(pred, size=target_hw, mode="bilinear", align_corners=False)
    depth = upsampled.squeeze().cpu().numpy().astype(np.float32)
    if larger_is_closer:
        depth = -depth  # normalize so larger value = farther
    return depth


def per_mask_stats(depth, masks):
    """Per-mask depth statistics. Returns list of dicts ordered as input.
    Convention: depth larger = farther, so sorting descending by median
    yields back-to-front order.
    """
    stats = []
    for m in masks:
        vals = depth[m]
        ys, xs = np.where(m)
        stats.append({
            "median":   float(np.median(vals)),
            "iqr":      float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "area":     int(m.sum()),
            "centroid": (float(ys.mean()), float(xs.mean())),
        })
    return stats


def sort_back_to_front(stats):
    """Return indices that sort masks farthest -> nearest."""
    return sorted(range(len(stats)), key=lambda i: stats[i]["median"], reverse=True)


TAU_IOU_THRESH = 0.3  # min IoU to treat a L>0 mask as the same region as an L0 mask


def _iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def kendall_tau(a, b):
    """Kendall's tau between two value lists. Ties are ignored."""
    n = len(a)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else float("nan")


def cross_level_tau(masks_lvl, stats_lvl, masks_l0, stats_l0,
                    iou_thresh=TAU_IOU_THRESH):
    """IoU-match each L>0 mask to its best L0 mask, then Kendall's tau between
    matched median depths. Reflects how well the model preserves the
    back-to-front order when boundaries shift under simplification.
    """
    if not masks_lvl or not masks_l0:
        return float("nan")
    matched_lvl, matched_l0 = [], []
    for m, s in zip(masks_lvl, stats_lvl):
        best_i, best_iou = -1, 0.0
        for i, m0 in enumerate(masks_l0):
            v = _iou(m, m0)
            if v > best_iou:
                best_iou, best_i = v, i
        if best_iou >= iou_thresh:
            matched_lvl.append(s["median"])
            matched_l0.append(stats_l0[best_i]["median"])
    return kendall_tau(matched_lvl, matched_l0)


def _render_diagnostic_row(ax_row, image_rgb, depth, masks, stats, row_label,
                           src_title, depth_title):
    """Render one row of the diagnostic strip into the supplied axes row.

    Columns: source image | depth heatmap | masks colored by depth value
             | progressive back-to-front composites.
    """
    order = sort_back_to_front(stats)
    masks_sorted = [masks[i] for i in order]
    stats_sorted = [stats[i] for i in order]
    K = len(masks_sorted)
    n_thumbs = len(ax_row) - 3

    ax_row[0].imshow(image_rgb)
    ax_row[0].set_title(f"{row_label}\n{src_title}", fontsize=10)
    ax_row[0].axis("off")

    ax_row[1].imshow(depth, cmap="turbo")
    ax_row[1].set_title(depth_title, fontsize=10)
    ax_row[1].axis("off")

    cmap = matplotlib.colormaps["turbo"]
    overlay = image_rgb.astype(np.float32) / 255.0
    d_lo, d_hi = float(depth.min()), float(depth.max())
    d_span = max(d_hi - d_lo, 1e-8)
    for s, m in zip(stats_sorted, masks_sorted):
        t = (s["median"] - d_lo) / d_span
        c = np.array(cmap(t))[:3]
        overlay[m] = 0.4 * overlay[m] + 0.6 * c
    ax_row[2].imshow(np.clip(overlay, 0.0, 1.0))
    for s in stats_sorted:
        cy, cx = s["centroid"]
        ax_row[2].text(cx, cy, f"{s['median']:+.2f}",
                       color="white", fontsize=6, ha="center", va="center",
                       bbox=dict(facecolor="black", alpha=0.55, pad=0.6, edgecolor="none"))
    ax_row[2].set_title("masks colored by median depth\n(red=far, blue=near)", fontsize=10)
    ax_row[2].axis("off")

    canvas = np.full_like(image_rgb, 255)
    stops = np.linspace(1, K, n_thumbs).round().astype(int).tolist() if K > 0 else []
    last_n = 0
    for col in range(n_thumbs):
        ax = ax_row[3 + col]
        if col < len(stops):
            n = stops[col]
            for k in range(last_n, n):
                m = masks_sorted[k]
                canvas[m] = image_rgb[m]
            last_n = n
            s = stats_sorted[n - 1]
            ax.imshow(canvas)
            ax.set_title(f"after {n}/{K}\nmed={s['median']:+.2f} iqr={s['iqr']:.2f}",
                         fontsize=8)
        else:
            ax.imshow(canvas)
            ax.set_title("", fontsize=8)
        ax.axis("off")


def render_multilevel_diagnostic(rows, out_path, model_key, image_name):
    """One PNG per (image, model). Each row corresponds to a simplification level.

    rows: list of dicts with keys: level, image_rgb, depth, masks, stats.
    """
    n_rows = len(rows)
    K_max = max((len(r["masks"]) for r in rows), default=0)
    n_thumbs = min(6, K_max) if K_max > 0 else 1
    n_cols = 3 + n_thumbs
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3.6 * n_rows),
                             squeeze=False)
    fig.suptitle(f"{model_key}  |  {image_name}", fontsize=14)

    for row_idx, r in enumerate(rows):
        _render_diagnostic_row(
            axes[row_idx], r["image_rgb"], r["depth"], r["masks"], r["stats"],
            row_label=f"L{r['level']}",
            src_title="(original)" if r["level"] == 0 else "(simplified)",
            depth_title="L0 depth (shared)\n(near=dark, far=bright)",
        )

    plt.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_cross_model_summary_csv(model_tables, out_path):
    """One CSV per image: per-level rows across all tested models, plus an
    'avg' row after each model summarizing its metrics across levels.

    Columns: model, level, n_masks, spread, mean_iqr, separability, tau.
    NaN tau (L0) is written as an empty cell; inf separability as 'inf'.
    The 'avg' row averages spread/mean_iqr/n_masks across all levels, and
    averages separability/tau across the levels where they are finite/defined.
    """
    if not model_tables:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "n_masks", "spread", "mean_iqr",
                    "separability", "tau"])
        for model_key, rows in model_tables.items():
            for r in rows:
                tau = "" if np.isnan(r["tau"]) else round(float(r["tau"]), 4)
                if np.isfinite(r["separability"]):
                    sep = round(float(r["separability"]), 4)
                else:
                    sep = "inf"
                w.writerow([
                    model_key,
                    r["level"],
                    r["n_masks"],
                    round(float(r["spread"]),   4),
                    round(float(r["mean_iqr"]), 4),
                    sep,
                    tau,
                ])

            spreads = [r["spread"]   for r in rows]
            iqrs    = [r["mean_iqr"] for r in rows]
            n_masks = [r["n_masks"]  for r in rows]
            seps    = [r["separability"] for r in rows if np.isfinite(r["separability"])]
            taus    = [r["tau"]          for r in rows if not np.isnan(r["tau"])]
            avg_sep = round(float(np.mean(seps)), 4) if seps else ""
            avg_tau = round(float(np.mean(taus)), 4) if taus else ""
            w.writerow([
                model_key,
                "avg",
                round(float(np.mean(n_masks)), 1),
                round(float(np.mean(spreads)), 4),
                round(float(np.mean(iqrs)),    4),
                avg_sep,
                avg_tau,
            ])


def aggregate_summaries_across_images(summaries_dir, out_path):
    """Combine all per-image summary CSVs in summaries_dir into one collapsed
    cross-image table: one row per model.

    For each image:
      - spread, mean_iqr, separability are taken from the L0 row (crisp masks,
        canonical model quality).
      - tau is averaged across L>0 (single ordering-stability number).
    The cross-image table is then the mean of those per-image values per model.
    'inf' separability values are excluded from their mean.
    """
    if not os.path.isdir(summaries_dir):
        return
    csv_paths = sorted(glob.glob(os.path.join(summaries_dir, "*.csv")))
    if not csv_paths:
        return

    per_image = {}  # model -> list of per-image dicts
    for p in csv_paths:
        rows_by_model = {}
        with open(p, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row["level"] == "avg":
                    continue
                rows_by_model.setdefault(row["model"], []).append(row)

        for model, rows in rows_by_model.items():
            l0 = next((r for r in rows if r["level"] == "0"), None)
            if l0 is None:
                continue
            sep_raw = l0["separability"]
            sep_val = float("inf") if sep_raw == "inf" else float(sep_raw)
            taus = [float(r["tau"]) for r in rows
                    if r["level"] != "0" and r["tau"] != ""]
            tau_avg = float(np.mean(taus)) if taus else float("nan")
            per_image.setdefault(model, []).append({
                "spread":       float(l0["spread"]),
                "mean_iqr":     float(l0["mean_iqr"]),
                "separability": sep_val,
                "tau":          tau_avg,
            })

    if not per_image:
        return

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_images", "spread", "mean_iqr",
                    "separability", "tau"])
        for model in sorted(per_image.keys()):
            entries = per_image[model]
            spreads = [e["spread"]   for e in entries]
            iqrs    = [e["mean_iqr"] for e in entries]
            seps    = [e["separability"] for e in entries if np.isfinite(e["separability"])]
            taus    = [e["tau"]          for e in entries if not np.isnan(e["tau"])]
            sep_cell = round(float(np.mean(seps)), 4) if seps else "inf"
            tau_cell = round(float(np.mean(taus)), 4) if taus else ""
            w.writerow([
                model,
                len(entries),
                round(float(np.mean(spreads)), 4),
                round(float(np.mean(iqrs)),    4),
                sep_cell,
                tau_cell,
            ])


SIMPLIFICATION_LEVELS = [0, 20, 40, 60, 80]  # 0 = original; 80 = most simplified


def load_simplified_levels(workdir, name):
    """Load all simplified-image-sequence frames for one image.

    Returns dict {level: rgb_np_uint8}. Raises if any expected level is missing.
    """
    seq_dir = os.path.join(workdir, name, "simplified_image_sequence")
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Missing simplified-sequence folder: {seq_dir}")
    out = {}
    for L in SIMPLIFICATION_LEVELS:
        path = os.path.join(seq_dir, f"{L}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing simplification level {L}: {path}")
        out[L] = np.array(Image.open(path).convert("RGB"))
    return out


def collect_per_level_masks(name, level_images, sam_cache_dir, device, sam_conf):
    """Returns dict {level: list_of_masks}. Caches per (name, level).

    For L0, migrates the legacy <name>.npz cache (from target_imgs/) if found,
    since the user confirmed L0 is identical to the original image.
    """
    out = {}
    pending = []  # (level, image, cache_path)
    for L, img in level_images.items():
        cache_path = os.path.join(sam_cache_dir, f"{name}__L{L}.npz")
        if L == 0 and not os.path.exists(cache_path):
            legacy = os.path.join(sam_cache_dir, f"{name}.npz")
            if os.path.exists(legacy):
                os.rename(legacy, cache_path)
                print(f"Migrated legacy SAM cache: {name}.npz -> {name}__L0.npz")
        if os.path.exists(cache_path):
            out[L] = load_cached_masks(cache_path)
        else:
            pending.append((L, img, cache_path))

    if pending:
        print(f"SAM cache miss for levels {[L for L, _, _ in pending]} — computing ...")
        mg = init_sam(device, sam_conf)
        for L, img, cache_path in pending:
            raw_masks = run_sam(mg, img)
            H, W = img.shape[:2]
            masks = filter_by_area(raw_masks, H * W, MIN_AREA_FRAC)
            save_cached_masks(cache_path, masks)
            out[L] = masks
            print(f"  L{L:>2d}: {len(masks)} masks above {MIN_AREA_FRAC*100:.1f}% of {H}x{W}")
        del mg
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out


def run_simplified_depth_phase(model_key, name, level_images, level_masks,
                               output_dir, device):
    """For one (image, model): infer depth on L0 only, render per-level rows."""
    meta = MODEL_REGISTRY[model_key]
    img_l0 = level_images[0]
    H, W = img_l0.shape[:2]

    print(f"Loading depth model: {meta['hf_id']} ...")
    processor, depth_model = load_depth_model(meta["hf_id"], device)
    depth = infer_depth(
        Image.fromarray(img_l0), processor, depth_model, device,
        target_hw=(H, W),
        larger_is_closer=meta["larger_is_closer"],
    )
    del processor, depth_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n=== {model_key} ({meta['hf_id']}) ===")
    print(f"L0 depth [{depth.min():.2f}, {depth.max():.2f}] (shared across all levels)")

    depth_range = float(depth.max() - depth.min())

    per_level = {}
    for L in SIMPLIFICATION_LEVELS:
        masks_all = level_masks[L]
        if not masks_all:
            print(f"L{L:>2d}: no masks above area threshold — skipping")
            continue
        stats_all = per_mask_stats(depth, masks_all)
        order = sort_back_to_front(stats_all)
        medians = [stats_all[i]["median"] for i in order]
        iqrs    = [stats_all[i]["iqr"]    for i in order]
        spread  = (medians[0] - medians[-1]) / depth_range if depth_range > 0 else 0.0
        mean_iqr = float(np.mean(iqrs)) / depth_range if depth_range > 0 else float("nan")
        separability = (spread / (spread + mean_iqr)) if mean_iqr > 0 else float("inf")
        per_level[L] = {
            "masks_all":    masks_all,
            "stats_all":    stats_all,
            "masks_vis":    masks_all[:VIS_MAX_MASKS],
            "stats_vis":    stats_all[:VIS_MAX_MASKS],
            "n_total":      len(masks_all),
            "spread":       spread,
            "mean_iqr":     mean_iqr,
            "separability": separability,
        }

    l0_data = per_level.get(0)
    rows, table_rows = [], []
    for L in SIMPLIFICATION_LEVELS:
        if L not in per_level:
            continue
        d = per_level[L]
        if L == 0 or l0_data is None:
            tau = float("nan")
        else:
            tau = cross_level_tau(
                d["masks_all"], d["stats_all"],
                l0_data["masks_all"], l0_data["stats_all"],
            )
        tau_str = "—" if np.isnan(tau) else f"{tau:+.2f}"
        print(f"L{L:>2d}: {d['n_total']} masks scored ({len(d['masks_vis'])} shown) | "
              f"spread {d['spread']:.2%} | meanIQR {d['mean_iqr']:.3f} | "
              f"sep {d['separability']:.2f} | tau vs L0: {tau_str}")

        rows.append({
            "level":     L,
            "image_rgb": level_images[L],
            "depth":     depth,
            "masks":     d["masks_vis"],
            "stats":     d["stats_vis"],
        })
        table_rows.append({
            "level":        L,
            "n_masks":      d["n_total"],
            "spread":       d["spread"],
            "mean_iqr":     d["mean_iqr"],
            "separability": d["separability"],
            "tau":          tau,
        })

    png_path = os.path.join(output_dir, model_key, f"{name}__multilevel.png")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    render_multilevel_diagnostic(rows, png_path, model_key, name)
    print(f"saved: {png_path}")
    return table_rows


def main():
    parser = argparse.ArgumentParser(
        description="Depth estimator evaluation across SDS-simplification levels "
                    "for one image. Reuses cached SAM masks; depth is inferred on "
                    "every level."
    )
    parser.add_argument("--name", required=True, nargs="+",
                        help="One or more image names (folders under workdir/, "
                             "e.g. --name gramps berengo_asylum).")
    parser.add_argument("--workdir", default="./workdir",
                        help="Folder containing <name>/simplified_image_sequence/.")
    parser.add_argument("--output", default="./depth_eval_output",
                        help="Folder for diagnostic outputs and caches.")
    parser.add_argument("--config", default="./config/base_config.yaml",
                        help="YAML containing SAM config under key 'sam'.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which depth models to run.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_conf = load_sam_config(args.config)

    sam_cache_dir = os.path.join(args.output, "sam_masks_full")
    os.makedirs(sam_cache_dir, exist_ok=True)
    summaries_dir = os.path.join(args.output, "summaries")

    for name in args.name:
        print(f"\n=== Image: {name} ===")
        try:
            level_images = load_simplified_levels(args.workdir, name)
        except FileNotFoundError as e:
            print(f"Skipping '{name}': {e}")
            continue
        print(f"Loaded {len(level_images)} levels for '{name}'.")

        level_masks = collect_per_level_masks(
            name, level_images, sam_cache_dir, device, sam_conf,
        )
        if not any(level_masks.values()):
            print(f"No masks above area threshold for '{name}' — skipping.")
            continue
        print("Per-level mask counts: " +
              ", ".join(f"L{L}={len(level_masks[L])}" for L in SIMPLIFICATION_LEVELS))

        model_tables = {}
        for model_key in args.models:
            table_rows = run_simplified_depth_phase(
                model_key, name, level_images, level_masks, args.output, device,
            )
            if table_rows:
                model_tables[model_key] = table_rows

        if model_tables:
            os.makedirs(summaries_dir, exist_ok=True)
            summary_path = os.path.join(summaries_dir, f"{name}__summary.csv")
            write_cross_model_summary_csv(model_tables, summary_path)
            print(f"saved cross-model summary: {summary_path}")

    cross_path = os.path.join(args.output, "cross_image_summary.csv")
    aggregate_summaries_across_images(summaries_dir, cross_path)
    if os.path.exists(cross_path):
        print(f"saved cross-image aggregate: {cross_path}")


if __name__ == "__main__":
    main()
