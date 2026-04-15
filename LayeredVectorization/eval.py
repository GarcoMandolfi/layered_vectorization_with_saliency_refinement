import argparse
import csv
import json
import os
import random
import sys
import yaml
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pydiffvg
from utils.img_process import svg_to_img, rgba_to_rgb, sam_image
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim, peak_signal_noise_ratio as psnr
import lpips
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel

def parse_svg(svg_path):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    return canvas_width, canvas_height, shapes, shape_groups


def render_to_numpy(canvas_width, canvas_height, shapes, shape_groups, device):
    img = svg_to_img(canvas_width, canvas_height, shapes, shape_groups, device)
    img = rgba_to_rgb(img, device)
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def load_target_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_caption_cache(output_dir):
    cache_path = os.path.join(output_dir, "captions.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}


def save_caption_cache(output_dir, cache):
    cache_path = os.path.join(output_dir, "captions.json")
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


def init_blip(device):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    model.eval()
    return processor, model


def caption_image(pil_img, processor, model, device):
    inputs = processor(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)


def init_clip(device):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return processor, model


def clip_similarity(np_imgs, text, processor, model, device):
    pil_imgs = [Image.fromarray(img) for img in np_imgs]
    inputs = processor(text=[text], images=pil_imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    sims = (image_embeds @ text_embeds.T).squeeze(-1)
    return sims.cpu().tolist()


def primitive_masks(shapes, shape_groups, height, width, device, alpha_threshold=0.01):
    masks = []
    for g in shape_groups:
        img = svg_to_img(width, height, shapes, [g], device)
        alpha = img[:, :, 3].detach().cpu().numpy()
        mask = (alpha > alpha_threshold).astype(np.uint8) * 255
        masks.append(mask)
    return masks


def compute_vec(prim_masks, sam_masks, containment_thresh=0.9):
    scores = []
    for M in sam_masks:
        n_interact = 0
        n_contained = 0
        for P in prim_masks:
            intersection = np.count_nonzero(P & M)
            if intersection == 0:
                continue
            n_interact += 1
            p_area = np.count_nonzero(P)
            if p_area > 0 and intersection / p_area >= containment_thresh:
                n_contained += 1
        if n_interact > 0:
            scores.append(n_contained / n_interact)
    return np.mean(scores) if scores else 0.0


def load_sam_config(config_path="./config/base_config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["sam"]


def get_largest_masks(target_img_np, name, output_dir, device, sam_conf, top_k=4):
    cache_dir = os.path.join(output_dir, "sam_masks")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return [data[k] for k in data.files]

    masks = sam_image(device, target_img_np, "-1", sam_conf)
    masks = masks[1:]  # drop the dummy whole-image mask at index 0
    masks.sort(key=lambda m: cv2.countNonZero(m), reverse=True)
    top_masks = masks[:top_k]

    np.savez_compressed(cache_path, *top_masks)
    return top_masks


def remove_paths_random(shapes, shape_groups, fraction=0.3, seed=0):
    rng = random.Random(seed)
    n = len(shape_groups)
    n_remove = int(n * fraction)
    remove_idx = set(rng.sample(range(n), n_remove))

    new_groups = []
    for i, g in enumerate(shape_groups):
        new_color = g.fill_color.clone()
        if i in remove_idx:
            new_color[3] = 0.0
        new_groups.append(pydiffvg.ShapeGroup(
            shape_ids=g.shape_ids,
            fill_color=new_color,
            stroke_color=g.stroke_color
        ))
    return shapes, new_groups


def evaluation(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_print_timing(False)

    sam_conf = load_sam_config()

    target_imgs = [f for f in os.listdir(args.target_imgs) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    caption_cache = load_caption_cache(args.output)
    blip_processor, blip_model = None, None
    clip_processor, clip_model = None, None

    N = 5  # number of random path-removal runs to average over

    baseline_mse, saliency_mse = [], []
    baseline_psnr, saliency_psnr = [], []
    baseline_ssim, saliency_ssim = [], []
    saliency_lpips, baseline_lpips = [], []
    baseline_path_sem, saliency_path_sem = [], []
    baseline_vec, saliency_vec = [], []

    for target_filename in tqdm(sorted(target_imgs), desc="Evaluating"):
        name = os.path.splitext(target_filename)[0]
        if name.startswith("resized_"):
            name = name[len("resized_"):]
        target_path = os.path.join(args.target_imgs, target_filename)

        baseline_svg = os.path.join(args.workdir, name, "final.svg")
        saliency_svg = os.path.join(args.workdir, f"{name}_with_saliency", "final.svg")

        if not os.path.exists(baseline_svg) or not os.path.exists(saliency_svg):
            continue

        target_img = load_target_image(target_path)
        sam_masks = get_largest_masks(target_img, name, args.output, device, sam_conf)
        b_w, b_h, b_shapes, b_groups = parse_svg(baseline_svg)
        s_w, s_h, s_shapes, s_groups = parse_svg(saliency_svg)
        baseline_img = render_to_numpy(b_w, b_h, b_shapes, b_groups, device)
        saliency_img = render_to_numpy(s_w, s_h, s_shapes, s_groups, device)

        if target_filename not in caption_cache:
            if blip_model is None:
                print("Loading BLIP model...")
                blip_processor, blip_model = init_blip(device)
            pil_img = Image.open(target_path).convert("RGB")
            caption_cache[target_filename] = caption_image(pil_img, blip_processor, blip_model, device)
            save_caption_cache(args.output, caption_cache)
        caption = caption_cache[target_filename]

        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
        baseline_tensor = torch.from_numpy(baseline_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
        saliency_tensor = torch.from_numpy(saliency_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

        baseline_mse.append(mse(target_img, baseline_img))
        baseline_psnr.append(psnr(target_img, baseline_img, data_range=255))
        baseline_ssim.append(ssim(target_img, baseline_img, channel_axis=2, data_range=255))
        baseline_lpips.append(lpips_fn(target_tensor, baseline_tensor).item())

        saliency_mse.append(mse(target_img, saliency_img))
        saliency_psnr.append(psnr(target_img, saliency_img, data_range=255))
        saliency_ssim.append(ssim(target_img, saliency_img, channel_axis=2, data_range=255))
        saliency_lpips.append(lpips_fn(target_tensor, saliency_tensor).item())

        if clip_model is None:
            print("Loading CLIP model...")
            clip_processor, clip_model = init_clip(device)

        baseline_reduced_imgs, saliency_reduced_imgs = [], []
        for run in range(N):
            _, b_red = remove_paths_random(b_shapes, b_groups, fraction=0.3, seed=run)
            baseline_reduced_imgs.append(render_to_numpy(b_w, b_h, b_shapes, b_red, device))
            _, s_red = remove_paths_random(s_shapes, s_groups, fraction=0.3, seed=run)
            saliency_reduced_imgs.append(render_to_numpy(s_w, s_h, s_shapes, s_red, device))

        all_imgs = [baseline_img, saliency_img] + baseline_reduced_imgs + saliency_reduced_imgs
        sims = clip_similarity(all_imgs, caption, clip_processor, clip_model, device)
        b_full, s_full = sims[0], sims[1]
        b_reduced = sims[2:2 + N]
        s_reduced = sims[2 + N:]
        baseline_path_sem.append(b_full - np.mean(b_reduced))
        saliency_path_sem.append(s_full - np.mean(s_reduced))

        target_h, target_w = target_img.shape[:2]
        b_prim = primitive_masks(b_shapes, b_groups, target_h, target_w, device)
        s_prim = primitive_masks(s_shapes, s_groups, target_h, target_w, device)
        baseline_vec.append(compute_vec(b_prim, sam_masks))
        saliency_vec.append(compute_vec(s_prim, sam_masks))


    csv_path = os.path.join(args.output, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "baseline", "saliency"])
        writer.writerow(["MSE",     round(np.mean(baseline_mse), 4),      round(np.mean(saliency_mse), 4)])
        writer.writerow(["PSNR",    round(np.mean(baseline_psnr), 2),     round(np.mean(saliency_psnr), 2)])
        writer.writerow(["SSIM",    round(np.mean(baseline_ssim), 4),     round(np.mean(saliency_ssim), 4)])
        writer.writerow(["LPIPS",   round(np.mean(baseline_lpips), 4),    round(np.mean(saliency_lpips), 4)])
        writer.writerow(["PathSem", round(np.mean(baseline_path_sem), 4), round(np.mean(saliency_path_sem), 4)])
        writer.writerow(["VeC",     round(np.mean(baseline_vec), 4),      round(np.mean(saliency_vec), 4)])
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vectorization quality: baseline vs saliency-guided")
    parser.add_argument("--workdir", type=str, default="./workdir",
                        help="Path to the workdir containing result folders")
    parser.add_argument("--target-imgs", type=str, default="./target_imgs",
                        help="Path to the folder with original input images")
    parser.add_argument("--output", type=str, default="./eval_output",
                        help="Path to the output directory for evaluation results")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    evaluation(args)
