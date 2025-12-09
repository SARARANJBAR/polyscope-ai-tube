#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import sys
import cv2
import time
import torch
import json
import argparse
import numpy as np
import torchvision.transforms as T

import tifffile
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageOps
import openslide as openslide
from collections import OrderedDict
from tube_utils import build_model_and_path, load_app_config, ensure_dir, check_input_valid

def get_custom_ss1(cfg):
    """ Custom because it specific to how this task is done in AI-sTIL"""
    out_mpp=0.22
    out_mpp_target_objective=40
    cws_objective_value=20
    ext = os.path.splitext(cfg.input_file_path)[1].lower()
    
    if ext in ['.svs', '.ndpi']:
        try:
            openslide_obj = openslide.OpenSlide(cfg.input_file_path)
            objective_power = float(openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            slide_dimension = openslide_obj.level_dimensions[0] # W, H
            in_mpp = float(openslide_obj.properties[openslide.PROPERTY_NAME_MPP_X])
            
            cws_objective_value = (cws_objective_value * \
                                   (objective_power / out_mpp_target_objective) * \
                                   (in_mpp / out_mpp))
            rescale = objective_power / cws_objective_value
            slide_dimension_rescale = np.round(np.array(slide_dimension) / rescale)
            slide_dimension_ss1 = (slide_dimension_rescale/16).astype(np.int32)
            
            ss1 = openslide_obj.get_thumbnail(slide_dimension_ss1)
            return ss1
        except openslide.OpenSlideUnsupportedFormatError as e:
            print(f"Error: Something wrong with the image. Details: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
            
    elif ext in ['.tif', '.tiff', '.qptiff']:
        try:
            tif_image = tifffile.TiffFile(cfg.input_file_path)
            objective_power = float(cfg.input_objective_power)
            h_l0, w_l0 = tif_image.series[0].shape[:2] # this is swapped the format of svs
            slide_dimension = [w_l0, h_l0]

            in_mpp = tif_image.pages[0].tags['XResolution'].value
            in_mpp = 10000 * in_mpp[1] / in_mpp[0]
            cws_objective_value = (cws_objective_value * \
                                   (objective_power / out_mpp_target_objective) * \
                                   (in_mpp / out_mpp))       
            rescale = objective_power / cws_objective_value
            slide_dimension_rescale = np.round(np.array(slide_dimension) / rescale)
            slide_dimension_ss1 = (slide_dimension_rescale/16).astype(np.int32)
            w_ss1, h_ss1 = slide_dimension_ss1  # (W, H)
            arr0 = tif_image.series[0].asarray()   # shape (H, W, C)
            arr0_re = cv2.resize(arr0, (w_ss1, h_ss1), interpolation=cv2.INTER_AREA)
            ss1 = Image.fromarray(arr0_re)
            return ss1

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    else:
        print(f"Getting SS1 from input file extension {ext} not implemented.")
        return None

        
def parse_patch_gen_size(patch_gen_size):
    if 'x' in patch_gen_size:
        base_size, factor = patch_gen_size.split('x')
        base_size = int(base_size); factor = int(factor)
        extract_size = base_size * factor
        downsample = factor
    else:
        base_size = int(patch_gen_size)
        extract_size = base_size
        downsample = 1
    return extract_size, downsample


def paste_clip(canvas, patch, top, left):
    H, W = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    end_y = min(top + ph, H)
    end_x = min(left + pw, W)
    h = end_y - top
    w = end_x - left
    canvas[top:end_y, left:end_x] += patch[:h, :w]


def pad_to_multiple(pil_img, multiple=512, fill=(255,255,255)):
    w,h = pil_img.size
    pad_w = (multiple - (w % multiple)) % multiple
    pad_h = (multiple - (h % h % multiple)) if False else (multiple - (h % multiple)) % multiple
    if pad_w == 0 and pad_h == 0:
        return pil_img, (w,h)
    padded = ImageOps.expand(pil_img, border=(0,0,pad_w,pad_h), fill=fill)
    return padded, (w,h)


def make_tissue_mask_simple(he_np, cutoff=240):
    return (np.mean(he_np, axis=-1) < cutoff).astype(np.uint8)


def eval_slide(img_path, model, device, id2color,
               patch_gen_size="512", input_size=512, stride_factor=2,
               tumor_bed_threshold=0.5, batch_size=32, tissue_threshold=0.0):
    
    slide_id = os.path.splitext(os.path.basename(img_path))[0]

    extract_size, downsample_factor = parse_patch_gen_size(patch_gen_size)
    stride = extract_size // stride_factor
    print(f"Processing {os.path.basename(img_path)}: \
         extract {extract_size}x{extract_size}, \
         downsample {downsample_factor}x")

    he = Image.open(img_path).convert("RGB")
    he_pad, (W0, H0) = pad_to_multiple(he, extract_size, (255,255,255))
    W, H = he_pad.size
    he_np = np.array(he_pad)
    tissue_mask = make_tissue_mask_simple(he_np)

    bed_vol = np.zeros((H, W), dtype=np.float32)
    bed_cnt = np.zeros((H, W), dtype=np.float32)

    tops  = list(range(0, H - extract_size + 1, stride))
    lefts = list(range(0, W - extract_size + 1, stride))

    transform_op = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    batch_imgs, batch_coords = [], []

    def flush_batch():
        nonlocal batch_imgs, batch_coords, bed_vol, bed_cnt
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            logits = model(x)                       # [B,1,h,w]
            prob = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
        for i, (top,left) in enumerate(batch_coords):
            if downsample_factor > 1:
                p_up = cv2.resize(prob[i], (extract_size, extract_size), interpolation=cv2.INTER_LINEAR)
                paste_clip(bed_vol, p_up, top, left)
                count_mask = np.ones((extract_size, extract_size), dtype=np.float32)
            else:
                paste_clip(bed_vol, prob[i], top, left)
                count_mask = np.ones_like(prob[i], dtype=np.float32)
            paste_clip(bed_cnt, count_mask, top, left)
        batch_imgs.clear()
        batch_coords.clear()

    total_patches = 0
    for top in tops:
        for left in lefts:
            bottom = top + extract_size
            right  = left + extract_size
            cov = tissue_mask[top:bottom, left:right].mean()
            # if cov < tissue_threshold:   # keep same behavior as your example (currently not enforced)
            #     continue
            crop = he_np[top:bottom, left:right, :]
            if downsample_factor > 1:
                crop = np.array(Image.fromarray(crop).resize((input_size, input_size), Image.BILINEAR))
            x = transform_op(Image.fromarray(crop))
            batch_imgs.append(x); batch_coords.append((top,left))
            total_patches += 1
            if len(batch_imgs) >= batch_size:
                flush_batch()
    flush_batch()

    np.divide(bed_vol, bed_cnt, out=bed_vol, where=(bed_cnt!=0))
    bed_pred_bin = (bed_vol > tumor_bed_threshold).astype(np.uint8)
    bed_pred_bin = bed_pred_bin[:H0, :W0]
    
    # create color image from binary mask
    tumor_color_hex = id2color["1"]
    tumor_color_rgb = tuple(int(tumor_color_hex[i:i+2], 16) for i in (1, 3, 5))
    bed_pred_bin = bed_pred_bin.astype(np.uint8)
    h, w = bed_pred_bin.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[bed_pred_bin == 1] = tumor_color_rgb
    out = Image.fromarray(out, mode="RGB")
    return out

def run_tube_model():

    p = argparse.ArgumentParser("TUBE Inference (Mask2Former/SegFormer)")
    p.add_argument("--input_file_path", help="Path to input image file", required=True, type=str)        
    p.add_argument("--output_dir", help="Path to output directory", required=True, type=str)
    p.add_argument("--input_objective_power", help="input image objective power. required for tifs", type=int)
    cfg = p.parse_args()

    # input inspection
    check_input_valid(cfg.input_file_path)
    ext = os.path.splitext(cfg.input_file_path)[1].lower()
    fname = os.path.basename(cfg.input_file_path)
    if not (ext.endswith("svs") or ext.endswith("ndpi")):
        if cfg.input_objective_power is None:
            p.error("Providing input_objective_power is required for non-svs/ndpi files")

    # output preparation
    ensure_dir(cfg.output_dir)
    out_file_tag = f"{fname}_Ss1_tbedPred.png"
    out_dir = os.path.join(cfg.output_dir, fname)
    os.makedirs(out_dir, exist_ok=True)
    output_fpath = os.path.join(out_dir, out_file_tag)
    if os.path.exists(output_fpath):
        print(f'{output_fpath} exists.')
        sys.exit()

    # add app related info to config variable
    for k, v in load_app_config().items():
        setattr(cfg, k, v)

    # prepare json report
    report = OrderedDict()
    report["metadata"] = {}
    report["metadata"]["file_name"] = fname
    report["metadata"]["input_file_path"] = cfg.input_file_path
    if vars(cfg)["input_objective_power"] is not None:
        report["metadata"]["input_objective_power"] = vars(cfg)["input_objective_power"]
    
    report["config"] = {}
    for k, v in vars(cfg).items():
        if k in ["input_file_path", "output_dir", "input_objective_power", "model_dir"]:
            continue
        report["config"][k] = v
    report["config"]["model_config"] = {}


    print(f"\nRunning tube model on {fname}")

    report["runtimes"] = {}
    # ============================================================
    # --- Step 1: get ss1, the AIsTIL way ---
    step_time = time.time()
    ss1_save_path = os.path.join(out_dir, fname+'_Ss1.jpg')
    ss1 = get_custom_ss1(cfg)
    ss1.save(ss1_save_path, format='JPEG')
    print(f"Saved {ss1_save_path}")
    report["runtimes"]["ss1_generation_sec"] = time.time()-step_time
    
    # ============================================================
    # --- Step 2: predict tube mask 
    step_time = time.time()
    model, device, model_config = build_model_and_path(cfg)
    bed_out = eval_slide(ss1_save_path,
                         model, device, model_config["id2color"],
                         patch_gen_size=model_config["patch_gen_size"], 
                         input_size=model_config["input_size"],
                         stride_factor=model_config["stride_factor"],
                         tumor_bed_threshold=model_config["tumor_bed_threshold"],
                         batch_size=model_config["batch_size"],
                         tissue_threshold=model_config["tissue_threshold"]
    )
    bed_out.save(output_fpath)
    print(f"saved {output_fpath}.")

    # add model config to report
    for k, v in model_config.items():
        if k in ["name"]:
            continue
        report["config"]["model_config"][k] = v
    
    report["runtimes"]["tube_inference_sec"] = time.time()-step_time
    # ============================================================
    
    # done! save json report 
    total_time = sum(report["runtimes"].values())
    report["runtimes"]["total_time_sec"] = total_time
    outputjsonpath = os.path.join(out_dir, fname+"_stats.json")
    if not os.path.exists(outputjsonpath):
        with open(outputjsonpath, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Saved {outputjsonpath}")
    print(f"\n---- {total_time:.0f} sec ({total_time / 60.0:.1f} min) for {fname}\n")

if __name__ == "__main__":
    run_tube_model()

