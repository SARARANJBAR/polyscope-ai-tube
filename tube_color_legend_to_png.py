#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tube_utils import load_app_config

cfg = load_app_config()
root_code_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(root_code_dir, cfg["model_dir"], cfg["tube_ckpt"])
model_config_path = os.path.join(model_dir, "config.json")
with open(model_config_path, "r") as f:
    model_config = yaml.safe_load(f)

colors = model_config["id2color"].values()
label2id = model_config["label2id"].keys()

img_width, img_height = 500, 250
legend = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

n = len(colors)
margin_top = 20
margin_bottom = 5
available_height = img_height - margin_top - margin_bottom

# Dynamic block + spacing calculation
block_h = int(available_height / n * 0.6)
spacing = int(available_height / n * 0.4)
block_w = 120

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
font_thickness = 1

y = margin_top
for name, hexcolor in zip(label2id, colors):
    bgr = tuple(int(hexcolor[i:i+2], 16) for i in (5, 3, 1))
    # Draw color block
    cv2.rectangle(legend, (50, y), (50 + block_w, y + block_h), bgr, -1)

    # Draw label
    text_x = 50 + block_w + 30
    text_y = y + int(block_h * 0.7)
    cv2.putText(
        legend,
        name,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA,
    )

    y += block_h + spacing

legend = cv2.resize(legend, (img_width, img_height), interpolation=cv2.INTER_AREA)

savepath = os.path.join(root_code_dir, "docs", "tmeseg_color_legend.png")

cv2.imwrite(savepath, legend)
print(savepath)

