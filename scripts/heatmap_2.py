import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor
)

# ----------------------------
# 配置
# ----------------------------
MODEL_NAME = "clip-vit-large-patch14-336"
IMAGE_PATH = "dog.jpg"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 加载模型
# ----------------------------
image_processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
vision_model = CLIPVisionModel.from_pretrained(MODEL_NAME).to(device).eval()

# ----------------------------
# 图像预处理
# ----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
pixel_values = image_processor(
    images=image, return_tensors="pt"
).pixel_values.to(device)

# ----------------------------
# 前向（hidden states + attention）
# ----------------------------
with torch.no_grad():
    outputs = vision_model(
        pixel_values,
        output_hidden_states=True,
        output_attentions=True
    )

    # ========= Patch features (for R) =========
    # 与 CRTP 一致：使用 select_layer = -2
    hidden = outputs.hidden_states[-2]     # [1, 577, D]
    patch_feats = hidden[:, 1:, :]          # [1, 576, D]

    # ========= 线索 A：CLS Attention =========
    attn = outputs.attentions[-2]           # [-2 layer]
    cls_to_patch = attn[:, :, 0, 1:]         # [1, H, 576]
    A = cls_to_patch.max(dim=1).values      # [1, 576]

    # ========= 线索 R：冗余度 =========
    B, N, D = patch_feats.shape
    h = w = int(N ** 0.5)  # 24
    R = torch.zeros(B, N, device=patch_feats.device)

    feats = patch_feats[0]                  # [576, D]
    feats = feats / feats.norm(dim=1, keepdim=True)

    for idx in range(N):
        i, j = idx // w, idx % w
        neighbors = []

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                n_idx = ni * w + nj
                sim = torch.dot(feats[idx], feats[n_idx])
                neighbors.append(sim)

        if neighbors:
            mean_sim = torch.stack(neighbors).mean()
            R[0, idx] = 1.0 - mean_sim
        else:
            R[0, idx] = 0.0

# ----------------------------
# reshape 为 24x24
# ----------------------------
A_map = A.squeeze(0).cpu().numpy().reshape(h, w)
R_map = R.squeeze(0).cpu().numpy().reshape(h, w)

def normalize_to_minus1_1(x):
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        return 2 * (x - x_min) / (x_max - x_min) - 1
    else:
        return np.zeros_like(x)

A_map = normalize_to_minus1_1(A_map)
R_map = normalize_to_minus1_1(R_map)

# ----------------------------
# 可视化
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 原图
axes[0].imshow(image)
axes[0].set_title("Original Image", fontsize=14)
axes[0].axis("off")

# 线索 A
axes[1].imshow(image, alpha=0.8)
im1 = axes[1].imshow(
    A_map,
    cmap="hot",
    alpha=0.4,
    extent=[0, image.width, image.height, 0]
)
axes[1].set_title("Cue A: [CLS] → Patch Attention", fontsize=14)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# 线索 R
axes[2].imshow(image, alpha=0.8)
im2 = axes[2].imshow(
    R_map,
    cmap="viridis",
    alpha=0.4,
    extent=[0, image.width, image.height, 0]
)
axes[2].set_title("Cue R: Local Redundancy (1 - CosSim)", fontsize=14)
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("Cue_A_and_R_Heatmaps.png", dpi=150)
plt.show()

print("Saved as Cue_A_and_R_Heatmaps.png")
