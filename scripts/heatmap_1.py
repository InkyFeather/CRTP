import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    AutoTokenizer,
    CLIPImageProcessor
)

# ----------------------------
# 配置
# ----------------------------
MODEL_NAME = "clip-vit-large-patch14-336"
IMAGE_PATH = "cat.jpg" 
TEXT_PROMPTS = ["kitten", "shoe"] # 不同文本提示
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 加载模型
# ----------------------------
image_processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
vision_model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(device).eval()
text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ----------------------------
# 图像与文本预处理
# ----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
inputs = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

# 预处理多个文本
text_inputs = tokenizer(TEXT_PROMPTS, return_tensors="pt", padding=True, truncation=True).to(device)

# ----------------------------
# 提取特征
# ----------------------------
with torch.no_grad():
    vision_outputs = vision_model.vision_model(inputs, output_hidden_states=True)
    all_image_features = vision_outputs.hidden_states[-2]  # [1, 577, 1024]
    patch_features = all_image_features[:, 1:, :]  # [1, 576, 1024]

    normalized_patch = vision_model.vision_model.post_layernorm(patch_features)
    proj_patch = vision_model.visual_projection(normalized_patch)  # [1, 576, 768]

    # 提取两个文本的嵌入
    text_embeds = text_model(**text_inputs).text_embeds  # [2, 768]
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # [2, 768]

# ----------------------------
# 计算两个文本的 CRTP Similarity
# ----------------------------
Similarity_maps = []
for i, text in enumerate(TEXT_PROMPTS):
    Similarity_raw = torch.matmul(
        proj_patch, text_embeds[i].unsqueeze(-1)
    ).squeeze(-1)  # [1, 576]

    Similarity_raw = -Similarity_raw
    Similarity_raw = Similarity_raw.squeeze(0).cpu().numpy()  # [576]

    # ---- 归一化到 [-1, 1] ----
    min_val = Similarity_raw.min()
    max_val = Similarity_raw.max()
    Similarity_raw = 2 * (Similarity_raw - min_val) / (max_val - min_val) - 1

    Similarity_maps.append(Similarity_raw)

# ----------------------------
# 转为 24x24 热力图
# ----------------------------
grid_size = 24
heatmap_kitten = Similarity_maps[0].reshape(grid_size, grid_size)
heatmap_remote = Similarity_maps[1].reshape(grid_size, grid_size)

# ----------------------------
# 可视化
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 原图
axes[0].imshow(image)
axes[0].set_title("Original Image", fontsize=14)
axes[0].axis("off")

# CRTP: Similarity for "kitten"
axes[1].imshow(image, alpha=0.8)
im1 = axes[1].imshow(heatmap_kitten, cmap='jet', alpha=0.4, extent=[0, image.width, image.height, 0])
axes[1].set_title("Similarity for 'kitten'", fontsize=14)
axes[1].axis("off")
cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_label('Similarity', rotation=270, labelpad=15)

# CRTP: Similarity for "shoe"
axes[2].imshow(image, alpha=0.8)
im2 = axes[2].imshow(heatmap_remote, cmap='jet', alpha=0.4, extent=[0, image.width, image.height, 0])
axes[2].set_title("Similarity for 'shoe'", fontsize=14)
axes[2].axis("off")
cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.set_label('Similarity', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("Cue_S_similarity_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'Cue_S_similarity_comparison.png'")