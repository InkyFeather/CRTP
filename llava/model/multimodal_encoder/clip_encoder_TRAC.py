import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np

# HuggingFace Transformers 中的 CLIP 相关组件
from transformers import (
    CLIPModel, 
    CLIPVisionModel, 
    CLIPVisionModelWithProjection, 
    CLIPImageProcessor, 
    CLIPVisionConfig, 
    CLIPTextModelWithProjection, 
    AutoTokenizer
)

import time


class CLIPVisionTower(nn.Module):
    """
    封装 CLIP 视觉编码器（Vision Tower）的模块，支持延迟加载、特征选择、以及基于文本相似度的 token 剪枝（TRIM）。
    主要用于多模态大模型（如 LLaVA）中提取图像特征。
    """
    
    def __init__(self, vision_tower, args, delay_load=False):
        """
        初始化视觉塔。
        
        Args:
            vision_tower (str): 预训练 CLIP 视觉模型名称（如 "clip-vit-large-patch14-336"）
            args: 包含配置参数的对象（如 mm_vision_select_layer 等）
            delay_load (bool): 是否延迟加载模型（用于节省内存或按需加载）
        """
        super().__init__()

        self.is_loaded = False  # 标记模型是否已加载

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer  # 从哪一层提取特征（如 -2 表示倒数第二层）
        # 选择特征类型：'patch'（仅图像 patch tokens，不含 cls token）或 'cls_patch'（包含 cls token）
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # token 剪枝函数，例如 'TRIM:0.5' 表示保留 50% 最相关的 tokens
        self.token_reduce_func = getattr(args, 'mm_vision_token_reduce_func', None)

        if not delay_load:
            # 立即加载模型
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            # 如果需要微调视觉塔，则也立即加载
            self.load_model()
        else:
            # 仅加载配置（不加载权重），用于延迟加载场景
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        """
        加载 CLIP 视觉模型及相关组件（如需要 TRIM，则额外加载文本塔）。
        """
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        # 加载图像预处理器（用于图像归一化、resize 等）
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        # 加载基础 CLIP 视觉模型（不含投影头）
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)  # 默认冻结视觉塔参数

        # 如果启用了 TRIM 机制（基于文本相似度剪枝 tokens）
        if self.token_reduce_func and 'TRIM' in self.token_reduce_func:
            if device_map:  # 推理模式（如评估）
                # 加载完整 CLIP 模型以获取 visual_projection（投影层）
                self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name)
                # 将投影层绑定到 vision_tower 上（注意：vision_tower 本身不含 projection）
                self.vision_tower.visual_projection = self.clip_model.visual_projection.to(self.device)
                self.clip_model.requires_grad_(False)
            else:
                # 训练模式：直接加载带投影的视觉模型
                self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)

            # 额外加载 CLIP 文本编码器（用于 TRIM 中计算图像-文本相似度）
            self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)
            self.text_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """
        从模型输出中选择指定层的特征，并按需裁剪 cls token。
        
        Args:
            image_forward_outs: CLIPVisionModel 的输出，包含 hidden_states
        
        Returns:
            image_features: 选定的图像特征（如仅 patch tokens）
            all_image_features: 完整的选定层特征（含 cls token）
        """
        all_image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            # 跳过 cls token（索引 0），只保留 patch tokens
            image_features = all_image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            # 保留 cls + patch tokens
            image_features = all_image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features, all_image_features

    def token_reduction(self, image_features, all_image_features, text_features=None, llm_attention_scores=None):
        """
        实现 TRIM 机制：基于图像 patch 与文本的相似度，保留 top-k 最相关的 tokens，
        并将剩余 tokens 聚合为一个“平均”token（用于保持序列长度一致）。
        
        Args:
            image_features: [B, N, D] 图像 patch 特征（不含 cls）
            all_image_features: [B, N+1, D] 包含 cls 的完整特征（TRIM 中未使用）
            text_features: [B, D] 文本嵌入（由 CLIP 文本编码器生成）
        
        Returns:
            image_features: 剪枝并聚合后的特征 [B, k*N + 1, D]
            actual_dims: 每个样本实际保留的 token 数（用于后续处理）
        """
        if not self.token_reduce_func:
            # 无剪枝，直接返回
            return image_features, [image_features.shape[1]] * image_features.shape[0]
        elif 'TRIM' in self.token_reduce_func:
            batch_size, tokens_number, dimension = image_features.shape
            # 从 'TRIM:0.5' 中提取 k（保留比例）
            k = float(self.token_reduce_func.replace('TRIM:', ''))

            # 获取图像嵌入（需经过 CLIP 的 post_layernorm 和 visual_projection）
            with torch.cuda.amp.autocast(enabled=False):
                # 确保 dtype 与模型一致（避免 mixed precision 问题）
                if image_features.dtype == torch.float16:
                    image_features = image_features.to(torch.float32)
                elif image_features.dtype == torch.float32:
                    # 转换为模型权重的 dtype（如 bfloat16）
                    target_dtype = self.vision_tower.vision_model.post_layernorm.weight.dtype
                    image_features = image_features.to(target_dtype)

                # 应用 post_layernorm（CLIP 视觉模型最后的 LayerNorm）
                normalized_image_features = self.vision_tower.vision_model.post_layernorm(image_features)
                # 投影到 CLIP 的共享嵌入空间
                proj_image_features = self.vision_tower.visual_projection(normalized_image_features)

            # 计算每个 patch 与文本的相似度（点积）
            # text_features: [B, D] → unsqueeze(2) → [B, D, 1]
            # proj_image_features: [B, N, D]
            # 结果: [B, N]
            similarities = torch.matmul(proj_image_features, text_features.unsqueeze(2)).squeeze(2)

            similarities = -similarities
            similarities = F.softmax(similarities, dim=-1)  # [B, N]

            # 动态确定 k（若 k == -1，则通过离群点检测自动计算保留比例）
            if k == -1:
                def outlier_detection(sim):
                    """使用 IQR 方法检测离群点比例作为保留率"""
                    sim_np = sim.to(dtype=torch.float32).cpu().numpy().flatten()
                    Q1 = np.percentile(sim_np, 25)
                    Q3 = np.percentile(sim_np, 75)
                    IQR = Q3 - Q1
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_indices = np.where(sim_np > upper_bound)[0]
                    ratio = len(outlier_indices) / len(sim_np)
                    return ratio

                k = outlier_detection(similarities)

            # 计算要保留的 token 数量
            num_tokens_to_keep = int(tokens_number * k)
            num_tokens_to_keep = max(1, min(num_tokens_to_keep, tokens_number))  # 边界保护

            # 选取 top-k 相似度最高的 tokens
            topk_values, topk_indices = torch.topk(
                similarities, num_tokens_to_keep, dim=1, largest=True, sorted=False
            )

            # 初始化输出特征（形状同输入）
            selected_image_features = torch.zeros_like(image_features)
            batch_indices = torch.arange(batch_size, device=image_features.device).unsqueeze(1)
            token_mask = torch.zeros(batch_size, tokens_number, device=image_features.device, dtype=torch.bool)
            token_mask[batch_indices, topk_indices] = True

            # 将选中的 tokens 放到前面
            selected_image_features[:, :num_tokens_to_keep] = image_features[token_mask].view(batch_size, num_tokens_to_keep, -1)

            # 处理剩余 tokens：计算均值作为聚合 token
            remaining_mask = torch.logical_not(token_mask)
            if remaining_mask.sum(dim=1).min().item() > 0:
                # 对每个样本，计算未选中 tokens 的加权平均（这里简单用算术平均）
                remaining_sum = (image_features * remaining_mask.unsqueeze(-1)).sum(dim=1)
                remaining_count = remaining_mask.sum(dim=1, keepdim=True).clamp(min=1)
                remaining_mean = remaining_sum / remaining_count
            else:
                remaining_mean = torch.zeros(batch_size, dimension, device=image_features.device)

            # 将聚合 token 放在第 num_tokens_to_keep 位置
            selected_image_features[:, num_tokens_to_keep] = remaining_mean

            # 实际维度：保留的 tokens + 1 个聚合 token
            actual_dims = [num_tokens_to_keep + 1] * batch_size
            image_features = selected_image_features

            # 恢复原始 dtype（如 float16）
            if image_features.dtype == torch.float32:
                image_features = image_features.to(torch.float16)

        elif 'TRAC' in self.token_reduce_func:
            batch_size, tokens_number, dimension = image_features.shape
            k_str = self.token_reduce_func.replace('TRAC:', '')
            if k_str == '-1':
                k = -1
            else:
                k = float(k_str)

            if llm_attention_scores is None:
                raise ValueError("TRAC mode requires `llm_attention_scores` as input to forward().")

            # 支持 [B, M, N] → [B, N]
            if llm_attention_scores.dim() == 3:
                scores = llm_attention_scores.sum(dim=1)  # 或 mean
            elif llm_attention_scores.dim() == 2:
                scores = llm_attention_scores
            else:
                raise ValueError(f"Invalid llm_attention_scores shape: {llm_attention_scores.shape}")

            assert scores.shape == (batch_size, tokens_number)

            # ===== 新增：k = -1 时自动计算保留比例 =====
            if k == -1:
                def outlier_detection(score_tensor):
                    # score_tensor: [B, N]
                    # 在每个样本上独立计算，或在 batch 上统一计算（这里选择 per-sample）
                    ratios = []
                    for b in range(score_tensor.shape[0]):
                        sim_np = score_tensor[b].to(dtype=torch.float32).cpu().numpy()
                        Q1 = np.percentile(sim_np, 25)
                        Q3 = np.percentile(sim_np, 75)
                        IQR = Q3 - Q1
                        upper_bound = Q3 + 1.5 * IQR
                        outlier_indices = np.where(sim_np > upper_bound)[0]
                        ratio = len(outlier_indices) / len(sim_np)
                        ratios.append(max(ratio, 1e-3))  # 至少保留 1 个 token
                    return torch.tensor(ratios, device=score_tensor.device)

                # 获取每个样本的保留比例
                k_per_sample = outlier_detection(scores)  # [B]
                num_tokens_to_keep_list = (k_per_sample * tokens_number).clamp(min=1).long()  # [B]
                max_tokens_to_keep = num_tokens_to_keep_list.max().item()
            else:
                num_tokens_to_keep = int(tokens_number * k)
                num_tokens_to_keep = max(1, min(num_tokens_to_keep, tokens_number))
                num_tokens_to_keep_list = None
                max_tokens_to_keep = num_tokens_to_keep
            # ==========================================

            # 构建输出张量（统一长度，便于后续处理）
            selected_image_features = torch.zeros(
                batch_size, max_tokens_to_keep + 1, dimension,
                device=image_features.device, dtype=image_features.dtype
            )

            for b in range(batch_size):
                importance = scores[b]  # [N]
                if k == -1:
                    num_keep = num_tokens_to_keep_list[b].item()
                else:
                    num_keep = max_tokens_to_keep

                # 选取 top-k
                topk_vals, topk_idxs = torch.topk(importance, num_keep, largest=True, sorted=False)
                mask = torch.zeros(tokens_number, dtype=torch.bool, device=image_features.device)
                mask[topk_idxs] = True

                # 填入选中 tokens
                selected_image_features[b, :num_keep] = image_features[b][mask]

                # 合并剩余 tokens（加权平均）
                remaining_mask = ~mask
                if remaining_mask.any():
                    weights = importance * remaining_mask.float()
                    weighted_sum = (image_features[b] * weights.unsqueeze(-1)).sum(dim=0)
                    weight_sum = weights.sum().clamp(min=1e-8)
                    merged = weighted_sum / weight_sum
                else:
                    merged = torch.zeros(dimension, device=image_features.device, dtype=image_features.dtype)

                selected_image_features[b, num_keep] = merged

            # 实际维度（用于后续处理，如 projector）
            if k == -1:
                actual_dims = (num_tokens_to_keep_list + 1).tolist()
            else:
                actual_dims = [max_tokens_to_keep + 1] * batch_size

            image_features = selected_image_features

        else:
            raise ValueError(f'Unknown Token Reduction Function::{self.token_reduce_func}')
        return image_features, actual_dims

    @torch.no_grad()
    def forward(self, images, texts=None, llm_attention_scores=None):
        text_features = None

        # 判断是否为 TRAC 模式且未提供 llm_attention_scores
        is_trac_mode = self.token_reduce_func and 'TRAC' in self.token_reduce_func
        need_compute_attention = is_trac_mode and llm_attention_scores is None

        if isinstance(images, list):
            image_features = []
            all_image_features = []
            all_attentions = [] if need_compute_attention else None

            for image in images:
                if need_compute_attention:
                    # 启用 output_attentions
                    image_forward_out = self.vision_tower(
                        image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                        output_attentions=True  # 👈 关键
                    )
                    all_attentions.append(image_forward_out.attentions)
                else:
                    image_forward_out = self.vision_tower(
                        image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True
                    )
                image_feature, all_image_feature = self.feature_select(image_forward_out)
                image_features.append(image_feature)
                all_image_features.append(all_image_feature)

            image_features = torch.cat(image_features, dim=0).to(images[0].dtype)
            all_image_features = torch.cat(all_image_features, dim=0).to(images[0].dtype)

            if need_compute_attention:
                # 合并 attentions: list of (num_layers, 1, num_heads, L, L)
                # 取最后一层
                batch_attentions = []
                for attn_tuple in all_attentions:
                    # attn_tuple: tuple of (layer_attentions)
                    last_layer_attn = attn_tuple[-1]  # [1, H, L, L]
                    batch_attentions.append(last_layer_attn)
                combined_attentions = torch.cat(batch_attentions, dim=0)  # [B, H, L, L]
                # L = 1 (cls) + N (patches)
                N = image_features.shape[1]
                # [CLS] index = 0, patch indices = 1 to N
                cls_to_patch_attn = combined_attentions[:, :, 0, 1:N+1]  # [B, H, N]
                # 平均所有 heads
                llm_attention_scores = cls_to_patch_attn.mean(dim=1)  # [B, N]

            if self.token_reduce_func and 'TRIM' in self.token_reduce_func:
                text_inputs = self.text_tokenizer(
                    text=texts, return_tensors="pt", truncation=True, padding=True
                )
                text_inputs = {k: v.to(device=image_features.device) for k, v in text_inputs.items()}
                text_features = self.text_tower(**text_inputs, output_hidden_states=False).text_embeds

        else:
            if need_compute_attention:
                image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True,
                    output_attentions=True
                )
            else:
                image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True
                )
            image_features, all_image_features = self.feature_select(image_forward_outs)
            image_features = image_features.to(images.dtype)
            all_image_features = all_image_features.to(images.dtype)

            if need_compute_attention:
                # 提取最后一层 attention
                last_layer_attn = image_forward_outs.attentions[-1]  # [B, H, L, L]
                N = image_features.shape[1]
                cls_to_patch_attn = last_layer_attn[:, :, 0, 1:N+1]  # [B, H, N]
                llm_attention_scores = cls_to_patch_attn.mean(dim=1)  # [B, N]

            if self.token_reduce_func and 'TRIM' in self.token_reduce_func:
                text_inputs = self.text_tokenizer(
                    text=texts, return_tensors="pt", truncation=True, padding=True
                )
                text_inputs = {k: v.to(device=image_features.device) for k, v in text_inputs.items()}
                text_features = self.text_tower(**text_inputs, output_hidden_states=False).text_embeds

        image_features, actual_dims = self.token_reduction(
            image_features, all_image_features,
            text_features=text_features,
            llm_attention_scores=llm_attention_scores
        )

        return image_features, actual_dims

    # 以下为属性方法，用于获取模型元信息
    @property
    def dummy_feature(self):
        """返回一个虚拟特征（用于占位）"""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class CLIPVisionTowerS2(CLIPVisionTower):
    """
    支持 S2（Scaling on Scales）多尺度推理的视觉塔。
    通过将大图切分为多个尺度（如 336, 672, 1008）进行特征提取，再融合。
    """

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        # 获取多尺度列表，如 '336,672,1008'
        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]      # 最小切分尺寸
        self.s2_image_size = self.s2_scales[-1]     # 最大图像尺寸

        # 动态导入 S2 多尺度推理函数
        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError(
                'Package s2wrapper not found! Please install by running: \n'
                'pip install git+https://github.com/bfshi/scaling_on_scales.git'
            )
        self.multiscale_forward = multiscale_forward

        # 修改图像预处理器的 resize/crop 尺寸为最大尺度
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        """重载 load_model，确保预处理器尺寸正确"""
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        # 设置预处理器尺寸为 S2 最大尺寸
        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        """单尺度特征提取（供 S2 调用）"""
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs)[0]  # 只取 image_features
        return image_features.to(images.dtype)

    @torch.no_grad()
    def forward(self, images):
        """
        使用 S2 多尺度策略处理图像。
        对每张图像，在多个尺度上切分、提取特征、再拼接。
        """
        if isinstance(images, list):
            image_features = []
            for image in images:
                # 对单张图像应用多尺度推理
                image_feature = self.multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size
                )
                image_features.append(image_feature)
            # 注意：S2 返回的特征已是拼接后的 [1, N_total, D]
            image_features = torch.cat(image_features, dim=0)
        else:
            image_features = self.multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size
            )

        return image_features

    @property
    def hidden_size(self):
        """S2 输出特征维度 = 原始 hidden_size × 尺度数量"""
        return self.config.hidden_size * len(self.s2_scales)