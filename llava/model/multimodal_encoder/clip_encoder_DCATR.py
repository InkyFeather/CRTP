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

# 交叉注意力评分模块
class CrossAttentionScorer(nn.Module):
    """
    轻量交叉注意力评分器，用于动态评估视觉 token 对文本的重要性。
    """
    def __init__(self, vis_dim, txt_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = vis_dim // num_heads
        assert self.head_dim * num_heads == vis_dim, "vis_dim 必须能被 num_heads 整除"

        self.q_proj = nn.Linear(vis_dim, vis_dim)
        self.k_proj = nn.Linear(txt_dim, vis_dim)

        # 初始化（模仿 ViT/CLIP 风格）
        self.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.q_proj.bias.data.zero_()
        self.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.k_proj.bias.data.zero_()

    def forward(self, vis_feats, txt_feats):
        """
        vis_feats: [B, N_v, D_v]
        txt_feats: [B, N_t, D_t]  # 文本 token-level 特征（非全局嵌入）
        Returns:
            scores: [B, N_v]  # 每个视觉 token 的重要性得分
        """
        vis_feats = vis_feats.to(self.q_proj.weight.dtype)
        txt_feats = txt_feats.to(self.k_proj.weight.dtype)

        B, N_v, D_v = vis_feats.shape
        _, N_t, D_t = txt_feats.shape

        Q = self.q_proj(vis_feats)  # [B, N_v, D_v]
        K = self.k_proj(txt_feats)  # [B, N_t, D_v]

        # 多头拆分
        Q = Q.view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_v, d]
        K = K.view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_t, d]

        # 计算注意力分数
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N_v, N_t]
        # 对文本维度平均 -> 每个视觉 token 的综合相关性
        attn = attn.mean(dim=-1)  # [B, H, N_v]
        scores = attn.mean(dim=1)  # [B, N_v]
        return scores

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
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        # DCATR
        if self.token_reduce_func and 'DCATR' in self.token_reduce_func:
            # 需要文本的 token-level 特征（中间层）
            self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)
            self.text_tower.requires_grad_(False)

            # 初始化交叉注意力评分器
            vis_dim = self.vision_tower.config.hidden_size
            txt_dim = self.text_tower.config.hidden_size
            self.attn_scorer = CrossAttentionScorer(vis_dim, txt_dim, num_heads=4).to(self.device)
            self.attn_scorer = self.attn_scorer.to(dtype=self.vision_tower.dtype)
            self.attn_scorer.eval()
            for p in self.attn_scorer.parameters():
                p.requires_grad_(False)

        # TRIM
        elif self.token_reduce_func and 'TRIM' in self.token_reduce_func:
            if device_map:  # eval
                self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name)
                self.vision_tower.visual_projection = self.clip_model.visual_projection.to(self.device)
                self.clip_model.requires_grad_(False)
            else:
                self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)

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

    def token_reduction(self, image_features, all_image_features, text_features=None):
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
        # =================== DCATR ===================
        elif 'DCATR' in self.token_reduce_func:
            if text_features is None:
                raise ValueError("DCATR: text_features is None!")
            if text_features.dim() != 3:
                raise ValueError(f"DCATR expects text_features of shape [B, N_t, D_t], got {text_features.shape}")
            batch_size, tokens_number, dimension = image_features.shape
            k = float(self.token_reduce_func.replace('DCATR:', ''))
            k = max(0.01, min(k, 1.0))  # 保护比例

            # 使用交叉注意力计算重要性分数
            with torch.no_grad():
                scores = self.attn_scorer(image_features, text_features)  # [B, N_v]

            # 支持动态 IQR（k=-1）
            if k == -1:
                def outlier_detection(score_batch):
                    ratios = []
                    for s in score_batch:
                        s_np = s.cpu().numpy()
                        Q1, Q3 = np.percentile(s_np, [25, 75])
                        IQR = Q3 - Q1
                        upper = Q3 + 1.5 * IQR
                        ratio = np.mean(s_np > upper)
                        ratios.append(ratio)
                    return np.mean(ratios)

                k = outlier_detection(scores)

            num_tokens_to_keep = max(1, int(tokens_number * k))

            # 选择得分最高的 tokens（注意：DCATR 分数越高越重要）
            _, topk_indices = torch.topk(scores, num_tokens_to_keep, dim=1, largest=True, sorted=False)

            # 构建 mask 和输出
            selected_image_features = torch.zeros_like(image_features)
            batch_idx = torch.arange(batch_size, device=image_features.device).unsqueeze(1)
            token_mask = torch.zeros(batch_size, tokens_number, dtype=torch.bool, device=image_features.device)
            token_mask[batch_idx, topk_indices] = True

            selected_image_features[:, :num_tokens_to_keep] = image_features[token_mask].view(batch_size, num_tokens_to_keep, -1)

            # 聚合剩余 tokens
            remaining_mask = ~token_mask
            if remaining_mask.sum() > 0:
                remaining_sum = (image_features * remaining_mask.unsqueeze(-1)).sum(dim=1)
                remaining_count = remaining_mask.sum(dim=1, keepdim=True).clamp(min=1)
                remaining_mean = remaining_sum / remaining_count
            else:
                remaining_mean = torch.zeros(batch_size, dimension, device=image_features.device)

            selected_image_features[:, num_tokens_to_keep] = remaining_mean
            actual_dims = [num_tokens_to_keep + 1] * batch_size
            image_features = selected_image_features

            # 保持 dtype 一致
            if image_features.dtype == torch.float32:
                image_features = image_features.to(torch.float16)
        # =================== TRIM ===================
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

        else:
            raise ValueError(f'Unknown Token Reduction Function::{self.token_reduce_func}')
        return image_features, actual_dims

    @torch.no_grad()
    def forward(self, images, texts=None):
        """
        前向传播：提取图像特征，并可选地进行 TRIM 剪枝。
        
        Args:
            images: 单张图像 [C, H, W] 或图像列表
            texts: 对应的文本描述（用于 TRIM）
        
        Returns:
            image_features: 处理后的图像特征
            actual_dims: 每个样本的实际 token 数
        """
        text_features = None

        if isinstance(images, list):  # 多图处理
            image_features = []
            all_image_features = []

            # 使用 CUDA 流并行处理图像和文本（理论上可加速）
            image_stream = torch.cuda.Stream()
            text_stream = torch.cuda.Stream()

            with torch.cuda.stream(image_stream):
                for image in images:
                    # 注意：这里 image 是单张 [C, H, W]，需 unsqueeze(0) 变成 [1, C, H, W]
                    image_forward_out = self.vision_tower(
                        image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True
                    )
                    image_feature, all_image_feature = self.feature_select(image_forward_out)
                    # 修正：原代码此处有 bug，应 append 后再统一转换 dtype
                    image_features.append(image_feature)
                    all_image_features.append(all_image_feature)

            # 批量转换 dtype（更高效）
            if image_features:
                image_features = torch.cat(image_features, dim=0).to(images[0].dtype)
                all_image_features = torch.cat(all_image_features, dim=0).to(images[0].dtype)

            # =================== 为 DCATR 提供 token-level 文本特征 ===================
            if self.token_reduce_func and 'DCATR' in self.token_reduce_func:
                with torch.cuda.stream(text_stream):
                    if texts is None:
                        raise ValueError("DCATR requires `texts` but got None in multi-image forward.")
                    text_inputs = self.text_tokenizer(
                        text=texts, return_tensors="pt", truncation=True, padding=True
                    )
                    text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
                    text_outputs = self.text_tower(**text_inputs, output_hidden_states=True)
                    text_features = text_outputs.hidden_states[-2]  # [B, N_t, D_t]

            # =================== TRIM 的 text_embeds ===================
            elif self.token_reduce_func and 'TRIM' in self.token_reduce_func:
                text_inputs = self.text_tokenizer(text=texts, return_tensors="pt", truncation=True, padding=True)
                text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
                text_features = self.text_tower(**text_inputs, output_hidden_states=False).text_embeds  # [B, D]

            torch.cuda.synchronize()  # 等待两个流完成

        else:  # 单图处理
            image_stream = torch.cuda.Stream()
            text_stream = torch.cuda.Stream()

            with torch.cuda.stream(image_stream):
                image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True
                )
                image_features, all_image_features = self.feature_select(image_forward_outs)
                image_features = image_features.to(images.dtype)
                all_image_features = all_image_features.to(images.dtype)

            if self.token_reduce_func and ('DCATR' in self.token_reduce_func or 'TRIM' in self.token_reduce_func):
                if texts is None:
                    raise ValueError("texts is None but token reduction is enabled!")
                # 确保 texts 是 list
                if isinstance(texts, str):
                    texts = [texts]
                # 此时 texts 是 List[str]，长度应等于 batch_size
                batch_size = image_features.shape[0]
                if len(texts) != batch_size:
                    # 重复 texts 以匹配 batch_size（常见于 VQA）
                    if len(texts) == 1:
                        texts = texts * batch_size
                    else:
                        raise ValueError(f"texts length {len(texts)} != batch_size {batch_size}")

                text_inputs = self.text_tokenizer(
                    text=texts, return_tensors="pt", truncation=True, padding=True
                )
                text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

                if 'DCATR' in self.token_reduce_func:
                    text_outputs = self.text_tower(**text_inputs, output_hidden_states=True)
                    text_features = text_outputs.hidden_states[-2]  # [B, N_t, D_t]
                elif 'TRIM' in self.token_reduce_func:
                    text_features = self.text_tower(**text_inputs, output_hidden_states=False).text_embeds  # [B, D]

            torch.cuda.synchronize()

        # 执行 token 剪枝（TRIM）
        image_features, actual_dims = self.token_reduction(image_features, all_image_features, text_features)

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