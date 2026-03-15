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
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.token_reduce_func = getattr(args, 'mm_vision_token_reduce_func', None)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        if self.token_reduce_func and ('TRIM' in self.token_reduce_func or 'CRTP' in self.token_reduce_func):
            if device_map:
                self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name)
                self.vision_tower.visual_projection = self.clip_model.visual_projection.to(self.device)
                self.clip_model.requires_grad_(False)
            else:
                self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name,
                                                                                  device_map=device_map)

            self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)
            self.text_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        all_image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = all_image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = all_image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features, all_image_features

    def _compute_geometric_median(self, X, eps=1e-6, max_iter=2):
        """
        Approximate geometric median using Weiszfeld algorithm (2 iterations is sufficient).
        X: [M, D]
        Returns: [D]
        """
        if X.shape[0] == 1:
            return X[0]
        y = X.mean(dim=0)
        for _ in range(max_iter):
            distances = torch.norm(X - y, dim=1, keepdim=True)
            nonzero = distances > eps
            if not nonzero.any():
                break
            weights = torch.where(nonzero, 1.0 / torch.clamp(distances, min=eps), torch.zeros_like(distances))
            y_new = (weights * X).sum(dim=0) / weights.sum()
            if torch.norm(y - y_new) < eps:
                y = y_new
                break
            y = y_new
        return y

    def token_reduction(self, image_features, all_image_features, text_features=None, cls_attention_scores=None):
        if not self.token_reduce_func:
            return image_features, [image_features.shape[1]] * image_features.shape[0]
        elif 'TRIM' in self.token_reduce_func:
            # ============ 保留原始 TRIM 逻辑不变 ============
            batch_size, tokens_number, dimension = image_features.shape
            k = float(self.token_reduce_func.replace('TRIM:', ''))

            with torch.cuda.amp.autocast(enabled=False):
                if image_features.dtype == torch.float16:
                    image_features = image_features.to(torch.float32)
                target_dtype = self.vision_tower.vision_model.post_layernorm.weight.dtype
                image_features = image_features.to(target_dtype)

                normalized_image_features = self.vision_tower.vision_model.post_layernorm(image_features)
                proj_image_features = self.vision_tower.visual_projection(normalized_image_features)

            similarities = torch.matmul(proj_image_features, text_features.unsqueeze(2)).squeeze(2)
            similarities = -similarities  # confirmed correct for layer -2
            similarities = F.softmax(similarities, dim=-1)

            if k == -1:
                def outlier_detection(sim):
                    sim_np = sim.to(dtype=torch.float32).cpu().numpy().flatten()
                    Q1 = np.percentile(sim_np, 25)
                    Q3 = np.percentile(sim_np, 75)
                    IQR = Q3 - Q1
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_indices = np.where(sim_np > upper_bound)[0]
                    ratio = len(outlier_indices) / len(sim_np)
                    return ratio

                k = outlier_detection(similarities)

            num_tokens_to_keep = int(tokens_number * k)
            num_tokens_to_keep = max(1, min(num_tokens_to_keep, tokens_number))

            topk_values, topk_indices = torch.topk(
                similarities, num_tokens_to_keep, dim=1, largest=True, sorted=False
            )

            selected_image_features = torch.zeros_like(image_features)
            batch_indices = torch.arange(batch_size, device=image_features.device).unsqueeze(1)
            token_mask = torch.zeros(batch_size, tokens_number, device=image_features.device, dtype=torch.bool)
            token_mask[batch_indices, topk_indices] = True

            selected_image_features[:, :num_tokens_to_keep] = image_features[token_mask].view(batch_size,
                                                                                              num_tokens_to_keep, -1)

            remaining_mask = torch.logical_not(token_mask)
            if remaining_mask.sum(dim=1).min().item() > 0:
                remaining_sum = (image_features * remaining_mask.unsqueeze(-1)).sum(dim=1)
                remaining_count = remaining_mask.sum(dim=1, keepdim=True).clamp(min=1)
                remaining_mean = remaining_sum / remaining_count
            else:
                remaining_mean = torch.zeros(batch_size, dimension, device=image_features.device)

            selected_image_features[:, num_tokens_to_keep] = remaining_mean
            actual_dims = [num_tokens_to_keep + 1] * batch_size
            image_features = selected_image_features

            # if image_features.dtype == torch.float32:
            #     image_features = image_features.to(torch.float16)

        elif 'CRTP' in self.token_reduce_func:
            # ============ CRTP 实现 (使用 RRF) ============
            batch_size, tokens_number, dimension = image_features.shape
            k_str = self.token_reduce_func.replace('CRTP:', '')
            use_auto = (k_str == '-1')

            if cls_attention_scores is None:
                raise ValueError("CRTP mode requires `cls_attention_scores` as input to forward().")

            # 线索 1: 语义相似度（来自 TRIM）
            with torch.cuda.amp.autocast(enabled=False):
                if image_features.dtype == torch.float16:
                    image_features_fp32 = image_features.to(torch.float32)
                target_dtype = self.vision_tower.vision_model.post_layernorm.weight.dtype
                image_features_fp32 = image_features_fp32.to(target_dtype)

                normalized_image_features = self.vision_tower.vision_model.post_layernorm(image_features_fp32)
                proj_image_features = self.vision_tower.visual_projection(normalized_image_features)
                similarities = torch.matmul(proj_image_features, text_features.unsqueeze(2)).squeeze(2)
                similarities = -similarities  # 对于 -2 层必须保留负号
                S = similarities  # [B, N]

            # 线索 2: [CLS] attention
            A = cls_attention_scores  # [B, N]

            # 线索 3: 冗余度（局部特征距离）
            # 构建空间邻域：假设 tokens 是 (H, W) 网格
            h = w = int(tokens_number ** 0.5)
            assert h * w == tokens_number, "Patch number must be square for spatial neighborhood"
            R = torch.zeros_like(S)
            for b in range(batch_size):
                feats = image_features[b]  # [N, D]
                norms = torch.norm(feats, dim=1, keepdim=True)
                cos_sim = torch.mm(feats, feats.t()) / (norms @ norms.t() + 1e-8)  # [N, N]
                # 对每个 patch，计算 4-邻域平均相似度
                red_vals = []
                for idx in range(tokens_number):
                    i, j = idx // w, idx % w
                    neighbors = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            n_idx = ni * w + nj
                            neighbors.append(cos_sim[idx, n_idx])
                    if neighbors:
                        mean_sim = torch.stack(neighbors).mean()
                    else:
                        mean_sim = torch.tensor(0.0, device=cos_sim.device)
                    red_vals.append(1.0 - mean_sim)
                R[b] = torch.stack(red_vals)

            # Step 2: Reciprocal Rank Fusion (RRF)
            # RRF score for item i = sum( 1 / (k + rank_i) ) for each ranking where i appears
            # Common choice is k=60. Higher RRF score means higher overall rank (more important).
            k_rrf = 60.0

            def get_rrf_score(ranks):
                # ranks: [B, N] where value is the rank (1 is best)
                # returns RRF scores [B, N]
                # RRF score = sum( 1 / (k + rank) )
                scores = 1.0 / (k_rrf + ranks.float())  # Convert rank to RRF contribution
                return scores

            rank_S = torch.argsort(torch.argsort(S, dim=1, descending=True), dim=1) + 1  # [B, N] -> [B, N] (ranks)
            rank_A = torch.argsort(torch.argsort(A, dim=1, descending=True), dim=1) + 1  # [B, N] -> [B, N] (ranks)
            rank_R = torch.argsort(torch.argsort(R, dim=1, descending=True), dim=1) + 1  # [B, N] -> [B, N] (ranks)

            rrf_score_S = get_rrf_score(rank_S)  # [B, N]
            rrf_score_A = get_rrf_score(rank_A)  # [B, N]
            rrf_score_R = get_rrf_score(rank_R)  # [B, N]

            V = rrf_score_S + rrf_score_A + rrf_score_R  # [B, N] # Final RRF score is sum of contributions
            # Note: Higher V means higher importance (unlike raw rank where lower is better)

            # Step 3: IQR + 冗余保护 (使用 RRF 分数 V)
            selected_masks = []
            actual_dims = []
            REDUNDANCY_THRESHOLD = 0.5  # 固定阈值，可调整

            for b in range(batch_size):
                scores = V[b]  # Higher score is better
                # IQR on RRF scores: Find outliers (high scores)
                scores_np = scores.cpu().numpy()
                Q1 = np.percentile(scores_np, 25)
                Q3 = np.percentile(scores_np, 75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 1.5 * IQR
                core_mask = scores >= upper_bound  # Select tokens with high RRF scores

                # 冗余保护：强制保留高冗余度 token
                unique_mask = R[b] > REDUNDANCY_THRESHOLD
                final_mask = core_mask | unique_mask
                selected_masks.append(final_mask)
                actual_dims.append(final_mask.sum().item() + 1)  # +1 for agg token

            max_len = max(actual_dims) - 1  # 不含 agg token
            selected_image_features = torch.zeros(
                batch_size, max_len + 1, dimension,
                device=image_features.device, dtype=image_features.dtype
            )

            for b in range(batch_size):
                mask = selected_masks[b]
                num_keep = mask.sum().item()
                selected_image_features[b, :num_keep] = image_features[b][mask]

                # Step 4: 几何中位数聚合残余
                rest_mask = ~mask
                if rest_mask.any():
                    rest_feats = image_features[b][rest_mask]
                    agg_token = self._compute_geometric_median(rest_feats)
                else:
                    agg_token = torch.zeros(dimension, device=image_features.device, dtype=image_features.dtype)
                selected_image_features[b, num_keep] = agg_token

            image_features = selected_image_features

        else:
            raise ValueError(f'Unknown Token Reduction Function::{self.token_reduce_func}')
        return image_features, actual_dims

    @torch.no_grad()
    def forward(self, images, texts=None, cls_attention_scores=None):
        text_features = None
        is_crtp_mode = self.token_reduce_func and 'CRTP' in self.token_reduce_func
        need_compute_attention = is_crtp_mode and cls_attention_scores is None

        if isinstance(images, list):
            image_features = []
            all_image_features = []
            all_attentions = [] if need_compute_attention else None

            for image in images:
                if need_compute_attention:
                    image_forward_out = self.vision_tower(
                        image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                        output_attentions=True
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
                batch_attentions = []
                for attn_tuple in all_attentions:
                    last_layer_attn = attn_tuple[-2]
                    batch_attentions.append(last_layer_attn)
                combined_attentions = torch.cat(batch_attentions, dim=0)
                N = image_features.shape[1]
                cls_to_patch_attn = combined_attentions[:, :, 0, 1:N + 1]
                cls_attention_scores = cls_to_patch_attn.max(dim=1).values

            if self.token_reduce_func and ('TRIM' in self.token_reduce_func or 'CRTP' in self.token_reduce_func):
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
                last_layer_attn = image_forward_outs.attentions[-2]
                N = image_features.shape[1]
                cls_to_patch_attn = last_layer_attn[:, :, 0, 1:N + 1]
                cls_attention_scores = cls_to_patch_attn.max(dim=1).values

            if self.token_reduce_func and ('TRIM' in self.token_reduce_func or 'CRTP' in self.token_reduce_func):
                text_inputs = self.text_tokenizer(
                    text=texts, return_tensors="pt", truncation=True, padding=True
                )
                text_inputs = {k: v.to(device=image_features.device) for k, v in text_inputs.items()}
                text_features = self.text_tower(**text_inputs, output_hidden_states=False).text_embeds

        image_features, actual_dims = self.token_reduction(
            image_features, all_image_features,
            text_features=text_features,
            cls_attention_scores=cls_attention_scores
        )

        return image_features, actual_dims

    @property
    def dummy_feature(self):
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
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError(
                'Package s2wrapper not found! Please install by running: \n'
                'pip install git+https://github.com/bfshi/scaling_on_scales.git    '
            )
        self.multiscale_forward = multiscale_forward

        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs)[0]
        return image_features.to(images.dtype)

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size
                )
                image_features.append(image_feature)
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
        return self.config.hidden_size * len(self.s2_scales)
