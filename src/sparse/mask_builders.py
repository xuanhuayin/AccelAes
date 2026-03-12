"""
Mask builders for sparse compute experiments.

Implements:
  1. ComplexityMaskBuilder: SDiT-style complexity-based mask (Scharr gradient on latent)
  2. SemanticMaskBuilder: Our semantic/aesthetic mask (token embeddings + cross-attn heatmap)
"""

import math
import torch
import torch.nn.functional as F
import numpy as np


class ComplexityMaskBuilder:
    """
    SDiT-style complexity mask (simplified, no Quickshift).

    Pipeline:
      1. Compute per-pixel complexity C(x) via Scharr gradient magnitude on latent
      2. Divide latent into grid blocks (e.g., 8x8 patches per block)
      3. Compute block-level complexity as mean of top-q% pixels in each block
      4. Select top-K blocks by complexity (K = ratio * total_blocks)
      5. Return mask: 1 = foreground (high complexity), 0 = background
    """

    def __init__(self, grid_size: int = 8, top_q: float = 0.75, ratio: float = 0.25):
        """
        Args:
            grid_size: number of grid divisions per spatial dimension
            top_q: quantile for block complexity (top-q% pixels)
            ratio: fraction of blocks to mark as foreground
        """
        self.grid_size = grid_size
        self.top_q = top_q
        self.ratio = ratio

    def compute_complexity(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel complexity using Scharr gradients.

        Args:
            latent: (B, C, H, W) tensor

        Returns:
            complexity: (B, 1, H, W) gradient magnitude map
        """
        # Average across channels
        gray = latent.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Scharr kernels
        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                                dtype=latent.dtype, device=latent.device).reshape(1, 1, 3, 3)
        scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                                dtype=latent.dtype, device=latent.device).reshape(1, 1, 3, 3)

        gx = F.conv2d(gray, scharr_x, padding=1)
        gy = F.conv2d(gray, scharr_y, padding=1)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        return magnitude

    def build_mask(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Build complexity-based mask.

        Args:
            latent: (B, C, H, W) latent tensor

        Returns:
            mask: (H, W) float tensor, 1=foreground, 0=background
        """
        complexity = self.compute_complexity(latent)  # (B, 1, H, W)
        c = complexity[0, 0]  # (H, W)
        H, W = c.shape

        # Divide into grid blocks
        gh = max(1, H // self.grid_size)
        gw = max(1, W // self.grid_size)
        num_blocks_h = math.ceil(H / gh)
        num_blocks_w = math.ceil(W / gw)

        # Compute block complexity
        block_scores = []
        block_coords = []
        for bi in range(num_blocks_h):
            for bj in range(num_blocks_w):
                h_start = bi * gh
                h_end = min((bi + 1) * gh, H)
                w_start = bj * gw
                w_end = min((bj + 1) * gw, W)
                block = c[h_start:h_end, w_start:w_end]

                # Top-q% mean as block complexity
                flat = block.flatten()
                k = max(1, int(len(flat) * self.top_q))
                topk_vals = flat.topk(k).values
                score = topk_vals.mean()

                block_scores.append(score)
                block_coords.append((h_start, h_end, w_start, w_end))

        block_scores = torch.stack(block_scores)

        # Select top-K blocks
        num_blocks = len(block_scores)
        K = max(1, int(num_blocks * self.ratio))
        topk_indices = block_scores.topk(K).indices

        # Build mask
        mask = torch.zeros(H, W, device=latent.device, dtype=latent.dtype)
        for idx in topk_indices:
            h_start, h_end, w_start, w_end = block_coords[idx]
            mask[h_start:h_end, w_start:w_end] = 1.0

        return mask

    def build_mask_with_viz(self, latent: torch.Tensor) -> tuple:
        """Build mask and return (mask, complexity_map) for visualization."""
        complexity = self.compute_complexity(latent)
        mask = self.build_mask(latent)
        return mask, complexity[0, 0]


class SemanticMaskBuilder:
    """
    Semantic/aesthetic mask builder (our method).

    Pipeline:
      1. Use CLIP text encoder to compute semantic similarity between
         prompt words and aesthetic anchor phrases (CLIP excels at this
         because its embedding space is trained for semantic alignment,
         unlike autoregressive LMs like Gemma whose embeddings are
         optimized for next-token prediction)
      2. Map word-level importance back to the generation model's token positions
      3. Aggregate cross-attention maps (token→patch) weighted by semantic importance
      4. Normalize → threshold → soft mask (Gaussian blur)
    """

    # Data-driven aesthetic anchors (from Pick-a-Pic 37K unique prompts)
    # Merged from two calibration rounds; 34 anchors covering style/quality/subject.
    DEFAULT_ANCHORS = [
        # ---- 风格 / 渲染质量 ----
        "photorealistic",        # SD community high-freq, strong aesthetic signal
        "realistic",             # 2137x (5.7%)
        "cinematic",             # framing/lighting quality
        "highly detailed",       # quality descriptor
        "artistic",              # aesthetic judgment
        "masterpiece",           # quality superlative
        "professional photography",  # photographic quality
        "soft bokeh",            # photographic depth
        "bokeh",                 # photographic depth
        "dramatic lighting",     # lighting quality
        "fantasy art",           # style category
        "studio lighting",       # lighting quality
        # ---- 细节 / 清晰度 ----
        "detailed",              # 4634x (12.3%)
        "intricate",             # 1353x (3.6%)
        "sharp focus",           # 601x (1.6%)
        "sharp",                 # 431x (1.1%)
        # ---- 审美判断 ----
        "stunning",              # 234x
        "beautiful",             # 2321x (6.2%)
        "elegant",               # 391x (1.0%)
        "vivid",                 # 106x (0.3%)
        "vibrant",               # 291x
        # ---- 摄影 / 空间构图 ----
        "depth of field",        # 221x - foreground/background separation
        "volumetric lighting",   # 628x - spatial lighting
        "close up",              # 392x (1.0%)
        "portrait",              # 1992x (5.3%)
        "full body",             # 549x (1.5%)
        # ---- 主体指称 ----
        "main subject",          # direct subject anchor
        "foreground",            # composition, 109x
        "focused",               # focal region
        # ---- 内容级兜底 (fallback for pure-content prompts) ----
        "subject",               # generic subject
        "character",             # 人物/角色
        "object",                # 物体
        "figure",                # 人形/轮廓
        "person",                # 人物
    ]

    # Confidence threshold: if max CLIP cosine similarity across all
    # non-stopword content words is below this value, fall back to
    # uniform content-word importance (stopwords always get 0).
    # Calibrated on CLIP ViT-L/14 space:
    #   true anchor matches (intricate, realistic, etc.) > 0.70
    #   function word noise  (of, as, a, in)             < 0.60
    #   non-matching content (panda, cathedral, BMW)     < 0.45
    CONFIDENCE_THRESHOLD = 0.60

    # English stopwords — always get importance=0 because they have
    # misleadingly high CLIP similarity with anchors (e.g. "of"→"full body"=0.55)
    # yet carry no spatial/semantic meaning for mask generation.
    STOPWORDS = frozenset({
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "with",
        "and", "or", "but", "is", "are", "was", "were", "be", "been",
        "as", "by", "from", "that", "this", "it", "its", "not", "no",
        "so", "if", "do", "does", "did", "has", "have", "had", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "than", "then", "there", "here", "where", "when", "how", "what",
        "which", "who", "whom", "whose", "into", "over", "under",
        "between", "through", "about", "up", "down", "out", "very",
    })

    # Default CLIP model for semantic similarity computation
    CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

    def __init__(
        self,
        anchors: list = None,
        ratio: float = 0.25,
        blur_sigma: float = 1.0,
        top_m: int = 5,
        confidence_threshold: float = None,
        clip_model_name: str = None,
        region_method: str = "threshold",
        n_segments: int = 64,
    ):
        """
        Args:
            anchors: list of aesthetic anchor phrases
            ratio: fraction of patches to mark as foreground
            blur_sigma: Gaussian blur sigma for soft mask
            top_m: number of top semantically-relevant tokens to use
            confidence_threshold: min cosine similarity to trust anchor matching;
                if all tokens fall below this, fall back to uniform importance
            clip_model_name: CLIP model for semantic similarity computation
            region_method: how to segment heatmap into regions before thresholding.
                "threshold" (default): direct quantile threshold on heatmap pixels.
                "slic": SLIC superpixel segmentation, score segments by mean heatmap.
                "quickshift": Quickshift superpixel segmentation, same scoring.
            n_segments: approximate number of superpixel segments (for slic/quickshift).
        """
        self.anchors = anchors or self.DEFAULT_ANCHORS
        self.ratio = ratio
        self.blur_sigma = blur_sigma
        self.top_m = top_m
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None
            else self.CONFIDENCE_THRESHOLD
        )
        self.clip_model_name = clip_model_name or self.CLIP_MODEL_NAME
        self.region_method = region_method
        self.n_segments = n_segments
        self._anchor_embeddings = None
        self._clip_tokenizer = None
        self._clip_encoder = None

    def _load_clip(self, device="cuda"):
        """Lazy-load CLIP text encoder for semantic similarity computation."""
        if self._clip_tokenizer is None:
            from transformers import CLIPTokenizer, CLIPTextModel
            self._clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)
            self._clip_encoder = (
                CLIPTextModel.from_pretrained(self.clip_model_name, torch_dtype=torch.float16)
                .to(device).eval()
            )

    def compute_anchor_embeddings(self, tokenizer=None, text_encoder=None, device="cuda"):
        """
        Pre-compute anchor phrase embeddings using CLIP.

        Args:
            tokenizer: ignored (kept for backward compatibility)
            text_encoder: ignored (kept for backward compatibility)
            device: torch device
        """
        self._load_clip(device)
        with torch.no_grad():
            tokens = self._clip_tokenizer(
                self.anchors,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            outputs = self._clip_encoder(**tokens)
            embeds = outputs.last_hidden_state.float()
            # Exclude BOS/EOS for cleaner anchor representations
            clip_bos = self._clip_tokenizer.bos_token_id
            clip_eos = self._clip_tokenizer.eos_token_id
            content_mask = tokens["attention_mask"].clone()
            for sid in (clip_bos, clip_eos):
                if sid is not None:
                    content_mask[tokens["input_ids"] == sid] = 0
            mask = content_mask.unsqueeze(-1).float()
            # Avoid division by zero for single-token anchors
            mask_sum = mask.sum(dim=1).clamp(min=1.0)
            pooled = (embeds * mask).sum(dim=1) / mask_sum
            self._anchor_embeddings = F.normalize(pooled, dim=-1)

    @staticmethod
    def _get_special_ids(tokenizer):
        """Collect special token IDs from a tokenizer."""
        special_ids = set()
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_ids.add(tid)
        return special_ids

    def _group_tokens_to_words(self, input_ids, tokenizer):
        """
        Group subword tokens into whole words using leading-space heuristic.

        SentencePiece / Gemma convention: tokens starting a new word have a
        leading space (or ▁). Tokens continuing the previous word do not.

        Returns:
            words: list of word strings (whitespace-stripped)
            word_indices: list[list[int]], token indices for each word
        """
        special_ids = self._get_special_ids(tokenizer)
        words = []
        word_indices = []
        current_chars = []
        current_indices = []

        for i, tid in enumerate(input_ids):
            tid_val = tid.item() if hasattr(tid, "item") else tid
            if tid_val in special_ids:
                if current_chars:
                    words.append("".join(current_chars))
                    word_indices.append(list(current_indices))
                    current_chars, current_indices = [], []
                continue

            token_str = tokenizer.decode([tid_val])

            # Leading space or ▁ indicates new word
            if token_str and (token_str[0] in (" ", "\u2581")):
                if current_chars:
                    words.append("".join(current_chars))
                    word_indices.append(list(current_indices))
                current_chars = [token_str.lstrip(" \u2581")]
                current_indices = [i]
            elif not current_chars:
                # First content token (may lack leading space)
                current_chars = [token_str]
                current_indices = [i]
            else:
                # Continuation of current word
                current_chars.append(token_str)
                current_indices.append(i)

        if current_chars:
            words.append("".join(current_chars))
            word_indices.append(list(current_indices))

        return words, word_indices

    def compute_token_importance(
        self,
        prompt: str,
        prompt_mask: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer,
    ) -> torch.Tensor:
        """
        Compute per-token semantic importance using CLIP, aligned to the
        generation model's (e.g. Gemma) token positions.

        Steps:
          1. Group model tokens into whole words
          2. Encode each word with CLIP → compute cosine similarity with anchors
          3. Map word-level importance back to model token positions
          4. Apply confidence threshold; fall back to uniform if no anchor matches

        Args:
            prompt: raw prompt string
            prompt_mask: (1, seq_len) model's attention mask
            input_ids: (1, seq_len) model's token IDs
            tokenizer: model's tokenizer (for decoding subwords to words)

        Returns:
            importance: (seq_len,) importance scores in [0, 1]
        """
        if self._anchor_embeddings is None:
            raise RuntimeError("Call compute_anchor_embeddings() first")

        device = self._anchor_embeddings.device
        seq_len = prompt_mask.shape[1]
        ids = input_ids[0]

        # 1. Group model tokens into words
        words, word_indices = self._group_tokens_to_words(ids, tokenizer)

        # Build content mask (non-special, non-padding)
        special_ids = self._get_special_ids(tokenizer)
        content_mask = prompt_mask[0].float().clone()
        for i in range(len(ids)):
            if ids[i].item() in special_ids:
                content_mask[i] = 0.0

        if not words:
            return content_mask

        # Identify stopwords — they always get importance=0
        is_stopword = [
            w.strip(",.;:!?'\"()[]{}").lower() in self.STOPWORDS
            or len(w.strip(",.;:!?'\"()[]{}")) <= 1  # single chars like ","
            for w in words
        ]

        # Content word indices (non-stopword) for CLIP encoding
        content_word_idx = [i for i, stop in enumerate(is_stopword) if not stop]

        if not content_word_idx:
            # All words are stopwords — uniform content mask
            return content_mask

        content_words = [words[i] for i in content_word_idx]

        # 2. Encode content words with CLIP
        with torch.no_grad():
            clip_toks = self._clip_tokenizer(
                content_words, padding=True, truncation=True, return_tensors="pt",
            ).to(device)
            clip_out = self._clip_encoder(**clip_toks)
            clip_embeds = clip_out.last_hidden_state.float()

            # Exclude CLIP special tokens from word pooling
            clip_content = clip_toks["attention_mask"].clone()
            for sid in (self._clip_tokenizer.bos_token_id,
                        self._clip_tokenizer.eos_token_id):
                if sid is not None:
                    clip_content[clip_toks["input_ids"] == sid] = 0
            cmask = clip_content.unsqueeze(-1).float()
            cmask_sum = cmask.sum(dim=1).clamp(min=1.0)
            word_pooled = (clip_embeds * cmask).sum(dim=1) / cmask_sum
            word_pooled = F.normalize(word_pooled, dim=-1)

            # 3. Per-word similarity with anchors
            sim = word_pooled @ self._anchor_embeddings.T  # (num_content_words, num_anchors)
            cw_importance = sim.max(dim=-1).values  # (num_content_words,)

        # 4. Confidence check (only on content words, stopwords excluded)
        best_match = cw_importance.max().item()

        if best_match < self.confidence_threshold:
            # No strong anchor match — uniform importance for content words,
            # stopwords stay at 0. Mask is driven by cross-attention of
            # content words only (better than weighting all tokens equally).
            importance = torch.zeros(seq_len, device=device)
            for wi in content_word_idx:
                for idx in word_indices[wi]:
                    importance[idx] = 1.0
        else:
            # Map CLIP importance to model token positions
            # Stopword tokens stay at 0
            importance = torch.zeros(seq_len, device=device)
            for ci, wi in enumerate(content_word_idx):
                for idx in word_indices[wi]:
                    importance[idx] = cw_importance[ci]

            # Normalize content positions to [0, 1]
            active = importance[importance > 0]
            if len(active) > 0 and active.max() > active.min():
                cmin = active.min()
                crange = active.max() - cmin + 1e-8
                importance = torch.where(
                    importance > 0,
                    (importance - cmin) / crange,
                    importance,
                )

        return importance

    # Minimum heatmap standard deviation for a mask to be considered
    # spatially meaningful. Below this, the heatmap is too scattered and
    # we return None so the caller can fall back to uniform CFG.
    MIN_HEATMAP_STD = 0.12

    def build_mask_from_cross_attention(
        self,
        cross_attn_maps: list,
        token_importance: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor | None:
        """
        Build spatial mask from cross-attention maps + token importance.

        Args:
            cross_attn_maps: list of (num_heads, num_patches, seq_len) attention tensors
            token_importance: (seq_len,) importance weights
            latent_h, latent_w: spatial dimensions of latent

        Returns:
            mask: (latent_h, latent_w) soft mask in [0, 1], or None if
                  the heatmap lacks spatial structure (caller should use
                  uniform CFG).
        """
        num_patches = latent_h * latent_w

        # Average across layers and heads
        all_attn = torch.stack([m.mean(dim=0) for m in cross_attn_maps])  # (num_layers, patches, seq)
        avg_attn = all_attn.mean(dim=0)  # (patches, seq_len)

        # Weight by token importance (None → uniform weights)
        if token_importance is None:
            token_importance = torch.ones(avg_attn.shape[1],
                                          device=avg_attn.device,
                                          dtype=avg_attn.dtype) / avg_attn.shape[1]
        weighted = avg_attn * token_importance.unsqueeze(0)  # (patches, seq_len)
        heatmap = weighted.sum(dim=-1)  # (patches,)

        # Reshape to spatial
        heatmap = heatmap[:num_patches].reshape(latent_h, latent_w)

        # Normalize
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Quality check: if heatmap has no spatial structure, skip mask
        heatmap_std = heatmap.std().item()
        if heatmap_std < self.MIN_HEATMAP_STD:
            return None

        # Build binary mask using selected region method
        if self.region_method in ("slic", "quickshift"):
            mask = self._superpixel_mask(heatmap)
        else:
            # Default: direct quantile threshold
            threshold = torch.quantile(heatmap.flatten(), 1.0 - self.ratio)
            mask = (heatmap >= threshold).float()

        # Soft mask via Gaussian blur
        if self.blur_sigma > 0:
            mask = self._gaussian_blur(mask, sigma=self.blur_sigma)

        return mask

    def _superpixel_mask(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Build foreground mask using superpixel segmentation on the heatmap.

        Segments the heatmap into N superpixel regions, scores each region
        by its mean heatmap value, and selects the top regions covering
        `self.ratio` fraction of total patches.

        Args:
            heatmap: (H, W) normalized float tensor in [0, 1]

        Returns:
            mask: (H, W) binary float tensor, 1=foreground
        """
        import numpy as np
        from skimage.segmentation import slic, quickshift

        H, W = heatmap.shape
        hmap_np = heatmap.cpu().float().numpy()  # (H, W)

        # skimage segmentation needs (H, W, C) image; use 3-channel copy
        hmap_3ch = np.stack([hmap_np, hmap_np, hmap_np], axis=-1)

        if self.region_method == "slic":
            labels = slic(
                hmap_3ch,
                n_segments=self.n_segments,
                compactness=0.1,    # low compactness → shape follows heatmap, not grid
                sigma=1.0,
                start_label=0,
            )
        else:  # quickshift
            labels = quickshift(
                hmap_3ch,
                kernel_size=3,
                max_dist=6,
                ratio=0.5,
                convert2lab=False,
            )

        # Score each segment by mean heatmap value
        unique_labels = np.unique(labels)
        seg_scores = []
        for lbl in unique_labels:
            seg_mask = labels == lbl
            seg_scores.append((hmap_np[seg_mask].mean(), lbl))
        seg_scores.sort(key=lambda x: -x[0])  # descending

        # Select top segments covering `ratio` fraction of patches
        total = H * W
        target = int(total * self.ratio)
        selected = set()
        count = 0
        for score, lbl in seg_scores:
            if count >= target:
                break
            selected.add(lbl)
            count += int((labels == lbl).sum())

        # Build mask
        fg = np.isin(labels, list(selected)).astype(np.float32)
        return torch.from_numpy(fg).to(heatmap.device)

    def build_mask_from_token_importance_only(
        self,
        token_importance: torch.Tensor,
        cross_attn_map: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        """
        Fallback: build mask using a single cross-attention snapshot.

        Args:
            token_importance: (seq_len,) importance weights
            cross_attn_map: (num_heads, num_patches, seq_len) single layer attention
            latent_h, latent_w: spatial dims

        Returns:
            mask: (latent_h, latent_w) soft mask
        """
        num_patches = latent_h * latent_w
        attn = cross_attn_map.mean(dim=0)  # (patches, seq_len)

        # Select top-m important tokens
        topk = min(self.top_m, len(token_importance))
        top_indices = token_importance.topk(topk).indices

        # Aggregate attention to these tokens
        heatmap = attn[:, top_indices].sum(dim=-1)  # (patches,)
        heatmap = heatmap[:num_patches].reshape(latent_h, latent_w)

        # Normalize
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Threshold
        threshold = torch.quantile(heatmap.flatten(), 1.0 - self.ratio)
        mask = (heatmap >= threshold).float()

        # Gaussian blur for soft edges
        if self.blur_sigma > 0:
            mask = self._gaussian_blur(mask, sigma=self.blur_sigma)

        return mask

    @staticmethod
    def _gaussian_blur(mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian blur to a 2D mask."""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.reshape(1, 1, kernel_size, kernel_size)

        padded = mask.unsqueeze(0).unsqueeze(0)
        pad = kernel_size // 2
        padded = F.pad(padded, [pad, pad, pad, pad], mode="reflect")
        blurred = F.conv2d(padded, kernel_2d)
        return blurred.squeeze(0).squeeze(0).clamp(0, 1)
