"""
Token merging core operations for DiT inference acceleration.

Implements bipartite soft matching (ToMe-style) with foreground mask protection.
Background tokens are merged to reduce self-attention sequence length while
preserving full resolution for foreground regions.

Reference: Bolya et al., "Token Merging: Your ViT But Faster" (ICLR 2023)
"""

import torch


def bipartite_soft_matching_with_mask(
    metric: torch.Tensor,
    fg_mask: torch.Tensor | None,
    r: int,
    patch_h: int,
    patch_w: int,
) -> tuple:
    """
    Bipartite soft matching with foreground protection.

    Partitions tokens into A/B sets via spatial checkerboard, protects
    foreground tokens from merging, then greedily matches the top-r most
    similar B→A pairs and merges them by averaging.

    Args:
        metric: (B, N, C) token features for similarity (e.g. K vectors).
        fg_mask: (N,) bool tensor, True = foreground (protected).
                 If None, no tokens are protected.
        r: number of tokens to merge (remove). Clamped to available pairs.
        patch_h: spatial height of the token grid.
        patch_w: spatial width of the token grid.

    Returns:
        (merge_fn, unmerge_fn) closures:
          merge_fn(x: (B, N, ...)) → (B, N-r_actual, ...)
          unmerge_fn(x: (B, N-r_actual, ...)) → (B, N, ...)
    """
    B, N, C = metric.shape
    assert N == patch_h * patch_w, f"N={N} != patch_h*patch_w={patch_h * patch_w}"

    if r <= 0:
        return _identity_merge_unmerge(B, N, metric.device)

    # --- 1. Checkerboard partition into A and B sets ---
    # A: (row+col) % 2 == 0, B: (row+col) % 2 == 1
    rows = torch.arange(patch_h, device=metric.device)
    cols = torch.arange(patch_w, device=metric.device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    parity = ((grid_r + grid_c) % 2).reshape(-1)  # (N,)

    a_indices = (parity == 0).nonzero(as_tuple=True)[0]  # indices in [0, N)
    b_indices = (parity == 1).nonzero(as_tuple=True)[0]

    # --- 2. Protect foreground tokens: only allow merging unprotected B tokens ---
    if fg_mask is not None:
        fg_mask = fg_mask.bool()
        # B tokens that are NOT foreground can be merged
        b_mergeable_mask = ~fg_mask[b_indices]  # True = can merge
        b_mergeable = b_indices[b_mergeable_mask]

        # A tokens that are NOT foreground can be targets
        a_target_mask = ~fg_mask[a_indices]
        a_targets = a_indices[a_target_mask]
    else:
        b_mergeable = b_indices
        a_targets = a_indices

    if len(b_mergeable) == 0 or len(a_targets) == 0:
        return _identity_merge_unmerge(B, N, metric.device)

    # Clamp r to available mergeable tokens
    r = min(r, len(b_mergeable))

    # --- 3. Compute similarity: mergeable B → target A ---
    # Normalize for cosine similarity
    metric_norm = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)

    # (B, num_b_mergeable, C) @ (B, C, num_a_targets) → (B, num_b_mergeable, num_a_targets)
    b_feats = metric_norm[:, b_mergeable]
    a_feats = metric_norm[:, a_targets]
    scores = torch.bmm(b_feats, a_feats.transpose(1, 2))

    # Average scores across batch for consistent merging
    scores_avg = scores.mean(dim=0)  # (num_b_mergeable, num_a_targets)

    # --- 4. Greedy matching: each B token matched to its most similar A target ---
    # For each B token, find best A target
    max_scores, best_a_local = scores_avg.max(dim=-1)  # (num_b_mergeable,)

    # Select top-r B tokens with highest match scores
    _, topk_b_local = max_scores.topk(r, largest=True)

    # Map local indices back to global
    merged_b_global = b_mergeable[topk_b_local]  # (r,) global indices of B tokens to remove
    matched_a_global = a_targets[best_a_local[topk_b_local]]  # (r,) global indices of A tokens to merge into

    # --- 5. Build merge/unmerge maps ---
    # After merge: kept tokens = all tokens except merged_b_global
    # Order: keep original order for stable indexing
    is_merged = torch.zeros(N, dtype=torch.bool, device=metric.device)
    is_merged[merged_b_global] = True
    kept_indices = (~is_merged).nonzero(as_tuple=True)[0]  # (N - r,)

    # For each merged B token, find its partner's position in the kept array
    # Build reverse map: global_idx → position in kept array
    kept_pos = torch.full((N,), -1, dtype=torch.long, device=metric.device)
    kept_pos[kept_indices] = torch.arange(len(kept_indices), device=metric.device)

    # Position of each merged B's partner in the kept array
    merge_dst_pos = kept_pos[matched_a_global]  # (r,) positions in kept array

    # For unmerge: track which kept positions receive merged contributions
    # and the original positions of merged tokens
    merged_b_global = merged_b_global.clone()
    merge_dst_pos = merge_dst_pos.clone()

    def merge_fn(x: torch.Tensor) -> torch.Tensor:
        """Merge tokens: (B, N, ...) → (B, N-r, ...)."""
        # Gather kept tokens
        extra_dims = x.shape[2:]
        out = x[:, kept_indices]  # (B, N-r, ...)

        # Average merged B tokens into their A partners
        src = x[:, merged_b_global]  # (B, r, ...)
        out = out.clone()
        # Accumulate and average
        out.scatter_add_(1, merge_dst_pos.view(1, r, *([1] * len(extra_dims))).expand(x.shape[0], r, *extra_dims), src)
        # Count: each dst position now has (original + 1 merged) = 2 tokens
        counts = torch.ones(out.shape[0], out.shape[1], *([1] * len(extra_dims)),
                           device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, merge_dst_pos.view(1, r, *([1] * len(extra_dims))).expand(x.shape[0], r, *([1] * len(extra_dims))),
                           torch.ones(x.shape[0], r, *([1] * len(extra_dims)), device=x.device, dtype=x.dtype))
        out = out / counts
        return out

    def unmerge_fn(x: torch.Tensor) -> torch.Tensor:
        """Unmerge tokens: (B, N-r, ...) → (B, N, ...)."""
        extra_dims = x.shape[2:]
        out = x.new_zeros(x.shape[0], N, *extra_dims)
        # Place kept tokens back
        out[:, kept_indices] = x
        # Copy from partner for merged tokens
        out[:, merged_b_global] = x[:, merge_dst_pos]
        return out

    return merge_fn, unmerge_fn


def _identity_merge_unmerge(B, N, device):
    """Return identity merge/unmerge when no merging is needed."""
    def merge_fn(x):
        return x
    def unmerge_fn(x):
        return x
    return merge_fn, unmerge_fn


def compute_merge_ratio_schedule(
    total_steps: int,
    mask_step: int,
    base_ratio: float,
    schedule: str = "constant",
) -> list[float]:
    """
    Compute per-step merge ratios.

    Args:
        total_steps: total denoising steps.
        mask_step: step at which merging begins.
        base_ratio: base merge ratio (fraction of background tokens to merge).
        schedule: "constant" or "cosine".

    Returns:
        list of merge ratios, one per step. Steps before mask_step have ratio=0.
    """
    import math
    ratios = [0.0] * total_steps
    active_steps = total_steps - mask_step

    if active_steps <= 0:
        return ratios

    for i in range(mask_step, total_steps):
        if schedule == "constant":
            ratios[i] = base_ratio
        elif schedule == "cosine":
            # Cosine: start at base_ratio, decay to 0 (less merging as detail increases)
            progress = (i - mask_step) / max(1, active_steps - 1)
            ratios[i] = base_ratio * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    return ratios
