"""Pairwise logistic loss and NDCG-weighted LambdaRank (paper Eq. 6 family).

Natural logarithms change only a constant factor versus the paper's log2.
This implements the LambdaRank configuration, not every LambdaLoss variant.
"""
import torch
from torch.nn import functional as F


def ranking_loss(scores, relevance, mode="lambda", k=5):
    """One query at a time; unjudged labels (-1) never become negatives.

    Mean over informative pairs. The trainer then averages over queries.
    Current ranks / swap weights are recomputed each step and detached.
    """
    if scores.ndim != 1 or relevance.shape != scores.shape:
        raise ValueError("Expected equal one-dimensional score and label tensors")
    if mode not in ("pairwise", "lambda") or k < 1:
        raise ValueError("mode must be pairwise/lambda, with k >= 1")
    judged = relevance >= 0
    s, y = scores[judged], relevance[judged]
    if y.numel() < 2:
        return scores.sum() * 0
    preferred = y[:, None] > y[None, :]
    if not preferred.any():
        return scores.sum() * 0
    losses = F.softplus(s[None, :] - s[:, None])
    if mode == "pairwise":
        return losses[preferred].mean()
    with torch.no_grad():
        positions = torch.arange(1, len(y) + 1, device=s.device, dtype=s.dtype)
        discount = torch.where(positions <= k, 1 / torch.log2(positions + 1), 0.0)
        gains = torch.pow(2.0, y) - 1
        ideal = torch.sort(gains, descending=True).values
        idcg = (ideal * discount).sum()
        if idcg <= 0:
            return scores.sum() * 0
        order = torch.argsort(s, descending=True, stable=True)
        rank_discount = torch.empty_like(discount)
        rank_discount[order] = discount
        swap_weight = (gains[:, None] - gains[None, :]).abs() * (
            rank_discount[:, None] - rank_discount[None, :]).abs() / idcg
        active = preferred & (swap_weight > 0)
    if not active.any():
        return scores.sum() * 0
    return (swap_weight[active] * losses[active]).mean()
