import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        v = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * v - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FullRelPos(nn.Module):

    def __init__(self, h, w, dim, drop_ratio=0.):
        super().__init__()
        self.h, self.w = h, w
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1,
                                                   dim // 2))  # [-(q-1), q-1]
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1,
                                                   dim // 2))  # [-(q-1), q-1]

        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        self.rel_idx_h = coords_h[None, :] - coords_h[:, None]
        self.rel_idx_w = coords_w[None, :] - coords_w[:, None]
        self.rel_idx_h += h - 1
        self.rel_idx_w += w - 1

        nn.init.normal_(self.rel_emb_w, std=dim**-0.5)
        nn.init.normal_(self.rel_emb_h, std=dim**-0.5)
        trunc_normal_(self.rel_emb_w, std=.02)
        trunc_normal_(self.rel_emb_h, std=.02)
        self.drop_ratio = drop_ratio

    def forward(self, q, attn):
        abs_pos_h = self.rel_emb_h[self.rel_idx_h.view(-1)]
        abs_pos_w = self.rel_emb_w[self.rel_idx_w.view(-1)]
        abs_pos_h = rearrange(
            abs_pos_h, '(q k) c -> q k c', q=self.h)  # [qh, kh, c]
        abs_pos_w = rearrange(
            abs_pos_w, '(q k) c -> q k c', q=self.w)  # [qw, kw, c]

        q = rearrange(
            q, 'b (qh qw) g (n c) -> b qh qw g n c', qh=self.h, qw=self.w, n=2)
        logits_h = torch.einsum('b h w g c, h k c -> b h w g k', q[..., 0, :],
                                abs_pos_h)
        logits_w = torch.einsum('b h w g c, w k c -> b h w g k', q[..., 1, :],
                                abs_pos_w)
        logits_h = rearrange(logits_h, 'b h w g k -> b (h w) g k 1')
        logits_w = rearrange(logits_w, 'b h w g k -> b (h w) g 1 k')

        attn = rearrange(
            attn, 'b q g (kh kw) -> b q g kh kw', kh=self.h, kw=self.w)
        attn += logits_h
        attn += logits_w
        return rearrange(attn, 'b q g h w -> b q g (h w)')


def apply_pos(tensor, pos, num_heads):

    def _return_tensor(t):
        return rearrange(t, 'b k p c -> b (k p) c') if len(t.shape) == 4 else t

    if pos is None:
        return _return_tensor(tensor)
    elif len(tensor.shape) != len(pos.shape):
        tensor = rearrange(tensor, 'b n (g c) -> b n g c', g=num_heads)
        tensor = tensor + pos
        tensor = rearrange(tensor, 'b n g c -> b n (g c)')

    return _return_tensor(tensor)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SimpleReasoning(nn.Module):

    def __init__(self, np, dim):
        super(SimpleReasoning, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Conv1d(np, np, kernel_size=1, bias=False)

    def forward(self, x):
        tokens = self.norm(x)
        tokens = self.linear(tokens)
        return x + tokens


class AnyAttention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = (dim / num_heads)**(-0.5)
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, dim)

    def get_qkv(self, q, k, v, qpos, kpos):
        q = apply_pos(q, qpos, self.num_heads)
        k = apply_pos(k, kpos, self.num_heads)
        v = apply_pos(v, None, 0)
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v

    def forward(self,
                q=None,
                k=None,
                v=None,
                qpos=None,
                kpos=None,
                mask=None,
                rel_pos=None):
        """
        Args:
            q: [B, len_q, C]
            k: [B, len_k, C]
            v: [B, len_k, C]
            qpos: [B, len_q, C]
            kpos: [B, len_k, C]
            mask: [B, 1, 1, len_k]

        Returns:
            out: [B, len_q, C]
        """
        q, k, v = self.get_qkv(q, k, v, qpos, kpos)

        # reshape
        q = rearrange(q, 'b n (g c) -> b n g c', g=self.num_heads)
        k = rearrange(k, 'b n (g c) -> b n g c', g=self.num_heads)
        v = rearrange(v, 'b n (g c) -> b n g c', g=self.num_heads)

        # attn matrix calculation
        attn = torch.einsum('b q g c, b k g c -> b q g k', q, k)
        if rel_pos is not None:
            attn = rel_pos(q, attn)
        attn *= self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=float('-inf'))
        attn = F.softmax(attn, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=0)
        out = torch.einsum('b q g k, b k g c -> b q g c', attn, v.float())
        out = rearrange(out, 'b q g c -> b q (g c)')
        out = self.proj(out)
        return out
