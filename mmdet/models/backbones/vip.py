import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from mmcv import SyncBatchNorm
from mmcv.runner import BaseModule

from mmdet.models.builder import BACKBONES
from mmdet.models.utils.vip_layer import (AnyAttention, DropPath, FullRelPos,
                                          Mlp, SimpleReasoning, trunc_normal_)


class PatchEmbed(nn.Module):

    def __init__(self, stride, has_mask=False, in_ch=0, out_ch=0):
        super().__init__()
        self.to_token = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_ch)
        self.proj = nn.Linear(in_ch, out_ch, bias=False)
        self.has_mask = has_mask

    def process_mask(self, x, mask, H, W):
        if mask is None and self.has_mask:
            mask = x.new_zeros((1, 1, H, W))
        if mask is not None:
            H_mask, W_mask = mask.shape[-2:]
            if H_mask != H or W_mask != W:
                mask = F.interpolate(mask, (H, W), mode='nearest')
        return mask

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, 1, H, W]

        Returns:
            out: [B, out_H * out_W, out_C]
            H, W: output hight & width
            mask: [B, 1, out_H, out_W]
        """
        out = self.to_token(x)
        B, C, H, W = out.shape
        mask = self.process_mask(out, mask, H, W)
        out = rearrange(out, 'b c h w -> b (h w) c').contiguous()
        out = self.proj(out)
        return out, H, W, mask


class Encoder(nn.Module):

    def __init__(self,
                 dim,
                 num_parts=64,
                 num_enc_heads=1,
                 drop_path=0.1,
                 act=nn.GELU,
                 has_ffn=True):
        super(Encoder, self).__init__()
        self.num_heads = num_enc_heads
        self.enc_attn = AnyAttention(dim, num_enc_heads)
        self.drop_path = DropPath(
            drop_prob=drop_path) if drop_path else nn.Identity()
        self.reason = SimpleReasoning(num_parts, dim)
        self.enc_ffn = Mlp(
            dim, hidden_features=dim, act_layer=act) if has_ffn else None

    def forward(self, feats, parts=None, qpos=None, kpos=None, mask=None):
        """
        Args:
            feats: [B, patch_num * patch_size, C]
            parts: [B, N, C]
            qpos: [B, N, 1, C]
            kpos: [B, patch_num * patch_size, C]
            mask: [B, 1, patch_num, patch_size]

        Returns:
            parts: [B, N, C]
        """
        B, P, C = feats.size()
        attn_out = self.enc_attn(
            q=parts,
            k=feats,
            v=feats,
            qpos=qpos,
            kpos=kpos,
            mask=mask.view(B, 1, 1, P))
        parts = parts + self.drop_path(attn_out)
        parts = self.reason(parts)
        if self.enc_ffn is not None:
            parts = parts + self.drop_path(self.enc_ffn(parts))
        return parts


class Decoder(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 patch_size=7,
                 ffn_exp=3,
                 act=nn.GELU,
                 drop_path=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.attn1 = AnyAttention(dim, num_heads)
        self.attn2 = AnyAttention(dim, num_heads)
        self.rel_pos = FullRelPos(patch_size, patch_size, dim // num_heads)
        self.ffn1 = Mlp(
            dim,
            hidden_features=dim * ffn_exp,
            act_layer=act,
            norm_layer=nn.LayerNorm)
        self.ffn2 = Mlp(
            dim,
            hidden_features=dim * ffn_exp,
            act_layer=act,
            norm_layer=nn.LayerNorm)
        self.drop_path = DropPath(drop_path)

    def forward(self,
                x,
                parts=None,
                part_kpos=None,
                mask=None,
                P=0):
        """
        Args:
            x: [B, patch_num * patch_size, C]
            parts: [B, N, C]
            part_kpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size]
            P: patch_num

        Returns:
            feat: [B, patch_num, patch_size, C]
        """
        # [B, patch_num * patch_size, 1, 1]
        dec_mask = None if mask is None else rearrange(
            mask.squeeze(1), 'b h w -> b (h w) 1 1')
        out = self.attn1(
            q=x, k=parts, v=parts, kpos=part_kpos, mask=dec_mask)
        out = x + self.drop_path(out)
        out = out + self.drop_path(self.ffn1(out))

        out = rearrange(out, 'b (p k) c -> (b p) k c', p=P)
        local_mask = None if mask is None else rearrange(
            mask.squeeze(1), 'b p k -> (b p) 1 1 k')
        local_out = self.attn2(
            q=out, k=out, v=out, mask=local_mask, rel_pos=self.rel_pos)
        if local_mask is not None:
            local_mask = rearrange(local_mask, 'b 1 1 k -> b k 1')
            local_out = local_out.masked_fill(local_mask.bool(), value=0)
        out = out + self.drop_path(local_out)
        out = out + self.drop_path(self.ffn2(out))
        return rearrange(out, '(b p) k c -> b p k c', p=P)


class ViPBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_exp=4,
                 drop_path=0.1,
                 patch_size=7,
                 num_heads=1,
                 num_enc_heads=1,
                 num_parts=0):
        super(ViPBlock, self).__init__()
        self.encoder = Encoder(
            dim,
            num_parts=num_parts,
            num_enc_heads=num_enc_heads,
            drop_path=drop_path)
        self.decoder = Decoder(
            dim,
            num_heads=num_heads,
            patch_size=patch_size,
            ffn_exp=ffn_exp,
            drop_path=drop_path)

    def forward(self,
                x,
                parts=None,
                part_qpos=None,
                part_kpos=None,
                mask=None):
        """
        Args:
            x: [B, patch_num, patch_size, C]
            parts: [B, N, C]
            part_qpos: [B, N, 1, C]
            part_kpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size]

        Returns:
            feats: [B, patch_num, patch_size, C]
            parts: [B, N, C]
            part_qpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size]
        """
        P = x.shape[1]
        x = rearrange(x, 'b p k c -> b (p k) c')
        parts = self.encoder(
            x, parts=parts, qpos=part_qpos, mask=mask)
        feats = self.decoder(
            x,
            parts=parts,
            part_kpos=part_kpos,
            mask=mask,
            P=P)
        return feats, parts, part_qpos, mask


class Stage(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 num_blocks,
                 patch_size=7,
                 num_heads=1,
                 num_enc_heads=1,
                 stride=1,
                 num_parts=0,
                 last_np=0,
                 H=0,
                 W=0,
                 drop_path=0.1,
                 has_mask=None,
                 ffn_exp=3):
        super(Stage, self).__init__()
        if isinstance(drop_path, float):
            drop_path = [drop_path for _ in range(num_blocks)]
        self.patch_size = patch_size
        self.rpn_qpos = nn.Parameter(
            torch.Tensor(1, num_parts, 1, out_ch // num_enc_heads))
        self.rpn_kpos = nn.Parameter(
            torch.Tensor(1, num_parts, 1, out_ch // num_heads))

        self.proj = PatchEmbed(
            stride, has_mask=has_mask, in_ch=in_ch, out_ch=out_ch)
        self.proj_token = nn.Sequential(
            nn.Conv1d(last_np, num_parts, 1, bias=False) if
            last_np != num_parts else nn.Identity(), nn.Linear(in_ch, out_ch),
            nn.LayerNorm(out_ch))
        self.proj_norm = nn.LayerNorm(out_ch)

        blocks = [
            ViPBlock(
                out_ch,
                patch_size=patch_size,
                num_heads=num_heads,
                num_enc_heads=num_enc_heads,
                num_parts=num_parts,
                ffn_exp=ffn_exp,
                drop_path=drop_path[i]) for i in range(num_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)

        self._init_weights()

    def _init_weights(self):
        init.kaiming_uniform_(self.rpn_qpos, a=math.sqrt(5))
        trunc_normal_(self.rpn_qpos, std=.02)
        init.kaiming_uniform_(self.rpn_kpos, a=math.sqrt(5))
        trunc_normal_(self.rpn_kpos, std=.02)

    def to_patch(self, x, patch_size, H, W, mask=None):
        x = rearrange(x, 'b (h w) c -> b h w c', h=H)
        pad_l = pad_t = 0
        pad_r = int(math.ceil(W / patch_size)) * patch_size - W
        pad_b = int(math.ceil(H / patch_size)) * patch_size - H
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        if mask is not None:
            mask = F.pad(mask, (pad_l, pad_r, pad_t, pad_b), value=1)
        x = rearrange(
            x,
            'b (sh kh) (sw kw) c -> b (sh sw) (kh kw) c',
            kh=patch_size,
            kw=patch_size)
        if mask is not None:
            mask = rearrange(
                mask,
                'b c (sh kh) (sw kw) -> b c (sh sw) (kh kw)',
                kh=patch_size,
                kw=patch_size)
        return x, mask, H + pad_b, W + pad_r

    def forward(self, x, parts=None, mask=None):
        """
        Args:
            x: [B, C, H, W]
            parts: [B, N, C]
            mask: [B, 1, H, W]

        Returns:
            x: [B, out_C, out_H, out_W]
            parts: [B, out_N, out_C]
            mask: [B, 1, out_H, out_W]
        """
        x, H, W, mask = self.proj(x, mask=mask)
        x = self.proj_norm(x)
        parts = self.proj_token(parts)

        rpn_qpos, rpn_kpos = self.rpn_qpos, self.rpn_kpos
        rpn_qpos = rpn_qpos.expand(x.shape[0], -1, -1, -1)
        rpn_kpos = rpn_kpos.expand(x.shape[0], -1, -1, -1)

        ori_H, ori_W = H, W
        x, mask, H, W = self.to_patch(x, self.patch_size, H, W, mask)
        for blk in self.blocks:
            x, parts, rpn_qpos, mask = blk(
                x,
                parts=parts,
                part_qpos=rpn_qpos,
                part_kpos=rpn_kpos,
                mask=mask)

        x = rearrange(
            x,
            'b (sh sw) (kh kw) c -> b c (sh kh) (sw kw)',
            kh=self.patch_size,
            sh=H // self.patch_size)
        x = x[:, :, :ori_H, :ori_W]

        if mask is not None:
            mask = rearrange(
                mask,
                'b c (sh sw) (kh kw) -> b c (sh kh) (sw kw)',
                kh=self.patch_size,
                sh=H // self.patch_size)
            mask = mask[:, :, :ori_H, :ori_W]
        return x, parts, mask


@BACKBONES.register_module()
class ViP(BaseModule):

    def __init__(self,
                 inplanes=64,
                 num_layers=(3, 4, 6, 3),
                 num_chs=(256, 512, 1024, 2048),
                 num_strides=(1, 2, 2, 2),
                 num_heads=(1, 1, 1, 1),
                 num_parts=(1, 1, 1, 1),
                 patch_sizes=(1, 1, 1, 1),
                 drop_path=0.1,
                 num_enc_heads=(1, 1, 1, 1),
                 act=nn.GELU,
                 ffn_exp=3,
                 no_pos_wd=False,
                 out_indices=[3],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.depth = len(num_layers)
        self.no_pos_wd = no_pos_wd
        self.out_indices = out_indices

        self.conv1 = nn.Conv2d(
            3, inplanes, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm1 = SyncBatchNorm(inplanes)
        self.act = act()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rpn_tokens = nn.Parameter(torch.Tensor(1, num_parts[0], inplanes))

        drop_path_ratios = torch.linspace(0, drop_path, sum(num_layers))
        last_chs = [inplanes, *num_chs[:-1]]
        last_nps = [num_parts[0], *num_parts[:-1]]

        for i, n_l in enumerate(num_layers):
            stage_ratios = [
                drop_path_ratios[sum(num_layers[:i]) + did]
                for did in range(n_l)
            ]
            setattr(
                self, 'layer_{}'.format(i),
                Stage(
                    last_chs[i],
                    num_chs[i],
                    n_l,
                    stride=num_strides[i],
                    num_heads=num_heads[i],
                    num_enc_heads=num_enc_heads[i],
                    patch_size=patch_sizes[i],
                    drop_path=stage_ratios,
                    ffn_exp=ffn_exp,
                    num_parts=num_parts[i],
                    last_np=last_nps[i]))

        for i_layer in out_indices:
            setattr(self, 'trans_norm_{}'.format(i_layer),
                    nn.LayerNorm(num_chs[i_layer]))

    def forward(self, x, img_metas=None):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        mask = x.new_ones((batch_size, 1, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            mask[img_id, :, :img_h, :img_w] = 0

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.pool1(out)

        rpn_tokens = self.rpn_tokens.expand(batch_size, -1, -1)

        out_tokens = []
        for i in range(self.depth):
            layer = getattr(self, 'layer_{}'.format(i))
            out, rpn_tokens, mask = layer(out, rpn_tokens, mask)
            out = out.contiguous()
            if i in self.out_indices:
                norm_layer = getattr(self, 'trans_norm_{}'.format(i))
                B, C, H, W = out.size()
                tmp_out = out.view(B, C, H * W).permute(0, 2, 1).contiguous()
                tmp_out = norm_layer(tmp_out)
                tmp_out = tmp_out.view(B, H, W, C).permute(0, 3, 1,
                                                           2).contiguous()
                out_tokens.append(tmp_out)

        return tuple(out_tokens)
