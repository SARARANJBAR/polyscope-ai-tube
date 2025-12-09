import torch
import torch.nn as nn
from transformers import SegformerModel
import torch.nn.functional as F
import torch.nn as nn
# =========================
# Models
# TODO: import hardcoded vars from model config
# =========================
# ---- A) Mask2Former-lite + MiT-B5 encoder (HF SegformerModel) ----
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, q, kv):
        attn, _ = self.cross_attn(q, kv, kv)
        q = self.norm1(q + attn)
        ff = self.ff(q)
        q = self.norm2(q + ff)
        return q


class FeaturePyramidFromHFSegformer(nn.Module):
    def __init__(self, ckpt_name="nvidia/mit-b5"):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(ckpt_name)
        self.backbone.config.output_hidden_states = True
    def forward(self, x):
        out = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
        feats = list(out.hidden_states)[-4:]  # [B,C,h,w] for strides 4/8/16/32
        return feats


class Mask2FormerLite(nn.Module):
    def __init__(self, in_channels=(64,128,320,512), embed_dim=256,
                 num_queries=8, num_layers=4, nheads=8):
        super().__init__()
        self.embed_dim = embed_dim
        # unify channels
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim*4, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model=embed_dim, nhead=nheads,
                                    dim_feedforward=4*embed_dim, dropout=0.0)
            for _ in range(num_layers)
        ])
        self.cls_head = nn.Linear(embed_dim, 1)  # scores per query
        self.mask_embed = nn.Linear(embed_dim, embed_dim)
        self.mask_proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, feats, input_size_hw):
        x1,x2,x3,x4 = feats
        B = x1.size(0)
        # unify to 1/4 scale
        u = [self.proj[i](f) for i,f in enumerate([x1,x2,x3,x4])]
        tgt_size = x1.shape[-2:]
        u[1] = F.interpolate(u[1], size=tgt_size, mode='bilinear', align_corners=False)
        u[2] = F.interpolate(u[2], size=tgt_size, mode='bilinear', align_corners=False)
        u[3] = F.interpolate(u[3], size=tgt_size, mode='bilinear', align_corners=False)
        F_ = self.fuse(torch.cat(u, dim=1))             # [B,D,H/4,W/4]
        H4,W4 = F_.shape[-2:]
        kv = F_.permute(0,2,3,1).reshape(B, H4*W4, self.embed_dim)  # [B,HW,D]
        q = self.query_embed.unsqueeze(0).repeat(B,1,1)             # [B,Q,D]
        for layer in self.decoder:
            q = layer(q, kv)
        cls_logits = self.cls_head(q).squeeze(-1)        # [B,Q]
        weights = torch.softmax(cls_logits, dim=1)       # [B,Q]
        mask_e = self.mask_embed(q)                      # [B,Q,D]
        mask_feat = self.mask_proj(F_)                   # [B,D,H/4,W/4]
        masks_q = torch.einsum('bqk,bkxy->bqxy', mask_e, mask_feat)   # [B,Q,H/4,W/4]
        bed_1_4 = torch.einsum('bq,bqxy->bxy', weights, masks_q).unsqueeze(1)  # [B,1,H/4,W/4]
        bed_full = F.interpolate(bed_1_4, size=input_size_hw, mode='bilinear', align_corners=False)
        return bed_full

class M2F_MiTB5_TumorBed(nn.Module):
    def __init__(self, ckpt_name="nvidia/mit-b5", embed_dim=256, num_queries=8, num_layers=4, nheads=8):
        super().__init__()
        self.backbone = FeaturePyramidFromHFSegformer(ckpt_name=ckpt_name)
        self.decoder = Mask2FormerLite(
            in_channels=(64,128,320,512),
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            nheads=nheads
        )
    def forward(self, x):
        feats = self.backbone(x)
        return self.decoder(feats, x.shape[-2:])  # [B,1,H,W]


class SegFormerBinary(nn.Module):
    def __init__(self, ckpt="nvidia/mit-b5", embed_dim=256):
        super().__init__()

        self.backbone = SegformerModel.from_pretrained(ckpt)
        self.backbone.config.output_hidden_states = True
        self.embed_dim = embed_dim
        self.proj = nn.ModuleList([
            nn.Conv2d(64,  embed_dim, 1),
            nn.Conv2d(128, embed_dim, 1),
            nn.Conv2d(320, embed_dim, 1),
            nn.Conv2d(512, embed_dim, 1),
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim*4, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x):
        out = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
        x1,x2,x3,x4 = list(out.hidden_states)[-4:]
        tgt = x1.shape[-2:]
        u1 = self.proj[0](x1)
        u2 = F.interpolate(self.proj[1](x2), size=tgt, mode="bilinear", align_corners=False)
        u3 = F.interpolate(self.proj[2](x3), size=tgt, mode="bilinear", align_corners=False)
        u4 = F.interpolate(self.proj[3](x4), size=tgt, mode="bilinear", align_corners=False)
        F_ = self.fuse(torch.cat([u1,u2,u3,u4], dim=1))  # [B,256,H/4,W/4]
        logits = self.head(F_)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits  # [B,1,H,W]
