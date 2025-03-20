import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from modules.dysample import DySample
from modules.pvt import PyramidVisionTransformer

from utils.metrics import get_ntrainparams
# from modules.SegNeXt.backbone import MSCANet
# from ConvNeXt import ConvNeXt_Backbone

class ToEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        return x


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, n, c = x.shape
        size = int(math.sqrt(n))
        x = x.permute(0, 2, 1).reshape(b, c, size, size)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        # Head size
        self.head_dim = embed_dim // num_heads

        # Linear layers for query, key, value projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, n, c = x1.shape  # Batch size, sequence length, embedding dimension

        # Check input dimensions
        assert x1.shape == x2.shape, "Input tensors x1 and x2 must have the same shape"
        assert c == self.embed_dim, f"Input embedding dimension {c} doesn't match the model's embedding dimension {self.embed_dim}"

        # Project inputs to queries, keys, values
        Q = self.query_proj(x1)  # Shape: (b, n, c)
        K = self.key_proj(x2)
        V = self.value_proj(x2)

        # Reshape to (b, num_heads, n, head_dim)
        Q = Q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Shape: (b, num_heads, n, n)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Apply softmax to get weights

        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, V)  # Shape: (b, num_heads, n, head_dim)

        # Reshape back to (b, n, c)
        attention_output = attention_output.transpose(1, 2).contiguous().view(b, n, c)

        # Final linear projection
        output = self.out_proj(attention_output)  # Shape: (b, n, c)
        return output


class Pvt_Backbone_Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 1, 2, 1],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

    def forward(self, x):
        x = self.encoder(x)
        _, _, _, x_kd = x
        x_kd = F.interpolate(x_kd, size=(16, 16), mode='bilinear', align_corners=True)
        return x_kd, x


class Pvt_Backbone(nn.Module):
    def __init__(self, teacher: str):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 6, 2],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        if teacher == 'Dinov2_b':
            self.neck = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1, stride=1, padding=0)
        elif teacher == 'Dinov2_g':
            self.neck = nn.Conv2d(in_channels=1024, out_channels=1536, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.encoder(x)
        _, _, _, x_kd = x
        x_kd = self.neck(x_kd)
        x_kd = F.interpolate(x_kd, size=(16, 16), mode='bilinear', align_corners=True)
        return x_kd, x


class Pvt_Backbone_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 6, 2],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

    def forward(self, x):
        x = self.encoder(x)
        _, _, _, x_kd = x
        x_kd = F.interpolate(x_kd, size=(16, 16), mode='bilinear', align_corners=True)
        return x_kd, x


class Pvt_Backbone_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[192, 384, 768, 1536],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 9, 3],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

    def forward(self, x):
        x = self.encoder(x)
        _, _, _, x_kd = x
        x_kd = F.interpolate(x_kd, size=(16, 16), mode='bilinear', align_corners=True)
        return x_kd, x


class Pvt_Backbone_XLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[256, 512, 1024, 2048],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 9, 3],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

    def forward(self, x):
        x = self.encoder(x)
        _, _, _, x_kd = x
        x_kd = F.interpolate(x_kd, size=(16, 16), mode='bilinear', align_corners=True)
        return x_kd, x


class Pvt_Backbone_Huge(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[4, 10, 16, 6],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

    def forward(self, x):
        x = self.encoder(x)
        _, _, _, x_kd = x
        x_kd = F.interpolate(x_kd, size=(16, 16), mode='bilinear', align_corners=True)
        return x_kd, x


class GAP_Sigmoid(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).reshape(b, c)
        fc = self.fc(avg).reshape(b, c, 1, 1)
        return fc  # [b, c, 1, 1]


class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                              bias=self.bias)

    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max, avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x
        return output


class Decoder_SEMSFF(nn.Module):  # Decoder 1: Spatial Enhanced Multi-Scale Feature Fusion Module
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 out_channels=128,
                 num_classes=2):
        super().__init__()
        self.cbr1 = nn.Sequential(nn.Conv2d(in_channels[0], out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                  nn.ReLU())
        self.cbr2 = nn.Sequential(nn.Conv2d(in_channels[1], out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                  nn.ReLU())
        self.cbr3 = nn.Sequential(nn.Conv2d(in_channels[2], out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                  nn.ReLU())
        self.cbr4 = nn.Sequential(nn.Conv2d(in_channels[3], out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                  nn.ReLU())

        self.gap1 = GAP_Sigmoid(out_channels)
        self.gap2 = GAP_Sigmoid(out_channels)
        self.gap3 = GAP_Sigmoid(out_channels)
        self.gap4 = GAP_Sigmoid(out_channels)

        self.softmax = nn.Softmax(dim=4)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.norm4 = nn.BatchNorm2d(out_channels)

        self.sa1 = SAM()
        self.sa2 = SAM()
        self.sa3 = SAM()
        self.sa4 = SAM()

        self.cm = nn.Sequential(nn.Conv2d(out_channels, num_classes, kernel_size=1))

    def forward(self, x):
        x1, x2, x3, x4 = x
        x1 = self.cbr1(x1)
        x2 = self.cbr2(x2)
        x3 = self.cbr3(x3)
        x4 = self.cbr4(x4)

        attn1 = self.gap1(x1)
        attn2 = self.gap2(x2)
        attn3 = self.gap3(x3)
        attn4 = self.gap4(x4)

        attn = torch.stack([attn1, attn2, attn3, attn4], dim=4)
        attn = self.softmax(attn)
        attn1, attn2, attn3, attn4 = attn[:, :, :, :, 0], attn[:, :, :, :, 1], attn[:, :, :, :, 2], attn[:, :, :, :, 3]

        g1 = x1 * attn1
        g2 = x2 * attn2
        g3 = x3 * attn3
        g4 = x4 * attn4

        s1 = self.sa1(g1) * g1
        s2 = self.sa2(g2) * g2
        s3 = self.sa3(g3) * g3
        s4 = self.sa4(g4) * g4

        s4 = self.norm1(s4)
        s3 = self.norm2(F.interpolate(s4, scale_factor=2, mode='bilinear', align_corners=True) + s3)
        s2 = self.norm3(F.interpolate(s3, scale_factor=2, mode='bilinear', align_corners=True) + s2)
        s1 = self.norm4(F.interpolate(s2, scale_factor=2, mode='bilinear', align_corners=True) + s1)
        out = F.interpolate(s1, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.cm(out)
        return out


class Decoder_DSMSFF(nn.Module):  # Decoder 2: Dynamic Sample Multi-Scale Feature Fusion Module
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 out_channels=128,
                 num_classes=2):
        super().__init__()
        self.CG2 = nn.Sequential(nn.Conv2d(in_channels[1], out_channels, kernel_size=1), nn.GELU())
        self.CG3 = nn.Sequential(nn.Conv2d(in_channels[2], out_channels, kernel_size=1), nn.GELU())
        self.CG4 = nn.Sequential(nn.Conv2d(in_channels[3], out_channels, kernel_size=1), nn.GELU())
        self.GAP1 = GAP_Sigmoid(out_channels)
        self.GAP2 = GAP_Sigmoid(out_channels)
        self.GAP3 = GAP_Sigmoid(out_channels)
        self.GAP4 = GAP_Sigmoid(out_channels)
        self.GSoftmax = nn.Softmax(dim=4)
        self.UP1 = DySample(out_channels, scale=4)
        self.UP2 = DySample(out_channels)
        self.UP3 = DySample(out_channels)
        self.UP4 = DySample(out_channels)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = x
        x2 = self.CG2(x2)
        x3 = self.CG3(x3)
        x4 = self.CG4(x4)
        attn1 = self.GAP1(x1)
        attn2 = self.GAP2(x2)
        attn3 = self.GAP3(x3)
        attn4 = self.GAP4(x4)
        attn = torch.stack([attn1, attn2, attn3, attn4], dim=4)
        attn = self.GSoftmax(attn)
        attn1, attn2, attn3, attn4 = attn[:, :, :, :, 0], attn[:, :, :, :, 1], attn[:, :, :, :, 2], attn[:, :, :, :, 3]
        w1 = x1 * attn1
        w2 = x2 * attn2
        w3 = x3 * attn3
        w4 = x4 * attn4
        w3 = self.UP4(w4) + w3
        w2 = self.UP3(w3) + w2
        w1 = self.UP2(w2) + w1
        w = self.UP1(w1)
        w = self.head(w)
        return w


class Decoder_DSMSFF_v2(nn.Module):
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 out_channels=128,
                 num_classes=2):
        super().__init__()
        self.CBG1 = nn.Sequential(nn.Conv2d(in_channels[0], out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                  nn.GELU())
        self.CBG2 = nn.Sequential(nn.Conv2d(in_channels[1], out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                  nn.GELU())
        self.CBG3 = nn.Sequential(nn.Conv2d(in_channels[2], out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                  nn.GELU())
        self.CBG4 = nn.Sequential(nn.Conv2d(in_channels[3], out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                  nn.GELU())
        self.GAP1 = GAP_Sigmoid(out_channels)
        self.GAP2 = GAP_Sigmoid(out_channels)
        self.GAP3 = GAP_Sigmoid(out_channels)
        self.GAP4 = GAP_Sigmoid(out_channels)
        self.GSoftmax = nn.Softmax(dim=4)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = x
        x1 = self.CBG1(x1)
        x2 = self.CBG2(x2)
        x3 = self.CBG3(x3)
        x4 = self.CBG4(x4)
        attn1 = self.GAP1(x1)
        attn2 = self.GAP2(x2)
        attn3 = self.GAP3(x3)
        attn4 = self.GAP4(x4)
        attn = torch.stack([attn1, attn2, attn3, attn4], dim=4)
        attn = self.GSoftmax(attn)
        attn1, attn2, attn3, attn4 = attn[:, :, :, :, 0], attn[:, :, :, :, 1], attn[:, :, :, :, 2], attn[:, :, :, :, 3]
        w1 = x1 * attn1
        w2 = x2 * attn2
        w3 = x3 * attn3
        w4 = x4 * attn4
        w3 = F.interpolate(w4, scale_factor=2, mode='bilinear') + w3
        w2 = F.interpolate(w3, scale_factor=2, mode='bilinear') + w2
        w1 = F.interpolate(w2, scale_factor=2, mode='bilinear') + w1
        w = F.interpolate(w1, scale_factor=4, mode='bilinear')
        w = self.head(w)
        return w


class LandFormer_PVT_DSMSFF_Tiny(nn.Module):  # vit depths=[1, 1, 2, 1]
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 1, 2, 1],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = Decoder_DSMSFF()

    def forward(self, x):
        multi = self.encoder(x)
        out = self.decoder(multi)
        return out


class LandFormer_PVT_DSMSFF(nn.Module):  # vit depths=[2, 4, 6, 2]
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 6, 2],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = Decoder_DSMSFF()

    def forward(self, x):
        multi = self.encoder(x)
        out = self.decoder(multi)
        return out


class LandFormer_PVT_DSMSFF_Base(nn.Module):  # vit depths=[2, 4, 6, 2]
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 6, 2],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = Decoder_DSMSFF()

    def forward(self, x):
        multi = self.encoder(x)
        out = self.decoder(multi)
        return out


class LandFormer_PVT_DSMSFF_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[192, 384, 768, 1536],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 4, 1],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = Decoder_DSMSFF(in_channels=[192, 384, 768, 1536], out_channels=192)

    def forward(self, x):
        multi = self.encoder(x)
        out = self.decoder(multi)
        return out


class LandFormer_PVT_DSMSFF_XLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[256, 512, 1024, 2048],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 4, 2],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = Decoder_DSMSFF(in_channels=[256, 512, 1024, 2048], out_channels=256)

    def forward(self, x):
        multi = self.encoder(x)
        out = self.decoder(multi)
        return out


class LandFormer_PVT_DSMSFF_Huge(nn.Module):  # vit depths=[4, 10, 16, 6]
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[4, 10, 16, 6],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = Decoder_DSMSFF()

    def forward(self, x):
        multi = self.encoder(x)
        out = self.decoder(multi)
        return out


class LandFormer_PVT_DSMSFF_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 6, 2],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = Decoder_DSMSFF_v2()

    def forward(self, x):
        multi = self.encoder(x)
        out = self.decoder(multi)
        return out


class LandFormer_PVT_SEMSFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=[128, 256, 512, 1024],
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 6, 2],
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = Decoder_SEMSFF()

    def forward(self, x):
        multi = self.encoder(x)
        out = self.decoder(multi)
        return out


class LandFormrt_Parallel(nn.Module):
    def __init__(self, embed_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.encoder1 = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=embed_dims,
                                                 num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 6, 4],
                                                 sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.encoder2 = MSCANet(in_channnels=3, embed_dims=embed_dims, ffn_ratios=[4, 4, 4, 4],
                                depths=[2, 4, 6, 4], num_stages=4)
        self.reshape = ToEmbedding()
        self.retensor = ToTensor()
        self.cross_attn1 = MultiHeadCrossAttention(embed_dim=embed_dims[0], num_heads=2)
        self.cross_attn2 = MultiHeadCrossAttention(embed_dim=embed_dims[1], num_heads=4)
        self.cross_attn3 = MultiHeadCrossAttention(embed_dim=embed_dims[2], num_heads=8)
        self.cross_attn4 = MultiHeadCrossAttention(embed_dim=embed_dims[3], num_heads=16)
        self.decoder = Decoder_DSMSFF()

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x)

        e1 = []
        e2 = []
        for e in x1:
            e = self.reshape(e)
            e1.append(e)
        for e in x2:
            e = self.reshape(e)
            e2.append(e)

        w0 = self.retensor(self.cross_attn1(e1[0], e2[0]))
        w1 = self.retensor(self.cross_attn2(e1[1], e2[1]))
        w2 = self.retensor(self.cross_attn3(e1[2], e2[2]))
        w3 = self.retensor(self.cross_attn4(e1[3], e2[3]))

        out = self.decoder([w0, w1, w2, w3])

        return out


class LandFormer_Conv_DSMSFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvNeXt_Backbone(in_chans=3, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.4,
                                layer_scale_init_value=1.0, out_indices=[0, 1, 2, 3])
        self.decoder = Decoder_DSMSFF(in_channels=[128, 256, 512, 1024], out_channels=128)

    def forward(self, x):
        multi = self.encoder(x)
        out = self.decoder(multi)  # modify backbone needed
        return out


if __name__ == '__main__':
    model = LandFormer_PVT_DSMSFF_Tiny()
    all_params, trainable_params = get_ntrainparams(model)
    print(all_params)
    print(trainable_params)