import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from functools import partial
from modules.UPerNet_head import FPNHEAD
from modules.SegNeXt_head import HamDecoder
from modules.SegFormer_head import SegformerHead, UNetHeadv2
from modules.pvt import PyramidVisionTransformer  # depths = [2, 4, 6, 2]

parent_path = Path(os.path.dirname(os.path.realpath(__file__))).parent
config_file = parent_path / 'modules' / 'SegNeXt' / 'config.yaml'
with open(config_file) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

depths=[2, 4, 6, 2]
embed_dims=[128, 256, 512, 1024]

class PVT_SegFormerDecoder(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=embed_dims,
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths,
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = SegformerHead(inchannels=embed_dims, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PVT_UNetDecoder(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=embed_dims,
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths,
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = UNetHeadv2(inchannels=embed_dims, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PVT_UPerNetDecoder(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=embed_dims,
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths,
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = FPNHEAD()
        self.cls_seg = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x = self.cls_seg(x)
        return x


class PVT_SegNeXtDecoder(nn.Module):
    def __init__(self, outChannels=128, num_classes=2):
        super().__init__()
        self.encoder = PyramidVisionTransformer(img_size=224, patch_size=4, embed_dims=embed_dims,
                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths,
                                                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        self.decoder = HamDecoder(outChannels=outChannels, config=config, enc_embed_dims=embed_dims)
        self.cls_seg = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(outChannels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.cls_seg(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img = torch.randn(1, 3, 224, 224).to(device)
    model = PVT_SegNeXtDecoder().to(device)
    out = model(img)
    print(out.shape)