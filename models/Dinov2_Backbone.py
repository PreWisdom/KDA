import torch
import torch.nn as nn

from torch.hub import load
from utils.metrics import get_ntrainparams


dino_backbones = {
    'Dinov2_conv_b': {
        'name': 'dinov2_vitb14',
        'in_channels': [2048, 1024, 256, 64],
        'out_channels': [768, 256, 64],
        'embedding_size': 768,
        'patch_size': 14
    },
    'Dinov2_b': {
        'name': 'dinov2_vitb14',
        'in_channels': [768, 512, 256, 64],
        'out_channels': [768, 512, 128],
        'embedding_size': 768,
        'patch_size': 14
    },
    'Dinov2_l': {
        'name': 'dinov2_vitl14',
        'in_channels': [1024, 512, 256, 64],
        'out_channels': [1024, 512, 128],
        'embedding_size': 1024,
        'patch_size': 14
    },
    'Dinov2_g': {
        'name': 'dinov2_vitg14',
        'in_channels': [1536, 768, 256, 64],
        'out_channels': [1536, 768, 256],
        'embedding_size': 1536,
        'patch_size': 14
    }
}

# model path
repo_dir = r'facebook_dinov2'  # local
# repo_dir = r'facebookresearch_dinov2_main'  # server


class Dinov2(nn.Module):
    def __init__(self,
                 backbone='Dinov2_l',  # 参数量 s{22116997} b{86690053} l{304510981} g{1136688645}
                 backbones=dino_backbones):
        super().__init__()
        self.backbones = backbones
        # self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])  # 联网获取模型
        self.backbone = load(repo_dir, self.backbones[backbone]['name'], source='local')  # 使用本地模型
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)
        x = self.backbone.forward_features(x)  # x.cuda()
        x = x['x_norm_patchtokens']
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, self.embedding_size, int(mask_dim[0]), int(mask_dim[1]))
        return x


class Dinov2_Discriminator(nn.Module):
    def __init__(self,
                 backbone='Dinov2_b',
                 backbones=dino_backbones):
        super().__init__()
        self.backbone = Dinov2(backbone=backbone)
        in_channels = backbones[backbone]['in_channels']
        out_channels = backbones[backbone]['out_channels']
        self.model_input = nn.ConvTranspose2d(1024, in_channels[1], kernel_size=4, stride=2, padding=1)
        self.model_input_l = nn.ConvTranspose2d(1536, in_channels[1], kernel_size=4, stride=2, padding=1)
        self.dino_input = nn.ConvTranspose2d(in_channels[0], in_channels[1], kernel_size=4, stride=2, padding=1)
        self.preprocess = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels[1], in_channels[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels[2], in_channels[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels[3]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels[3], 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        )
        self.head = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(),
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(),
            nn.Conv2d(out_channels[2], 1, kernel_size=1, stride=1, padding=0)
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1024:
            x = self.model_input(x)
        elif c == 1536:
            x = self.model_input_l(x)
        else:
            x = self.dino_input(x)
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.head(x)
        x = self.flatten(x)
        x = torch.squeeze(x, dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = Dinov2()
    all_params, trainable_params = get_ntrainparams(model)
    print(all_params)
    print(trainable_params)