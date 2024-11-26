import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s
from torchvision.models._utils import IntermediateLayerGetter

class FPNEfficientNetV2_S(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(FPNEfficientNetV2_S, self).__init__()
        
        # Load pretrained EfficientNetV2-B0 as backbone
        efficientnet = efficientnet_v2_s(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Define backbone layers
        self.backbone = IntermediateLayerGetter(efficientnet.features, {
            '1': 'stage1',  # 24 channels
            '2': 'stage2',  # 48 channels
            '3': 'stage3',  # 64 channels
            '5': 'stage4',  # 160 channels
            '6': 'stage5'   # 256 channels
        })
        
        # Lateral layers
        self.lateral5 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral4 = nn.Conv2d(160, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(64, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(48, 256, kernel_size=1)
        self.lateral1 = nn.Conv2d(24, 256, kernel_size=1)
        
        # FPN layers
        self.fpn5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Output layer
        self.output = nn.Conv2d(256, num_classes, kernel_size=1)
        
        self.freeze_initial_layers()

    def freeze_initial_layers(self):
        # Freeze the initial layers (first two stages)
        for name, param in self.backbone.named_parameters():
            if 'stage1' in name or 'stage2' in name:
                param.requires_grad = False

    def forward(self, x):
        # Bottom-up pathway
        features = self.backbone(x)
        c2, c3, c4, c5 = features['stage2'], features['stage3'], features['stage4'], features['stage5']

        # Top-down pathway and lateral connections
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')

        # FPN layers
        p5 = self.fpn5(p5)
        p4 = self.fpn4(p4)
        p3 = self.fpn3(p3)
        p2 = self.fpn2(p2)

        # Upsample to match input size
        p5 = F.interpolate(p5, scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)

        # Combine FPN levels
        out = p2 + p3 + p4 + p5

        # Final prediction
        out = self.output(out)

        return out

# Example usage
if __name__ == '__main__':
    # Example instantiation
    model = FPNEfficientNetV2_S(num_classes=20)
    
    # Test input
    x = torch.randn(2, 3, 512, 512)  # Batch size 2, RGB, 512x512
    output = model(x)
    print(f"Output shape: {output.shape}")  # Expected: [2, num_classes, 128, 128]