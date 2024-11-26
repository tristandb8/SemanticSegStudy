import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.models._utils import IntermediateLayerGetter

class FPNMobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(FPNMobileNetV3Large, self).__init__()
        
        # Load pretrained MobileNetV3Large as backbone
        mobilenet = mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Define backbone layers
        self.backbone = IntermediateLayerGetter(mobilenet.features, {
            '3': 'stage1',   # 24 channels
            '6': 'stage2',   # 40 channels
            '12': 'stage3',  # 112 channels
            '16': 'stage4'   # 960 channels
        })
        
        # Lateral layers
        self.lateral4 = nn.Conv2d(960, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(112, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(40, 256, kernel_size=1)
        self.lateral1 = nn.Conv2d(24, 256, kernel_size=1)
        
        # FPN layers
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
        c1, c2, c3, c4 = features['stage1'], features['stage2'], features['stage3'], features['stage4']
        
        # Top-down pathway and lateral connections
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='nearest')

        # FPN layers
        p4 = self.fpn4(p4)
        p3 = self.fpn3(p3)
        p2 = self.fpn2(p2)
        p1 = self.fpn1(p1)
        
        p4 = F.interpolate(p4, scale_factor=8, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Combine FPN levels
        out = p1 + p2 + p3 + p4
        
        # Final prediction
        out = self.output(out)
        
        return out
    
# Example usage
if __name__ == '__main__':
    # Example instantiation
    model = FPNMobileNetV3Large(num_classes=20)
    
    # Test input
    x = torch.randn(2, 3, 512, 512)  # Batch size 2, RGB, 512x512
    output = model(x)
    print(f"Output shape: {output.shape}")  # Expected: [2, num_classes, 128, 128]