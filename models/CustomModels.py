import torch
import torch.nn as nn
import torchvision.models as models
import timm
import warnings
warnings.filterwarnings("ignore")

class CustomModel(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', decoder='FPN', pretrained=True):
        super(CustomModel, self).__init__()
        
        self.backbone = backbone.lower()

        self._setup_timm_backbone(self.backbone, pretrained)
        
        self.decoder_architecture = decoder
        
        # Decoder setup
        if self.decoder_architecture == 'FPN':
            self._setup_FPN_decoder()
        elif self.decoder_architecture == 'Unet':
            self._setup_Unet_decoder()
        else:
            assert False, f"decoder is {self.decoder_architecture}"

        # Final classification layer
        self.final_conv = nn.Conv2d(self.decoder_final_channels, num_classes, kernel_size=1)
        
    def _setup_timm_backbone(self, model_name, pretrained):
        """Setup TIMM backbone"""
        # Create model with feature extraction
        self.timm_model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)  # Extract features from all stages
        )
        
        # Set encoder channels based on the specific model
        if model_name == 'mobilenetv3_large_100':
            encoder_channels = [16, 24, 40, 112, 960]
        elif model_name == 'mobilenetv4_conv_small.e2400_r224_in1k':
            encoder_channels = [32, 32, 64, 96, 960]
        elif model_name == 'mobilenetv4_hybrid_medium.ix_e550_r384_in1k':
            encoder_channels = [32, 48, 80, 160, 960]
        elif model_name == 'efficientnet_b0':
            encoder_channels = [16, 24, 40, 112, 320]
        elif model_name == 'rexnetr_200.sw_in12k_ft_in1k':
            encoder_channels = [32, 80, 120, 256, 368]
        elif model_name == 'maxvit_base_tf_512.in21k_ft_in1k':
            encoder_channels = [64, 96, 192, 384, 768]
        elif model_name == 'resnet34':
            encoder_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported TIMM model: {model_name}")
        
        self.encoder_channels = encoder_channels
        
    def _setup_Unet_decoder(self):
        """Setup decoder blocks with flexible channel reduction"""
        # Reduce channel dimensions for concatenation
        self.reduce_encoder3 = nn.Conv2d(self.encoder_channels[3], self.encoder_channels[3] // 2, kernel_size=1)
        self.reduce_encoder2 = nn.Conv2d(self.encoder_channels[2], self.encoder_channels[2] // 2, kernel_size=1)
        self.reduce_encoder1 = nn.Conv2d(self.encoder_channels[1], self.encoder_channels[1] // 2, kernel_size=1)
        self.reduce_encoder0 = nn.Conv2d(self.encoder_channels[0], self.encoder_channels[0] // 2, kernel_size=1)
        
        # Augment channel dimensions for concatenation
        self.augment_decoder4 = nn.Conv2d(512, 512 * 2, kernel_size=1)
        self.augment_encoder3 = nn.Conv2d(self.encoder_channels[3], self.encoder_channels[3] * 2, kernel_size=1)

        # Decoder blocks with upsampling and skip connections
        self.decoder4 = self._decoder_block(self.encoder_channels[4], self.encoder_channels[3] // 2)
        self.decoder3 = self._decoder_block(self.encoder_channels[3], self.encoder_channels[2] // 2)
        self.decoder2 = self._decoder_block(self.encoder_channels[2], self.encoder_channels[2])
        
        self.decoder_final_channels = self.encoder_channels[2]

    def _setup_FPN_decoder(self):
        """Setup FPN's lateral connections and top-down pathway."""
        self.lateral4 = nn.Conv2d(self.encoder_channels[4], 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(self.encoder_channels[3], 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(self.encoder_channels[2], 256, kernel_size=1)
        self.lateral1 = nn.Conv2d(self.encoder_channels[1], 256, kernel_size=1)
        
        # Top-down pathway (upsampling and adding lateral connections)
        self.top_down4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.top_down3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.top_down2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.top_down1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.decoder_final_channels = 256


    def _decoder_block(self, in_channels, out_channels):
        """
        Create a decoder block with upsampling, conv, batch norm, and ReLU
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        features = self.timm_model(x)
        _, enc1, enc2, enc3, enc4 = features

        if self.decoder_architecture == 'FPN':
            # Lateral connections
            lateral4 = self.lateral4(enc4)
            lateral3 = self.lateral3(enc3)
            lateral2 = self.lateral2(enc2)
            lateral1 = self.lateral1(enc1)
            
            # Top-down pathway (upsampling and adding)
            up4 = nn.functional.interpolate(lateral4, scale_factor=2, mode='bilinear', align_corners=False)
            top_down4 = self.top_down4(up4 + lateral3)
            
            up3 = nn.functional.interpolate(top_down4, scale_factor=2, mode='bilinear', align_corners=False)
            top_down3 = self.top_down3(up3 + lateral2)
            
            up2 = nn.functional.interpolate(top_down3, scale_factor=2, mode='bilinear', align_corners=False)
            decoder_out = self.top_down2(up2 + lateral1)
        elif self.decoder_architecture == 'Unet':
            # Reduce channel dimensions for concatenation
            reduced_enc3 = self.reduce_encoder3(enc3)
            reduced_enc2 = self.reduce_encoder2(enc2)
            
            # Decoder path with skip connections
            dec4 = self.decoder4(enc4)
            dec3 = self.decoder3(torch.cat([dec4, reduced_enc3], dim=1))
            decoder_out = self.decoder2(torch.cat([dec3, reduced_enc2], dim=1))
        else:
            assert False
        
        # Final classification
        return self.final_conv(decoder_out)

# Example usage
def main():
    device = torch.device("cuda")
    # Create models with different backbones
    backbones = ['resnet34',
                 'mobilenetv3_large_100',
                 'mobilenetv4_conv_small.e2400_r224_in1k',
                 'mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
                 'efficientnet_b0',
                 'rexnetr_200.sw_in12k_ft_in1k',
                 'maxvit_base_tf_512.in21k_ft_in1k']
    for backbone in backbones:
        print("\n_______________________________")
        print(f"Testing {backbone} backbone:")
        for decoder in ['FPN', 'Unet']:
            print(f"\nTesting {decoder} architecture:")
            model = CustomModel(num_classes=22, backbone=backbone, decoder=decoder).to(device)
            
            # Example input (batch_size, channels, height, width)
            x = torch.randn(4, 3, 512, 512).to(device)
            
            # Forward pass
            print(f"Input shape: {x.shape}")
            output = model(x)
            print(f"Output shape: {output.shape}")

if __name__ == '__main__':
    main()