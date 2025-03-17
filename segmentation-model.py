import torch
import torch.nn as nn
import torchvision.models as models

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        # MobileNetV2 encoder
        self.encoder = models.mobilenet_v2(pretrained=True).features

        # Custom decoder with skip connections
        self.decoder = nn.Sequential(
            # Block 1: Input from encoder's last layer (1280, 4, 4) -> (320, 4, 4)
            nn.Sequential(
                nn.ConvTranspose2d(1280, 320, kernel_size=1, stride=1),
                nn.BatchNorm2d(320),
                nn.ReLU(),
                nn.Dropout2d(0.4)
            ),

            # Block 2: (320*2, 4, 4) -> (160, 4, 4)
            nn.Sequential(
                nn.ConvTranspose2d(320*2, 160, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(160),
                nn.ReLU(),
                nn.Dropout2d(0.4)
            ),

            # Block 3: (160*2, 4, 4) -> (96, 8, 8)
            nn.Sequential(
                nn.ConvTranspose2d(160*2, 96, kernel_size=2, stride=2),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.Dropout2d(0.4)
            ),

            # Block 4: (96*2, 8, 8) -> (64, 8, 8)
            nn.Sequential(
                nn.ConvTranspose2d(96*2, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout2d(0.4)
            ),

            # Block 5: (64*2, 8, 8) -> (32, 16, 16)
            nn.Sequential(
                nn.ConvTranspose2d(64*2, 32, kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout2d(0.4)
            ),

            # Block 6: (32*2, 16, 16) -> (24, 32, 32)
            nn.Sequential(
                nn.ConvTranspose2d(32*2, 24, kernel_size=2, stride=2),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Dropout2d(0.4)
            ),

            # Block 7: (24*2, 32, 32) -> (16, 64, 64)
            nn.Sequential(
                nn.ConvTranspose2d(24*2, 16, kernel_size=2, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Dropout2d(0.4)
            ),

            # Block 8: (16*2, 64, 64) -> (32, 64, 64)
            nn.Sequential(
                nn.ConvTranspose2d(16*2, 32, kernel_size=1, stride=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout2d(0.4)
            ),

            # Block 9: (32*2, 64, 64) -> (3, 128, 128)
            nn.Sequential(
                nn.ConvTranspose2d(32*2, 3, kernel_size=2, stride=2),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Dropout2d(0.4)
            ),

            # Final block: (3*2, 128, 128) -> (1, 128, 128)
            nn.Sequential(
                nn.ConvTranspose2d(3*2, 1, kernel_size=1, stride=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
        )

    def forward(self, x):
        # Store original input for skip connection
        r0 = x.clone()

        # Encoder forward pass with skip connections
        x = self.encoder[0](x)  # 32, 128, 128
        r1 = x.clone()

        x = self.encoder[1](x)  # 16, 64, 64
        r2 = x.clone()

        x = self.encoder[2](x)  # 24, 32, 32
        x = self.encoder[3](x)  # 24, 32, 32
        r3 = x.clone()

        x = self.encoder[4](x)  # 32, 16, 16
        r4 = x.clone()
        x = self.encoder[5](x)  # 32, 16, 16
        x = self.encoder[6](x)  # 32, 16, 16

        x = self.encoder[7](x)  # 64, 8, 8
        r5 = x.clone()
        x = self.encoder[8](x)  # 64, 8, 8
        x = self.encoder[9](x)  # 64, 8, 8
        x = self.encoder[10](x) # 64, 8, 8

        x = self.encoder[11](x) # 96, 8, 8
        r6 = x.clone()
        x = self.encoder[12](x) # 96, 8, 8
        x = self.encoder[13](x) # 96, 8, 8

        x = self.encoder[14](x) # 160, 4, 4
        r7 = x.clone()
        x = self.encoder[15](x) # 160, 4, 4
        x = self.encoder[16](x) # 160, 4, 4

        x = self.encoder[17](x) # 320, 4, 4
        r8 = x.clone()
        x = self.encoder[18](x) # 1280, 4, 4

        # Decoder forward pass with skip connections
        x = self.decoder[0](x)  # 320, 4, 4
        x = torch.cat((x, r8), dim=1)
        
        x = self.decoder[1](x)  # 160, 4, 4
        x = torch.cat((x, r7), dim=1)
        
        x = self.decoder[2](x)  # 96, 8, 8
        x = torch.cat((x, r6), dim=1)
        
        x = self.decoder[3](x)  # 64, 8, 8
        x = torch.cat((x, r5), dim=1)
        
        x = self.decoder[4](x)  # 32, 16, 16
        x = torch.cat((x, r4), dim=1)
        
        x = self.decoder[5](x)  # 24, 32, 32
        x = torch.cat((x, r3), dim=1)
        
        x = self.decoder[6](x)  # 16, 64, 64
        x = torch.cat((x, r2), dim=1)
        
        x = self.decoder[7](x)  # 32, 64, 64
        x = torch.cat((x, r1), dim=1)
        
        x = self.decoder[8](x)  # 3, 128, 128
        x = torch.cat((x, r0), dim=1)
        
        x = self.decoder[9](x)  # 1, 128, 128
        
        return x
