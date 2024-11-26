import torch.nn as nn
import torch
import torch.nn.functional as F 
class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
            # Encoder (Convolutional Layers)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers

        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.decoder0 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(8)  # Use Tanh for RGB output
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  
            nn.Tanh()  # Use Tanh for RGB output
        )
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        
        # Decoder forward pass
        
        ### FILL: encoder-decoder forward pass

        # Encoder forward pass
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # Decoder forward pass
        dec0 = self.decoder0(enc5)
        dec1 = self.decoder1(dec0 + enc4)
        dec2 = self.decoder2(dec1 + enc3)  # Skip connection
        dec3 = self.decoder3(dec2 + enc2)
        dec4 = self.decoder4(dec3 + enc1)  # Skip connection

        output = dec4
        
        return output
    
        self.encoder1 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.decoder0 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Use Tanh for RGB output
        )

    def forward(self, x):
        # Encoder forward pass
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # Decoder forward pass
        dec0 = self.decoder0(enc5)
        dec1 = self.decoder1(enc4 * 0.1 + dec0 * 0.9)  # Skip connection
        dec2 = self.decoder2(enc3 *0.1 + dec1 * 0.9)  # Skip connection
        dec3 = self.decoder3(enc2 *0.1 + dec2 *0.9)  # Skip connection
        dec4 = self.decoder4(enc1 *0.1 + dec3 *0.9)  # Skip connection

        return dec4

    #     self.encoder1 = nn.Sequential(
    #             nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
    #             nn.BatchNorm2d(8),
    #             nn.ReLU(inplace=True)
    #         )
    #     self.encoder2 = nn.Sequential(
    #         nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(16),
    #         nn.ReLU(inplace=True)
    #     )
    #     self.encoder3 = nn.Sequential(
    #         nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(inplace=True)


    #     # Decoder (Deconvolutional Layers)
    #     )
    #     self.decoder1 = nn.Sequential(
    #         nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(16),
    #         nn.ReLU(inplace=True)
    #     )
    #     self.decoder2 = nn.Sequential(
    #         nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(8),
    #         nn.ReLU(inplace=True)
    #     )
    #     self.decoder3 = nn.Sequential(
    #         nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
    #         nn.Tanh()  # Use Tanh for RGB output
    #     )

    # def forward(self, x):
    #     # Encoder forward pass
    #     # enc1 = self.encoder1(x)
    #     # enc2 = self.encoder2(enc1)
    #     # enc3 = self.encoder3(enc2)


    #     # # Decoder forward pass
    #     # dec1 = self.decoder1(enc3)
    #     # dec2 = self.decoder2(enc2 * 0.1 + dec1 * 0.9)  # Skip connection
    #     # dec3 = self.decoder3(enc1 *0.1 + dec2 * 0.9)  # Skip connection

    #     enc1 = self.encoder1(x)
    #     enc2 = self.encoder2(enc1)
    #     enc3 = self.encoder3(enc2)


    #     # Decoder forward pass
    #     dec1 = self.decoder1(enc3)
    #     dec2 = self.decoder2(dec1)  # Skip connection
    #     dec3 = self.decoder3(dec2)  # Skip connection

    #     return dec3

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels, condition_channels, ndf=64, n_layers=1):
        super(PatchGANDiscriminator, self).__init__()
        self.input_channels = input_channels
        self.condition_channels = condition_channels
        self.ndf = ndf
        self.n_layers = n_layers

        # Concatenate input and condition along the channel dimension
        combined_channels = input_channels + condition_channels

        # Initial convolutional layer
        layers = [
            nn.Conv2d(combined_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Subsequent convolutional layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # Final convolutional layer
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, input_image, condition):
        # Concatenate input image and condition along the channel dimension
        combined_input = torch.cat((input_image, condition), dim=1)
        return self.model(combined_input)