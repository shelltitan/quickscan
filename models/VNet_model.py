import torch
import torch.nn as nn
from torchsummary import summary

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class Singleconv(nn.Module):
    """(convolution => [BN] => ReLU) keeps size of input"""
     
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sinngleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=10, padding=9, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.sinngleconv(x)
    
class Encode(nn.Module):
    """Encoding with a convultion that changes the dimensions by a factor of
    0.5 and a Singleconv"""

    def __init__(self, in_channels, out_channels):
         super().__init__()
         self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2,stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Singleconv(out_channels, out_channels))

    def forward(self, x):
        return self.encode(x)

class Decode(nn.Module):
    """Decoding with a convultion that changes the dimensions by a factor of 2 
    and a Singleconv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decode =  nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv = Singleconv(in_channels, out_channels)
    
    def forward(self, x,x_2):
        x = self.decode(x)
        
        x = torch.cat([x_2, x], dim=1)
        return self.conv(x)

class Outconv(nn.Module):
    """convolution => ReLU => output"""

    def __init__(self, in_channels, out_channels):
        super(Outconv, self).__init__()
        self.convout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.convout(x)

class VNet_Torch(nn.Module):
    def __init__(self, n_channels=1):
        super(VNet_Torch, self).__init__()
        
        self.encode1 = Singleconv(n_channels, 16)
        self.encode2 = Encode(16, 32)
        self.encode3 = Encode(32, 64)
        self.encode4 = Encode(64, 128)
        self.encode5 = Encode(128, 256)
        self.decode5 = Decode(256, 128)
        self.decode4 = Decode(128, 64)
        self.decode3 = Decode(64, 32)
        self.decode2 = Decode(32, 16)
        self.decode1 = Outconv(16,1)

    def forward(self, x):
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)
        x = self.decode5(x5, x4)
        x = self.decode4(x, x3)
        x = self.decode3(x, x2)
        x = self.decode2(x, x1)
        logits = self.decode1(x)
        return logits

#model = VNet_Torch().to(device)
#summary = summary(model, (1, 480, 720))