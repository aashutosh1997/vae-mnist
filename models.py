import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalLogProb(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mu, sig, z):
        var = torch.pow(sig, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z-mu, 2) / (2*var)
    
class BernoulliLogProb(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, logits, target):
        return -self.loss(logits, target)
    
class Encoder(nn.Module):
        
    class DownConvBlock(nn.Module):
        def __init__(self, n_input, n_output, stride = 1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(n_input, n_output, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU(),
                nn.Conv2d(n_output, n_output, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU()
            )
            
        def forward(self,x):
            return self.net(x)
    
    def __init__(self):
        super().__init__()
        self.conv1 = self.DownConvBlock(1, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = self.DownConvBlock(16, 32)
        self.conv4 = self.DownConvBlock(32, 64)
        self.conv5 = self.DownConvBlock(64, 128)
        self.pool6 = nn.AvgPool2d(kernel_size=3, stride=1)
        self.lin7 = nn.Linear(128, 256)
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.pool2(h)
        h = self.conv3(h)
        h = self.pool2(h)
        h = self.conv4(h)
        h = self.pool2(h)
        h = self.conv5(h)
        h = self.pool6(h)
        h = h.squeeze()
        h = self.lin7(h)
        return h

class VAE(nn.Module):
    
    class Decoder(nn.Module):
        class UpConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, out_padding=0):
                super().__init__()
                self.net = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=out_padding),
                    nn.BatchNorm2d(out_channels)                    
                )
                
            def forward(self,x):
                return self.net(x)
            
        def __init__(self):
            super().__init__()
            self.upconv1 = self.UpConvBlock(128, 64, out_padding=1)
            self.upconv2 = self.UpConvBlock(64, 32, out_padding=1)
            self.upconv3 = self.UpConvBlock(32, 16)
            self.upconv4 = self.UpConvBlock(16, 1)
            
        def forward(self, x):
            h = x
            if x.dim()<2:
                h = h.unsqueeze(0)    
            h = h.unsqueeze(2)
            h = h.unsqueeze(3)
            h = self.upconv1(h)
            h = self.upconv2(h)
            h = self.upconv3(h)
            h = self.upconv4(h)
            return h
        
    def __init__(self):
        super().__init__()
        self.register_buffer("p_z_mu", torch.zeros(128))
        self.register_buffer("p_z_sig", torch.ones(128))
        self.log_p_z = NormalLogProb()
        self.log_p_x = BernoulliLogProb()
        self.decoder = self.Decoder()
    
    def forward(self, z, x):
        log_p_z = self.log_p_z(self.p_z_mu, self.p_z_sig, z).sum(-1, keepdim=True)
        logits = self.decoder(z)
        log_p_x = self.log_p_x(logits, x).sum(-1).sum(-1)
        return log_p_z+log_p_x
    
class VariationalMeanField(nn.Module):
    """
    Network to infer posterior parameters
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.log_q_z = NormalLogProb()
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        mu, sig = torch.chunk(
            self.encoder(x),chunks=2, dim=-1
        )
        sig = self.softplus(sig)
        eps = torch.randn(mu.shape, device=mu.device)
        z = mu + sig * eps
        log_q_z = self.log_q_z(mu, sig, z).sum(-1, keepdim=True)
        return z, log_q_z
    
if __name__ == '__main__':
    variational = VariationalMeanField()
    x = torch.randn([32,1,28,28])
    z, log_q_z = variational(x)
    print(z.shape, log_q_z.shape)
    model = VAE()
    log_p_xz = model(z,x)
    print(log_p_xz.shape)