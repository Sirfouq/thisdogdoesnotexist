
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,encoder : nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self,input: torch.Tensor):
        output = self.encoder(input)
        mu , log_var = torch.chunk(output,2,dim=1)
        return mu , log_var


class Decoder(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self,z : torch.Tensor):
        output = self.decoder(z)
        return output



class VAE(nn.Module):
    def __init__(self,encoder,decoder,latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterization(mu,log_var):
        std = torch.exp(0.5 *log_var) 
        eps = torch.randn_like(std)
    
        z = mu + std*eps
        return z

    def forward(self,X) :
        mu, log_var = self.encoder(X)
        z = VAE.reparameterization(mu=mu , log_var=log_var)
        x_hat = self.decoder(z)
        return mu, log_var, x_hat

    @torch.no_grad()
    def sample(self,batch_size):
        device = next(self.parameters()).device
        z = torch.randn((batch_size,self.latent_dim),device=device)
        output = self.decoder(z)
        return output 
