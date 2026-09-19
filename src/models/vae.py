import torch
import torch.nn as nn

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


def vae_loss(batch_size,x_hat, x, mu, log_var):
    assert batch_size == x.shape[0]
    #reconstruction needs to sum since : p(x|z) = ∏ p(x_i | z) => log p(x|z) = Σ log p(x_i | z) . Product of logs is a sum .
    recon = nn.MSELoss(reduction='sum')(x_hat,x)/batch_size
    KL = (-0.5 * torch.sum(1+log_var - mu**2 - torch.exp(log_var),dim=1)).mean()

    negative_ELBO = recon + KL
    return negative_ELBO,KL,recon



