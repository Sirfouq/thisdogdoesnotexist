import torch
import torch.nn as nn
import torchinfo


class Encoder(nn.Module):
    def __init__(self,encoder : nn.Module):
        super().__init__()
        self.encoder = encoder

    # We split the output of the network to mean and var(log_var). We will use these to sample latent vector z that will be used to reconstruct the image.
    # We are learning q(z|x) approximation. We have used log_var instead of var since the linear head can output negatives and a variance cannot be negative.
    # Also the KL closed form uses log(sigma^2) directly so we never have to call log() on a near zero number.
    def forward(self,input: torch.Tensor):
        output = self.encoder(input)
        mu , log_var = torch.chunk(input=output,chunks=2,dim=1)
        return mu , log_var


class Decoder(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    # Maps z to the per-pixel mean of p(x|z). That mean is the output image. Used in training and in sample().
    def forward(self,z : torch.Tensor):
        output = self.decoder(z)
        return output



class VAE(nn.Module):
    def __init__(self,encoder,decoder,latent_dim):
        super().__init__()
        self.encoder = Encoder(encoder=encoder)
        self.decoder = Decoder(decoder=decoder)
        self.latent_dim = latent_dim

    # Traditional sampling eps ~ N(mu,std^2) introduces a serious issue. Sampling is not differentiable.
    # Therefore the gradients cannot travel throughout the network . To overcome this problem we sample in a different way 
    # which is mathematically equivalent : z = mu + std * eps. Now the equation is differentiable 
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
        # Return all three : mu and log_var feed the KL term,
        # x_hat feeds the reconstruction term. Both halves of the ELBO.
        return mu, log_var, x_hat

    # Sampling only used during generation. Therefore no grad. Skips building the autograd graph ans saves memory.
    @torch.no_grad()
    def sample(self,batch_size):
        device = next(self.parameters()).device
        z = torch.randn((batch_size,self.latent_dim),device=device)
        output = self.decoder(z)
        return output 


def vae_loss(batch_size : int,
            x_hat : torch.Tensor,
            x : torch.Tensor,
            mu : torch.Tensor,
            log_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert batch_size == x.shape[0]
    # reconstruction needs to sum since : p(x|z) = ∏ p(x_i | z) => log p(x|z) = Σ log p(x_i | z) . Product of logs is a sum .
    recon = nn.MSELoss(reduction='sum')(x_hat,x)/batch_size
    # KL : closed form for N(mu, sigma^2) || N(0, I), summed over latent dims, averaged over the batch.
    KL = (-0.5 * torch.sum(1+log_var - mu**2 - torch.exp(log_var),dim=1)).mean()

    # Negative ELBO = recon + KL
    negative_ELBO = recon + KL
    return (negative_ELBO,KL,recon)



def build_encoder_net(image_size, channels, latent_dim):
    spatial = image_size // (2**4)
    encoder_net = nn.Sequential(
        nn.Conv2d(in_channels=channels, out_channels= 64, kernel_size=(4,4), stride=2, padding=1), # stride 0f 2 and padding of 1 ] makes h , w =(64,64) -> h,w = (32,32)
        nn.GroupNorm(8,64),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=2, padding=1 ),
        nn.GroupNorm(8,128),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), stride=2, padding=1 ),
        nn.GroupNorm(8,256),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), stride=2, padding=1 ),
        nn.GroupNorm(8,512),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Flatten(),
        nn.Linear(512*spatial*spatial, 2*latent_dim)
    )
    return encoder_net


def build_decoder_net(latent_dim, channels, image_size):
    spatial = image_size//(2**4)
    decoder_net = nn.Sequential(
        nn.Linear(latent_dim,512*spatial*spatial),
        nn.Unflatten(1,torch.Size([512,spatial,spatial])),
        nn.GroupNorm(8,512),
        nn.LeakyReLU(negative_slope=0.1),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4),stride=2,padding=1),
        nn.GroupNorm(8,256),
        nn.LeakyReLU(negative_slope=0.1),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4),stride=2,padding=1),
        nn.GroupNorm(8,128),
        nn.LeakyReLU(negative_slope=0.1),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4),stride=2,padding=1),
        nn.GroupNorm(8,64),
        nn.LeakyReLU(negative_slope=0.1),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4,4),stride=2, padding=1),
        nn.GroupNorm(8,64),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(in_channels=64, out_channels=channels,kernel_size=(3,3), stride=1, padding=1),
        nn.Tanh()
    )
    return decoder_net



def load_vae_model(path,device = 'cpu'):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    model = VAE(encoder = build_encoder_net(**config),
                decoder= build_decoder_net(**config),
                latent_dim=config['latent_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device=device)