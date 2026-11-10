import torch
import torch.nn as nn
from models.unet import UNet
from ema import EMA
class DDPM(nn.Module):

    # Essential attributes for noise scheduling. These annotations exist so type_checkers know that register_buffer attributes are tensors.
    betas:torch.Tensor
    alphas:torch.Tensor
    sqrt_alphas_hat:torch.Tensor
    sqrt_one_minus_alphas_hat:torch.Tensor

    def __init__(self , unet : nn.Module , T : int):
        super().__init__()
        self.unet = unet
        self.T = T
        betas = cosine_beta_schedule(timesteps=T)
        alphas = 1-betas
        alphas_hat = torch.cumprod(alphas, dim=0)
        sqrt_alphas_hat = alphas_hat.sqrt()
        sqrt_one_minus_alphas_hat = (1-alphas_hat).sqrt()
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_hat', alphas_hat)
        self.register_buffer('sqrt_alphas_hat', sqrt_alphas_hat)
        self.register_buffer('sqrt_one_minus_alphas_hat', sqrt_one_minus_alphas_hat)

    # Here we jump straight to step t of a noised image instead of iterating t steps.
    # t is per image. For a batch of size B , the t tensor has a shape of (B,). Each image gets its own random t and indexing the schedule with it gives coefficients of shape (B,)
    # which view(-1,1,1,1) reshapes to (B,1,1,1) so they broadcast over x_start's (B,C,H,W). 

    def q_sample(self, x_start : torch.Tensor , t: torch.Tensor , noise : torch.Tensor|None = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_hat_t = self.sqrt_alphas_hat[t].view(-1,1,1,1)
        sqrt_one_minus_alphas_hat = self.sqrt_one_minus_alphas_hat[t].view(-1,1,1,1)
        return sqrt_alpha_hat_t*x_start + sqrt_one_minus_alphas_hat*noise

    def forward(self,x_start:torch.Tensor):
        batch_size = x_start.shape[0]
        device = x_start.device
        noise = torch.randn_like(x_start)
        t = torch.randint(low= 0 , high= self.T , size=(batch_size,), device=device)
        x_t = self.q_sample(x_start=x_start, t = t, noise=noise )
        return self.ddpm_loss(x_t=x_t , t=t , noise= noise)


    # Simply MSELoss between the UNet's predicted noise,  against the noise we have actually  used .
    def ddpm_loss(self,x_t : torch.Tensor, t : torch.Tensor ,noise : torch.Tensor):
        loss = nn.MSELoss()
        return loss(self.unet(x_t , t,) , noise)

    # One reverse step : takes x_t and returns x_(t-1). Only used during sampling . Training does not touch it . Therefore we use no_grad
    @torch.no_grad()
    def p_sample(self,x_t : torch.Tensor,t : torch.Tensor):
        noise_hat  = self.unet(x_t , t)

        # clip the x_0 that noise_hat implies to [-1,1] (images live there) and convert back.
        # without this, dividing by sqrt(alpha_hat_t) at t=999 amplifies any error in noise_hat by ~2e4.
        # the mean formula is expressed in terms of noise, so we re-derive noise_hat from the clamped x_0.
        sqrt_alpha_hat_t = self.sqrt_alphas_hat[t].view(-1,1,1,1)
        sqrt_one_minus_alphas_hat_t = self.sqrt_one_minus_alphas_hat[t].view(-1,1,1,1)
        x_0 = ((x_t - sqrt_one_minus_alphas_hat_t*noise_hat)/sqrt_alpha_hat_t).clamp(-1,1)
        noise_hat = (x_t - sqrt_alpha_hat_t*x_0)/sqrt_one_minus_alphas_hat_t
        mean = (1/self.alphas[t].sqrt().view(-1,1,1,1))*(x_t - (self.betas[t].view(-1,1,1,1)/self.sqrt_one_minus_alphas_hat[t].view(-1,1,1,1))*noise_hat)
        fresh_noise = torch.randn_like(x_t)

        # at t=0 we return the mean with no noise added : x_0 is the final image, adding noise would just be visible static.
        if t[0].item()==0:
            return mean
        return mean + self.betas[t].sqrt().view(-1,1,1,1)* fresh_noise  

    # Running p-sample T times in reverse. The denoising process. Generation only like p_sample.
    @torch.no_grad()
    def sample(self,batch_size :int, image_size : tuple, channel_size = 3):
        device = next(self.parameters()).device
        x_t = torch.randn(batch_size, channel_size ,*image_size , device=device)
        for t in reversed(range(self.T)):
            t_batch = torch.ones(batch_size , device=device, dtype=torch.int32) * t
            x_prev = self.p_sample(x_t=x_t , t=t_batch)
            x_t = x_prev
        return x_t.clamp(-1,1)


def load_ddpm_model(path, device='cpu', use_ema =True):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    unet = UNet(**config['unet'])
    model = DDPM(unet=unet, T=config['T'])
    model.load_state_dict(checkpoint['state_dict'])
    if use_ema and 'ema_state_dict' in checkpoint:
        ema = EMA(model=model, decay=0.0)
        ema.shadow = checkpoint['ema_state_dict']
        ema.copy_to(model=model)

    return model.to(device=device)


# A noise schedule decides how fast an image is destroyed on the way to pure noise. This is the reason we have selected cosine scheduling instead of linear scheduling .
# beta_t is the noise added at step t; the cumulative product ᾱ_t is the fraction of the original image still left at step t, sliding from 1 (clean) to 0 (pure N(0,I)).
# The pacing is what matters : destroy too fast and most timesteps are pure noise with nothing to learn, too slow and x_T never actually reaches the N(0,I) we start sampling from.

def cosine_beta_schedule(timesteps, s = 0.008):
    t = torch.linspace(start=0, end=timesteps, steps=timesteps+1)
    f_t =  torch.cos(((t/timesteps+s)/(1+s)) * 0.5 * torch.pi) ** 2
    alphas_hat = f_t / f_t[0]
    betas = 1 - alphas_hat[1:] / alphas_hat[:-1]
    return torch.clamp(betas,min=0.0001 , max=0.999)

# Linear schedules drive ᾱ to ~0 by t≈670, so the last third of timesteps are already pure noise and
# teach the model almost nothing. Cosine keeps ᾱ falling gradually, so more timesteps stay informative.
# https://arxiv.org/abs/2102.09672