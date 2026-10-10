import torch
import torch.nn as nn

class DDPM(nn.Module):

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

    
    def q_sample(self, x_start : torch.Tensor , t: torch.Tensor , noise : torch.Tensor|None = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_hat_t = self.sqrt_alphas_hat[t].view(-1,1,1,1)
        sqrt_one_minus_alphas_hat = self.sqrt_one_minus_alphas_hat[t].view(-1,1,1,1)
        return sqrt_alpha_hat_t*x_start + sqrt_one_minus_alphas_hat*noise

    




def cosine_beta_schedule(timesteps, s = 0.008):
    t = torch.linspace(start=0, end=timesteps, steps=timesteps+1)
    f_t =  torch.cos(((t/timesteps+s)/(1+s)) * 0.5 * torch.pi) ** 2
    alphas_hat = f_t / f_t[0]
    betas = 1 - alphas_hat[1:] / alphas_hat[:-1]
    return torch.clamp(betas,min=0.0001 , max=0.9999 )
