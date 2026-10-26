import torch
import torch.nn as nn
from models.unet import UNet
from ema import EMA
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

    def forward(self,x_start:torch.Tensor):
        B = x_start.shape[0]
        device = x_start.device
        noise = torch.randn_like(x_start)
        t = torch.randint(low= 0 , high= self.T , size=(B,), device=device)
        x_t = self.q_sample(x_start=x_start, t = t, noise=noise )
        return self.ddpm_loss(x_t=x_t , t=t , noise= noise)

    def ddpm_loss(self,x_t : torch.Tensor, t : torch.Tensor ,noise : torch.Tensor):
        loss = nn.MSELoss()
        return loss(self.unet(x_t , t,) , noise)

    @torch.no_grad()
    def p_sample(self,x_t : torch.Tensor,t : torch.Tensor):
        noise_hat  = self.unet(x_t , t)

        # clip the x_0 that noise_hat implies to [-1,1] (images live there) and convert back.
        # without this, 1/sqrt(alpha_t) at t=999 is ~100 and any error in noise_hat explodes.

        sqrt_alpha_hat_t = self.sqrt_alphas_hat[t].view(-1,1,1,1)
        sqrt_one_minus_alphas_hat_t = self.sqrt_one_minus_alphas_hat[t].view(-1,1,1,1)
        x_0 = ((x_t - sqrt_one_minus_alphas_hat_t*noise_hat)/sqrt_alpha_hat_t).clamp(-1,1)
        noise_hat = (x_t - sqrt_alpha_hat_t*x_0)/sqrt_one_minus_alphas_hat_t
        mean = (1/self.alphas[t].sqrt().view(-1,1,1,1))*(x_t - (self.betas[t].view(-1,1,1,1)/self.sqrt_one_minus_alphas_hat[t].view(-1,1,1,1))*noise_hat)
        fresh_noise = torch.randn_like(x_t)
        if t[0].item()==0:
            return mean
        return mean + self.betas[t].sqrt().view(-1,1,1,1)* fresh_noise  

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

def cosine_beta_schedule(timesteps, s = 0.008):
    t = torch.linspace(start=0, end=timesteps, steps=timesteps+1)
    f_t =  torch.cos(((t/timesteps+s)/(1+s)) * 0.5 * torch.pi) ** 2
    alphas_hat = f_t / f_t[0]
    betas = 1 - alphas_hat[1:] / alphas_hat[:-1]
    return torch.clamp(betas,min=0.0001 , max=0.999)
