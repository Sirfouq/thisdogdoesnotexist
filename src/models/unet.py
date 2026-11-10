from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
class SinusoidalTimeEmbedding(nn.Module):

    # Sinusoidal time embedding : t (B,) -> (B,dim) , every randomly selected timestep for every image in the batch gets translated
    # as a dim/2 sized tensor of (sin,cos) pairs. Every pair represents a clock hand moving at its own speed. Fast hands (low i)
    # separate nearby timesteps, slow hands (high i) encode early vs late.
    def __init__(self,dim):
        super().__init__()
        self.dim = dim  

    def forward(self , t:torch.Tensor):
        indices = torch.arange(self.dim//2, device=t.device) # i = 0 ... dim/2 - 1, one per sin,cos pair.
        freqs = 1/(10000 ** (2*indices/self.dim) ) # w_i = 10000^(-2i/dim): for i = 0 -> 1 (fast frequency) , for i = dim/2 - 1 -> -1e-4 (slow frequency).
        angles  = t.unsqueeze(1) * freqs # t is of shape (B,) so the multiplication is not broadcastable . Unsqueeze makes it (B,1) and now the operation is valid.
        pos_emb = torch.zeros(t.shape[0] , self.dim , device=t.device) 
        pos_emb[:, :self.dim//2] = torch.sin(angles) # first half , sines
        pos_emb[:, self.dim//2:] = torch.cos(angles) # second half , cosines
        return pos_emb # returns (B,dim) tensor , with values in [-1,1]


class AttentionBlock(nn.Module):

    # Self attention over spatial positions. Every pixel looks at every other pixel and gathers infromation from the ones 
    # that are most relevant to it. This gives the U-Net global context that local convolutions can't provide.

    def __init__(self, in_channels,num_groups =32):
        super().__init__()

        # Convolutions with kernel = 1 are equal to Linear (C,C). Weights are (c_out,c_in) nothing spatial.
        self.Q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.K = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.V = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.grp_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)

                
        # Maps attention's output back into x's feature space and controls how much of it is added.
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        B,C,H,W = x.shape
        h = self.grp_norm(x)
        q, k, v = self.Q(h) , self.K(h), self.V(h)


        # Treat each pixel as a token: (B, C, H, W) -> (B, C, HW) -> (B, HW, C)
        q = q.flatten(2).transpose(1,2)        
        k = k.flatten(2).transpose(1,2)
        v = v.flatten(2).transpose(1,2)

        # scores  = q @ k.transpose(-2,-1) / (C ** 0.5)   # [B, HW, HW] every position vs every position
        # weights = scores.softmax(dim=-1)                 # rows sum to 1
        # out     = weights @ v                            # [B, HW, C] weighted gather
        
        # out = out.transpose(1,2).unflatten(2,(H,W))
        # return x + self.proj(out)


        out = F.scaled_dot_product_attention(query=q, key=k, value=v)
        out = out.transpose(1,2).unflatten(2,(H,W))
        return x + self.proj(out) # residual -> x + learned global context


class ResidualBlock(nn.Module):

    # Residual block with time conditioning (B,C_in,H,W) -> (B,C_out,H,W). Learns a correction on top of x : x = x + f(x).
    # steps : Look locally (conv) -> add time emb (noise) -> refine (conv) -> add x back -> (optinal) attention for global awareness
    def __init__(self, in_channels , out_channels ,time_dim, num_of_groups = 32 , use_attention = False):
        super().__init__()


        

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_of_groups , num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(
                    in_channels=in_channels ,
                    out_channels=out_channels,
                    kernel_size=3 ,
                    padding=1)
        )

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features = time_dim, out_features = out_channels)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups= num_of_groups, num_channels=out_channels ),
            nn.SiLU(),
            nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1)
        )

        # residual path: matches x's channels to C_out (1x1 conv) if needed, else passes x unchanged
        self.shortcut = nn.Conv2d(
                            in_channels= in_channels ,
                            out_channels=out_channels,kernel_size=1) if in_channels !=out_channels else nn.Identity()

        # Optional attention for global context ( only used in low resolution )
        self.attention_block = AttentionBlock(in_channels=out_channels, num_groups=num_of_groups) if use_attention else nn.Identity()

    def forward(self,x,t_emb):
        h = self.block1(x)
        h = h + self.mlp(t_emb).view(x.shape[0],-1,1,1)
        h = self.block2(h)
        h = h + self.shortcut(x)
        return self.attention_block(h)




# halves the resolution. Channel size remains unchanged (B,C,H,W) -> (B,C,H/2,W/2). For example 64x64 -> 32x32.
def downsample(num_channels : int):
    return nn.Conv2d(in_channels=num_channels , out_channels= num_channels , kernel_size=3, stride = 2, padding=1)

# Doubles resolution: (B, C, H, W) -> (B, C, 2H, 2W). Nearest copies each pixel into a 2x2 block.
# Used instead of ConvTranspose2d, whose uneven kernel overlap causes checkerboard artifacts.
def upsample(num_channels : int):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)

    )

class UNet(nn.Module):
    # Noise predictor eps_hat = f(x_t, t): (B, 3, H, W) -> (B, 3, H, W).
    # Down path shrinks resolution for global context and saves every output as a skip.
    # Up path restores resolution, concatenating the matching skip at each block to recover fine detail.
    def __init__(self,in_channels =3,
                base_channel = 64,
                time_emb_dim = 64,
                channels_mults = (1,2,2,4),
                num_res_blocks = 2,
                num_groups =32,
                attention_res = (16,8),
                image_size = 64):
        super().__init__()
        # t -> sinusoidal embedding -> MLP : learned time embedding shared by every ResBlock. Every ResBlock learns something different about it.
        self.time_cond_dim = time_emb_dim*4 
        self.time_mlp = nn.Sequential (
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, self.time_cond_dim),
            nn.SiLU(),
            nn.Linear(self.time_cond_dim, self.time_cond_dim)
        )
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=base_channel,kernel_size=3,stride=1,padding=1)
        channels = [base_channel * m for m in channels_mults]

        # Output head : back to image channels that reflect predicted noise.
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, base_channel),
            nn.SiLU(),
            nn.Conv2d(base_channel, in_channels, kernel_size=3, padding=1),
        )

        self.downs = nn.ModuleList()
        current_num_channels = base_channel
        skip_channels = [base_channel] # channel count of each saved skip , so up blocks know their input channel size.

        current_res = image_size
        for lvl , channel in enumerate(channels):
            for i in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                                in_channels=current_num_channels, 
                                out_channels=channel, 
                                time_dim=self.time_cond_dim,
                                num_of_groups=num_groups,
                                use_attention= current_res in attention_res )) # attention only used at low resolutions (16,8)
                current_num_channels = channel
                skip_channels.append(current_num_channels)
            if lvl != len(channels)-1:
                self.downs.append(downsample(num_channels=current_num_channels))
                skip_channels.append(current_num_channels)
                current_res //=2
            
        
        self.mid1 = ResidualBlock(in_channels=current_num_channels ,
                                  out_channels= current_num_channels,
                                  time_dim=self.time_cond_dim,
                                  num_of_groups=num_groups,
                                  use_attention= current_res in attention_res)
        
        
        self.mid2 = ResidualBlock(in_channels=current_num_channels ,
                                  out_channels= current_num_channels,
                                  time_dim=self.time_cond_dim,
                                  num_of_groups=num_groups)

        self.ups = nn.ModuleList()
        # Down path saved 12 skips: 1 init_conv + 8 ResBlocks (4 levels × 2) + 3 downsamples (none after the last level).
        # That's 3 per resolution, and each up block consumes one skip of the same size, so num_res_blocks + 1 = 3 blocks per level.
        for lvl, channel in enumerate(reversed(channels)):
            for i in range( num_res_blocks+1):
                in_ch = current_num_channels+ skip_channels.pop()
                self.ups.append(ResidualBlock(
                                            in_channels= in_ch,
                                            out_channels=channel,
                                            time_dim=self.time_cond_dim,
                                            num_of_groups=num_groups,
                                            use_attention= current_res in attention_res))
                current_num_channels = channel
            if lvl != len(channels)-1:
                    self.ups.append(upsample(num_channels=current_num_channels))
                    current_res *= 2

    def forward(self, x_t : torch.Tensor, t: torch.Tensor):
        time_embs = self.time_mlp(t)
        h = self.init_conv(x_t)
        skips = [h]
        for module in self.downs:
            h = module(h,time_embs) if isinstance(module,ResidualBlock) else module(h)
            skips.append(h) # save every down output for skipping

        h = self.mid1(h, time_embs)
        h = self.mid2(h,time_embs)

        for module in self.ups:
            if isinstance(module,ResidualBlock):
                h = torch.cat((h ,skips.pop()),dim=1) # last saved = first used. concat across channels and let next conv decide how to learn the meaning.
                h = module(h,time_embs)
            else :
                h = module(h)

        return self.out(h) # predicted noise
               




    