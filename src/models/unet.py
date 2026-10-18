from typing import Any

import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self,dim):
        super().__init__()
        self.dim = dim  

    def forward(self , t:torch.Tensor):
        indices = torch.arange(self.dim//2, device=t.device)
        freqs = 1/(10000 ** (2*indices/self.dim) )
        angles  = t.unsqueeze(1) * freqs
        pe = torch.zeros(t.shape[0] , self.dim , device=t.device)
        pe[:, :self.dim//2] = torch.sin(angles)
        pe[:, self.dim//2:] = torch.cos(angles)
        return pe 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels , out_channels ,time_dim, num_of_groups = 32):
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

        self.shortcut = nn.Conv2d(
                            in_channels= in_channels ,
                            out_channels=out_channels,kernel_size=1) if in_channels !=out_channels else nn.Identity()

    def forward(self,x,t_emb):
        h = self.block1(x)
        h = h + self.mlp(t_emb).view(x.shape[0],-1,1,1)
        h = self.block2(h)
        return h + self.shortcut(x)


def downsample(num_channels : int):
    return nn.Conv2d(in_channels=num_channels , out_channels= num_channels , kernel_size=3, stride = 2, padding=1)

def upsample(num_channels : int):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)

    )

class UNet(nn.Module):
    def __init__(self,in_channels =3,
                base_channel = 64,
                time_emb_dim = 64,
                channels_mults = (1,2,2,4),
                num_res_blocks = 2,
                num_groups =32):
        super().__init__()
        self.time_cond_dim = time_emb_dim*4 # conditioned:    ε̂ = f(x_t, t) -> it gives the network an extra input that steers what it produces .
        self.time_mlp = nn.Sequential (
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, self.time_cond_dim),
            nn.SiLU(),
            nn.Linear(self.time_cond_dim, self.time_cond_dim)
        )
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=base_channel,kernel_size=3,stride=1,padding=1)
        channels = [base_channel * m for m in channels_mults]

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, base_channel),
            nn.SiLU(),
            nn.Conv2d(base_channel, in_channels, kernel_size=3, padding=1),
        )

        self.downs = nn.ModuleList()
        current_num_channels = base_channel
        skip_channels = [base_channel]


        for lvl , channel in enumerate(channels):
            for i in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                                in_channels=current_num_channels, 
                                out_channels=channel, 
                                time_dim=self.time_cond_dim,
                                num_of_groups=num_groups))
                current_num_channels = channel
                skip_channels.append(current_num_channels)
            if lvl != len(channels)-1:
                self.downs.append(downsample(num_channels=current_num_channels))
                skip_channels.append(current_num_channels)
        
        self.mid1 = ResidualBlock(in_channels=current_num_channels ,
                                  out_channels= current_num_channels,
                                  time_dim=self.time_cond_dim,
                                  num_of_groups=num_groups)
        self.mid2 = ResidualBlock(in_channels=current_num_channels ,
                                  out_channels= current_num_channels,
                                  time_dim=self.time_cond_dim,
                                  num_of_groups=num_groups)

        self.ups = nn.ModuleList()

        for lvl, channel in enumerate(reversed(channels)):
            for i in range( num_res_blocks+1):
                in_ch = current_num_channels+ skip_channels.pop()
                self.ups.append(ResidualBlock(
                                            in_channels= in_ch,
                                            out_channels=channel,
                                            time_dim=self.time_cond_dim,
                                            num_of_groups=num_groups))
                current_num_channels = channel
            if lvl != len(channels)-1:
                    self.ups.append(upsample(num_channels=current_num_channels))
                




    