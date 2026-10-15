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
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels, kernel_size=3 ,padding=1)
        )

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features = time_dim, out_features = out_channels)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups= num_of_groups, num_channels=out_channels ),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Conv2d(in_channels= in_channels , out_channels=out_channels,kernel_size=1) if in_channels !=out_channels else nn.Identity()

    def forward(self,x,t_emb):
        h = self.block1(x)
        h = h + self.mlp(t_emb).view(x.shape[0],-1,1,1)
        h = self.block2(h)
        return h + self.shortcut(x)