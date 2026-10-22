import torch
from data import DogSet
from torch.utils.data import DataLoader,Subset
from torch import optim
from models.unet import UNet
from models.ddpm import DDPM
import yaml
import json
import os
from ema import EMA
from pathlib import Path
from torch.amp import autocast,GradScaler

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT/'config.yaml','r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

data_root = ROOT/config['data']['root']

num_epochs = config['train']['epochs']
batch_size = config['train']['batch_size']
seed       = config['train']['seed']
lr         = config['train']['lr']

image_size = config['data']['image_size']
T = config['ddpm']['timesteps']


# Overfitting test
train_set_no_aug = DogSet(data_dir=data_root, image_size=image_size, hflip=False)
overfit_sample   = Subset(train_set_no_aug, range(100))
overfit_loader   = DataLoader(overfit_sample, batch_size=16, shuffle=True)

train_set = DogSet(data_dir=data_root, image_size=image_size, hflip=True)
validation_set = DogSet(data_dir= data_root, train=False, image_size=image_size)

train_loader = DataLoader(dataset=train_set, batch_size= batch_size, shuffle= True )
val_loader = DataLoader(dataset=validation_set, batch_size= batch_size, shuffle= False )



if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

use_amp = (device=='cuda')
scaler = GradScaler(enabled=use_amp)

torch.manual_seed(seed=seed)
unet = UNet(**config['unet'])
model = DDPM(unet=unet, T=T).to(device=device)

ema = EMA(model=model, decay=0.9999)

optimizer = optim.Adam(params=model.parameters(),lr=lr, betas=(0.9, 0.999))

history = {'train_loss' : [] , 'val_loss' : []}
val_lowest_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in overfit_loader:
        batch=batch.to(device=device)
        optimizer.zero_grad()

        # convs and matmuls run in fp16 (~5x faster on a T4); reductions stay fp32. Enabled=False on mps/cpu, so this is a no-op locally.
        with autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            loss = model(batch)
        # fp16 can't represent gradients below ~6e-5, so scale the loss up before .Backward and unscale before stepping. scaler.step replaces optimizer.step.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model=model)
        total_loss +=loss.item()
    avg_loss = total_loss/len(overfit_loader)
    history['train_loss'].append(avg_loss)
    if epoch % 10 ==0 :
        print(f'Epoch {epoch:}| loss{avg_loss}')

os.makedirs(ROOT/'checkpoints', exist_ok=True)
torch.save({
    'config': {'T': T, 'unet': config['unet']},
    'state_dict': model.state_dict(),
    'ema_state_dict' : ema.shadow
}, ROOT/'checkpoints/ddpm_overfit100.pt')
