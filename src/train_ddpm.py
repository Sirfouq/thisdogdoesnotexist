import torch
from data import DogSet
from torch.utils.data import DataLoader,Subset
from torch import optim
from models.unet import UNet
from models.ddpm import DDPM
import yaml
import json
import os

with open('config.yaml','r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)


num_epochs = config['train']['epochs']
batch_size = config['train']['batch_size']
seed       = config['train']['seed']
lr         = config['train']['lr']

image_size = config['data']['image_size']


# Overfitting test
train_set_no_aug = DogSet(data_dir='data/afhq', image_size=image_size, hflip=False)
overfit_sample   = Subset(train_set_no_aug, range(100))
overfit_loader   = DataLoader(overfit_sample, batch_size=16, shuffle=True)

train_set = DogSet(data_dir='data/afhq', image_size=image_size, hflip=True)
validation_set = DogSet(data_dir= 'data/afhq', train=False, image_size=image_size)

train_loader = DataLoader(dataset=train_set, batch_size= batch_size, shuffle= True )
val_loader = DataLoader(dataset=validation_set, batch_size= batch_size, shuffle= False )



if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


torch.manual_seed(seed=seed)
unet = UNet(**config['unet'])
model = DDPM(unet=unet, T=1000).to(device=device)

optimizer = optim.Adam(params=model.parameters(),lr=lr, betas=(0.9, 0.999))

history = {'train_loss' : [] , 'val_loss' : []}
val_lowest_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in overfit_loader:
        batch=batch.to(device=device)
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
    avg_loss = total_loss/len(overfit_loader)
    history['train_loss'].append(avg_loss)
    if epoch % 10 ==0 :
        print(f'Epoch {epoch:}| loss{avg_loss}')

os.makedirs('checkpoints', exist_ok=True)
torch.save({
    'config': {'T': 1000, 'unet': config['unet']},
    'state_dict': model.state_dict(),
}, 'checkpoints/ddpm_overfit100.pt')
