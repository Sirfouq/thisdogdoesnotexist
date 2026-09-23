import torch
import torchinfo
from torchvision.utils import save_image
from data import DogSet,denormalize
from torch.utils.data import DataLoader,Subset
from torch import optim
import random
from models import vae
import json
import yaml




with open('config.yaml','r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

image_size = config['data']['image_size']
channels   = config['vae']['channels']
latent_dim = config['vae']['latent_dim']
num_epochs = config['train']['epochs']
batch_size = config['train']['batch_size']
seed       = config['train']['seed']
lr         = config['train']['lr']

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# the training set we will use for the full train_loop
training_set = DogSet(data_dir='data/afhq', image_size=image_size,hflip= True)

# we sample 100 images from the training without hoprizontal flip to confirm that the model is learning before attempting a full training loop.
training_set_no_aug = DogSet(data_dir='data/afhq', image_size=image_size,hflip= False)
overfit_sample = Subset(training_set_no_aug,range(100))


validation_set = DogSet(data_dir= 'data/afhq', train=False, image_size=image_size)


#train loop
torch.manual_seed(seed=seed)
model = vae.VAE(
                encoder= vae.build_encoder_net(image_size=image_size,channels=channels,latent_dim=latent_dim),
                decoder= vae.build_decoder_net(image_size=image_size, channels=channels,latent_dim=latent_dim),
                latent_dim=latent_dim).to(device=device)
optimizer = optim.Adam(params=model.parameters(),lr=lr, betas=(0.9, 0.999))


train_loader = DataLoader(dataset=training_set, batch_size=batch_size,shuffle=True)
val_loader =  DataLoader(dataset=validation_set, batch_size=batch_size,shuffle=False)


history = {
    'train_loss': [], 'train_recon': [], 'train_kl': [],
    'val_loss':   [], 'val_recon':   [], 'val_kl':   [],
}
val_lowest_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_dict = {'loss': 0.0, 'kl': 0.0, 'recon': 0.0}
    n_train_batches = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        mu,log_var,x_hat = model(batch)
        loss,kl,recon = vae.vae_loss(batch_size=batch.shape[0], x_hat= x_hat, x=batch,mu=mu ,log_var=log_var)
        loss.backward()
        optimizer.step()
        train_dict['loss']  += loss.item()
        train_dict['kl']    += kl.item()
        train_dict['recon'] += recon.item()
        n_train_batches += 1
    model.eval()
    val_dict = {'loss':0.0 , 'kl':0.0, 'recon':0.0  }
    n_val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            mu,log_var,x_hat = model(batch)
            loss,kl,recon = vae.vae_loss(batch_size=batch.shape[0], x_hat= x_hat, x=batch,mu=mu ,log_var=log_var)
            val_dict['loss'] += loss.item()
            val_dict['kl'] += kl.item()
            val_dict['recon'] += recon.item()
            n_val_batches+=1
    train_loss = train_dict['loss'] / n_train_batches
    val_loss   = val_dict['loss']   / n_val_batches
    history['train_loss'].append(train_loss)
    history['train_recon'].append(train_dict['recon'] / n_train_batches)
    history['train_kl'].append(train_dict['kl'] / n_train_batches)
    history['val_loss'].append(val_loss)
    history['val_recon'].append(val_dict['recon']/n_val_batches)
    history['val_kl'].append(val_dict['kl']/n_val_batches)
    with open('checkpoints/history.json','w') as f:
        json.dump(history, f)
    if epoch%10 ==0:
             print(f'Epoch {epoch:4d} | '
              f'train {train_loss:8.1f} (recon {train_dict["recon"]/n_train_batches:7.1f} '
              f'kl {train_dict["kl"]/n_train_batches:6.2f}) | '
              f'val {val_loss:8.1f} (recon {val_dict["recon"]/n_val_batches:7.1f} '
              f'kl {val_dict["kl"]/n_val_batches:6.2f})')

    if val_loss < val_lowest_loss:
        torch.save({
             'config' : {'image_size' : image_size,
                        'channels' : channels,
                        'latent_dim' : latent_dim},
             'state_dict' : model.state_dict(),
        },'checkpoints/vae_best.pt'
        )
        val_lowest_loss = val_loss 

         

        






