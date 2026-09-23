import torch
from models import vae
from torchvision.utils import save_image
from data import denormalize
import os

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'   

path = 'samples/'
os.makedirs(path, exist_ok=True)

model = vae.load_vae_model(path='checkpoints/vae_best_v2.pt' , device=device)
model.eval()
samples = model.sample(batch_size=16)
save_image(denormalize(samples.cpu()), f'{path}samples.png', nrow= 4)

