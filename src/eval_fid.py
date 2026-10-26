from cleanfid import fid
import torch
import os
import yaml
from pathlib import Path
from torchvision.utils import save_image
from data import DogSet, denormalize
from models.ddpm import DDPM, load_ddpm_model
from models.vae import VAE, load_vae_model

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT/'config.yaml' , 'r') as f:
    config = yaml.load(f, Loader= yaml.SafeLoader)
 
def dump_real(out_dir : Path, image_size = 64):
    
    os.makedirs(out_dir, exist_ok=True)
    ds = DogSet(data_dir= ROOT/config['data']['root'], image_size=image_size, hflip=False)
    for i in range(len(ds)):
        save_image(tensor=denormalize(ds[i]),
                   fp= Path(out_dir)/f'{i:05d}.png')
    print(f'Wrote {len(ds)} real images on {out_dir}' , flush=True)

def dump_generated(
                model,
                out_dir: Path,
                num_samples : int = 10000,
                batch_size = 128,
                image_size = 64,
                is_ddpm = True):

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    current = 0 
    while current < num_samples:
        batch = min(batch_size , num_samples - current)
        x = model.sample(batch_size=batch, image_size=(image_size,image_size)) if is_ddpm else model.sample(batch_size=batch)
        for sample in x:
            save_image(denormalize(sample),fp= Path(out_dir)/f'sample{current:05d}.png') # not indexing by sample as sample contains tensors and not indices. Thus we use current.
            current+=1
        print(f'{current}/{num_samples}', flush=True)

def fid_compute(real_dir, generated_dir, device = 'cpu', num_workers = 0):
    score = fid.compute_fid(str(real_dir), str(generated_dir),device=torch.device(device=device),num_workers=num_workers)
    return score 

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    N = 10000

    dump_real(ROOT/'fid_data/real')

    ddpm = load_ddpm_model(ROOT/config['checkpoints']['ddpm'], device=device, use_ema=True)
    dump_generated(ddpm, ROOT/'fid_data/ddpm', num_samples=N, is_ddpm=True)

    vae = load_vae_model(ROOT/config['checkpoints']['vae'], device=device)
    dump_generated(vae, ROOT/'fid_data/vae', num_samples=N, is_ddpm=False)

    print( 'DDPM FID: ',fid_compute(ROOT/'fid_data/real', ROOT/'fid_data/ddpm', device=device))
    print( 'VAE FID: ',fid_compute(ROOT/'fid_data/real', ROOT/'fid_data/vae', device=device))
    


    
