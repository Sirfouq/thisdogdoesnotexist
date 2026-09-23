import torch
from pathlib import Path 
from torch import clamp
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image



class DogSet(Dataset):

    def __init__(self,data_dir,image_size,train = True, hflip = True):
        split = 'train' if train else 'val'
        dog_dir = Path(data_dir)/split/'dog'
        self.paths = sorted(path for path in dog_dir.iterdir() if path.suffix.lower() in {'.jpg' , '.png'})
        transforms : list  = [v2.Resize(size= (image_size,image_size) , antialias= True)] 
        if train and hflip:
            transforms.append(v2.RandomHorizontalFlip())
        transforms+=[
            v2.ToImage(),
            v2.ToDtype(dtype= torch.float32, scale=True),
            v2.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ]
        self.transform = v2.Compose(transforms=transforms)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

# Denormalization formula : x = ((x_norm - a)/ b - a) * (max - min) + min, where a,b -> the current range (a,b) and max, min -> target range (min,max)
def denormalize ( x_norm, x_min = -1.0, x_max= 1.0, original_min = 0.0, original_max=1.0):
    return clamp((((x_norm - x_min) / (x_max-x_min))* (original_max - original_min) + original_min),original_min,original_max)
    
    
        
        
        

        
