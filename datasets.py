import torch 
import torchvision.transforms as tfs 
import numpy as np
from glob import glob
from PIL import Image 
import os 
from tqdm import tqdm 
import pandas as pd      

class CelebA_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode=0, classification=False, ffhq=''):
        #filter for those in the training set
        self.datums = pd.read_csv('celeba.csv')
        self.datums = self.datums[self.datums['set'] == mode]  
        self.ffhq = ffhq 
        #instantiate the base directory 
        self.base = '../img_align_celeba' 
        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        
        
        self.data = torch.randn(1, 3, 512, 512)
        
    def __len__(self): 
        return len(self.datums) - 1 

    
    def __getitem__(self, idx):
        path = '{}/{}'.format(self.base, 
                self.datums.iloc[idx]['id']) 
        
        x = Image.open(path) 
        
        # Crop the center of the image
        w, h = x.size 
        crop_size = min(w, h)

        left    = (w - crop_size)/2
        top     = (h - crop_size)/2
        right   = (w + crop_size)/2
        bottom  = (h + crop_size)/2

        # Crop the center of the image
        x = x.crop((left, top, right, bottom))

        # resize the image
        x = x.resize((512, 512))      
                
        if self.transform is not None:
            x = self.transform(x) 
                    
        labels = torch.tensor(self.datums.iloc[idx].drop(['id', 'set']).values.astype(float))
        return {'img': x.to(torch.float32),  
                'index' : idx, 'path': path, 'labels': labels}   
            
    
        return {'img': self.data, 'index' : idx} 
    
# class FFHQ_HDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         self.base = '/home/rmapaij/HSpace-SAEs/datasets/FFHQ/h_seven/*' 
#         self.latents = glob(self.base) 
        
#     def __len__(self): 
#         return len(self.latents) - 1 
    
#     def __getitem__(self, idx):
#         h = torch.load(self.latents[idx], map_location='cpu', mmap = 'r').clone().detach() 
#         h.requires_grad = False
        
#         return {'h': h, 'id' : idx}  