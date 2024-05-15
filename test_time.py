import time 
from torch.utils.data import DataLoader
from datasets import CelebA_Dataset 

start_time = time.time()
data = next(iter(DataLoader(CelebA_Dataset(mode=0)))) 
end_time = time.time() 

print(f"Time taken to load the dataset: {end_time - start_time} seconds") 
