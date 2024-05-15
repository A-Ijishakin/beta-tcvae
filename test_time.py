import time 
from torch.utils.data import DataLoader
from datasets import CelebA_Dataset 
import multiprocessing 
BATCH_SIZE = 8 

def test_time(): 
    start_time = time.time()

    train_loader = DataLoader(CelebA_Dataset(mode=0), batch_size=BATCH_SIZE,
                            num_workers=8,         
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4, 
                            shuffle=True) 

    # data = next(iter(DataLoader(CelebA_Dataset(mode=0), batch_size=8))) 
    data = next(iter(train_loader)) 
    end_time = time.time() 
    data = data['img'].to('cuda:0')

    print(f"Time taken to load the dataset: {end_time - start_time} seconds") 

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn') 
    test_time() 