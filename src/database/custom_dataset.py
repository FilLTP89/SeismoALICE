from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import h5py
from torch import from_numpy

class Hdf5Dataset(Dataset): 
    def __init__(self, hdf5_file,root_dir):
        self.root_dir   = root_dir
        self.hdf5_file  = hdf5_file
        self.dataset    = None
        file            = f"{self.root_dir}/{self.hdf5_file}"
        with h5py.File(file, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            self.dataset = list(f[a_group_key])
        
    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self,index):
        data = self.dataset[index]
        y    = torch.transpose(torch.tensor(data),0,1)
        return y
    

if __name__ == "__main__":

    #for test extraction 
    hdf5 = Hdf5Dataset(hdf5_file="dataset_niigata.hdf5",root_dir="../Niigata/")
    train_set, test_set = torch.utils.data.random_split(hdf5, [810,1])
    train_loader    = DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    test_loader     = DataLoader(dataset=test_set, batch_size=32,shuffle=True)

    for b, batch in enumerate(train_loader):
        y = batch
        print(y.shape)