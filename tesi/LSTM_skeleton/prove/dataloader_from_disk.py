import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
from tqdm import tqdm
import glob
import argparse



class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, path):

        self.path = path
        self.motions = glob.glob(os.path.join(path, "*motion.npy"))
        self.lengths = glob.glob(os.path.join(path, "*length*"))
        self.texts = glob.glob(os.path.join(path, "*text*"))
        self.len = len(self.motions)

    def __getitem__(self, index):
        motion = torch.from_numpy(np.load(self.motions[index]))
        with open(self.lengths[index], 'r') as f:
            length = int(f.read())
        with open(self.texts[index], 'r') as f:
            text = f.read()

        
        return {'motion': motion,
                'length': length,
                'text': text
                }

    def __len__(self):
        return self.len

def get_dataloaders(args):
    dataset = {}
    train_data = Dataset(args.path_train)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    valid_data = Dataset(args.path_val)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_data = Dataset(args.path_test)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data for motion, text, and length")
    parser.add_argument('--path_train', type=str, required=True, help='Path to the training data')
    parser.add_argument('--path_val', type=str, required=True, help='Path to the validation data')
    parser.add_argument('--path_test', type=str, required=True, help='Path to the test data')
    
    args = parser.parse_args()
    
    dataloaders = get_dataloaders(args)
    
    for batch in tqdm(dataloaders['train']):
        print(batch)

