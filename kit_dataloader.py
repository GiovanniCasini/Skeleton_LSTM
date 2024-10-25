import logging
import warnings

# Suppress warnings from the transformers library
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message="A parameter name that contains `gamma` will be renamed internally to `weight`.*"
)
warnings.filterwarnings(
    "ignore",
    message="A parameter name that contains `beta` will be renamed internally to `bias`.*"
)

import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
from tqdm import tqdm
import glob
import argparse
import matplotlib.pyplot as plt

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, path, target_length=100):

        self.path = path
        self.target_length = target_length
        self.motions = []
        self.lengths = []
        self.texts = []
        motions = glob.glob(os.path.join(path, "*motion.npy"))
        for m in motions: 
            self.motions.append(m)
            self.lengths.append(m.replace("motion", "length").replace("npy", "txt"))
            self.texts.append(m.replace("motion", "text").replace("npy", "txt"))
            
        self.len = len(self.motions)
        self.maxx, self.minn = 6675.25, -6442.60 # tutti gli elementi

    def __getitem__(self, index):
        motion = np.load(self.motions[index])
        normalized_motion = (motion - self.minn) / (self.maxx - self.minn)
        resampled_motion = self.resample_motion(normalized_motion, self.target_length)
        length = resampled_motion.shape[0]

        with open(self.texts[index], 'r') as f:
            text = f.read()

        return {
            'x': torch.from_numpy(resampled_motion.astype(np.float32)),
            'length': length,
            'text': text
        }

    def __len__(self):
        return self.len
    
    @staticmethod
    def resample_motion(motion, target_length):
        original_length = motion.shape[0]
        if original_length == target_length:
            return motion

        original_indices = np.linspace(0, original_length - 1, num=original_length)
        target_indices = np.linspace(0, original_length - 1, num=target_length)

        resampled_motion = np.zeros((target_length, motion.shape[1], motion.shape[2]))
        for i in range(motion.shape[1]):
            for j in range(motion.shape[2]):
                resampled_motion[:, i, j] = np.interp(target_indices, original_indices, motion[:, i, j])
        return resampled_motion

def collate_fn(batch):
    max_length = max([item['length'] for item in batch])
    for item in batch:
        motion, length = item['x'], item['length']
        padding_length = max_length - length
        padding = torch.zeros((padding_length, motion.shape[1], motion.shape[2]))
        item['x'] = torch.cat((motion, padding), dim=0)
        item['length'] = max_length

    motions = torch.stack([el["x"] for el in batch])
    motions = motions.flatten(2, 3) #(8, seq_len, 21, 3) -> (8, seq_len, 63)
    texts = [el["text"] for el in batch]
    batch_= {}
    batch_["x"] = motions
    batch_["text"] = texts
    batch_["length"] = max_length

    return batch_

def get_lengths(dataset):
    lengths = []
    for motion_file in tqdm(dataset.motions):
        motion = np.load(motion_file, mmap_mode='r')
        length = motion.shape[0]
        lengths.append(length)
    return lengths

def compute_global_max_min(paths):
    max_value = None
    min_value = None
    for path in paths:
        motions = glob.glob(os.path.join(path, "*motion.npy"))
        total_sequences = len(motions)
        filtered_sequences = 0
        for m in tqdm(motions, desc=f"Calcolo max/min in {path}"):
            length_path = m.replace("motion", "length").replace("npy", "txt")
            with open(length_path, 'r') as f:
                length = int(f.read().strip())
            if length <= 120:
                motion = np.load(m, mmap_mode='r')
                current_max = motion.max()
                current_min = motion.min()
                if max_value is None or current_max > max_value:
                    max_value = current_max
                if min_value is None or current_min < min_value:
                    min_value = current_min
            else:
                filtered_sequences += 1
        #print(f"Dataset '{path}': totale sequenze = {total_sequences}, sequenze filtrate = {filtered_sequences}")
    return max_value, min_value

def get_dataloaders(args, bs=1):
    dataset = {}
    train_data = Dataset(args.path_train)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    valid_data = Dataset(args.path_val)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    test_data = Dataset(args.path_test)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data for motion, text, and length")
    parser.add_argument('--path_train', type=str, default=f"{os.getcwd()}/kit_numpy/train", help='Path to the training data')
    parser.add_argument('--path_val', type=str, default=f"{os.getcwd()}/kit_numpy/validation", help='Path to the validation data')
    parser.add_argument('--path_test', type=str, default=f"{os.getcwd()}/kit_numpy/test", help='Path to the test data')
    
    args = parser.parse_args()

    dataloaders = get_dataloaders(args, bs=1)
    for batch in dataloaders["valid"]:
        print(batch["x"].shape)  # Should output: torch.Size([8, 100, 63])

    
    '''
    # Calcolo max e min globali
    dataset_paths = [args.path_train, args.path_val, args.path_test]
    maxx, minn = compute_global_max_min(dataset_paths)
    print(f"Max globale: {maxx}, Min globale: {minn}")

    # Plot delle lunghezze
    train_data = Dataset(args.path_train)
    valid_data = Dataset(args.path_val)
    test_data = Dataset(args.path_test)

    train_lengths = get_lengths(train_data)
    valid_lengths = get_lengths(valid_data)
    test_lengths = get_lengths(test_data)
    all_lengths = train_lengths + valid_lengths + test_lengths
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=500, color='skyblue', edgecolor='black')
    plt.xlabel('Lunghezza della sequenza')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione delle lunghezze delle sequenze nel dataset')
    plt.grid(True)
    plt.savefig('distribuzione_lunghezze_sequenze2.png')
    '''
        