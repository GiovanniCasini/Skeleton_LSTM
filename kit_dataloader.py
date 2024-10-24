import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
from tqdm import tqdm
import glob
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, path):

        self.path = path
        self.target_length = 200
        samples = {}
        self.motions = []
        self.lengths = []
        self.texts = []
        motions = glob.glob(os.path.join(path, "*motion.npy"))
        for m in motions: 
            length_path = m.replace("motion", "length").replace("npy", "txt")
            with open(length_path, 'r') as f:
                length = int(f.read().strip())
            # Filtra le sequenze con lunghezza <= 120
            if length <= 2000:
                self.motions.append(m)
                self.lengths.append(length_path)
                self.texts.append(m.replace("motion", "text").replace("npy", "txt"))
            
        self.len = len(self.motions)
        #self.maxx, self.minn = 6675.25, -6442.60 # tutti gli elementi
        #self.maxx, self.minn = 5912.18, -4839.69 # elementi lunghi < 120
        self.std, self.mean = 790.71, 357.44

    def __getitem__(self, index):
        motion = torch.from_numpy(np.load(self.motions[index])).reshape(-1, 63)
        normalized_motion = (motion - self.mean) /self.std #(motion - self.minn) / (self.maxx - self.minn)
        length = motion.shape[0]
        with open(self.texts[index], 'r') as f:
            text = f.read()

        return {'x': normalized_motion,
                'length': length,
                'text': text
                }
    def __getitem__old(self, index):
        motion = torch.from_numpy(np.load(self.motions[index])).reshape(-1, 63)
        length = motion.shape[0]

        # Normalize the motion
        normalized_motion = (motion - self.mean) / self.std

        if length < self.target_length:
            # Upsample using interpolation (nearest, linear, bilinear, etc.)
            normalized_motion = normalized_motion.transpose(0, 1).unsqueeze(0)
            normalized_motion = F.interpolate(
                normalized_motion,  # Add a batch dimension
                size=(self.target_length,),
                mode='linear',  # You can change the mode based on your needs
                align_corners=False
            ).squeeze(0).transpose(0, 1)  # Remove the batch dimension
        elif length > self.target_length:
            # Downsample using uniform sampling
            indices = torch.linspace(0, length - 1, steps=self.target_length).long()
            normalized_motion = normalized_motion[indices]

        with open(self.texts[index], 'r') as f:
            text = f.read()

        return {
            'x': normalized_motion,
            'length': self.target_length,  # Now all sequences have the same length
            'text': text
        }
        
    def __len__(self):
        return self.len

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
            if length <= 2000:
                motion = np.load(m, mmap_mode='r')
                current_max = motion.max()
                current_min = motion.min()
                if max_value is None or current_max > max_value:
                    max_value = current_max
                if min_value is None or current_min < min_value:
                    min_value = current_min
            else:
                filtered_sequences += 1
        print(f"Dataset '{path}': totale sequenze = {total_sequences}, sequenze filtrate = {filtered_sequences}")
    return max_value, min_value

def compute_global_stats(paths):
    max_value = None
    min_value = None
    total_sum = 0
    total_squared_sum = 0
    total_count = 0
    

    for path in paths:
        motions = glob.glob(os.path.join(path, "*motion.npy"))
        total_sequences = len(motions)
        filtered_sequences = 0
        count = 0
        for m in tqdm(motions, desc=f"Calcolo max/min/media/dev std in {path}"):
            length_path = m.replace("motion", "length").replace("npy", "txt")
            with open(length_path, 'r') as f:
                length = int(f.read().strip())
            
            if length <= 2000:
                motion = np.load(m, mmap_mode='r')
                
                if count != 0:
                    motions = np.concatenate((motions, motion), axis=0) 
                else:
                    motions = motion
                
                # Calcola il massimo e il minimo
                current_max = np.max(motion, axis=(0, 1))#.max()
                current_min = np.min(motion, axis=(0, 1))#.min()
                
                if max_value is None or np.linalg.norm(current_max) > max_value:
                    max_value = np.linalg.norm(current_max)
                if min_value is None or np.linalg.norm(current_min) < min_value:
                    min_value = np.linalg.norm(current_min)
                
                # Aggiorna somma e somma dei quadrati
                #total_sum += np.sum(motion, axis=(0, 1))#.sum()
                #total_squared_sum += np.sum(motion ** 2, axis=(0, 1))#.sum()
                #total_count += motion.shape[0]
            else:
                filtered_sequences += 1
                
            count+=1
        
        print(f"Dataset '{path}': totale sequenze = {total_sequences}, sequenze filtrate = {filtered_sequences}")

    # Calcolo della media e della deviazione standard
    mean_value = total_sum / total_count
    variance = (total_squared_sum / total_count) - (mean_value ** 2)
    std_dev = np.sqrt(variance)

    return max_value, min_value, mean_value, std_dev

def get_dataloaders(args, bs=1):
    dataset = {}
    train_data = Dataset(args.path_train)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)#, collate_fn=collate_fn)
    valid_data = Dataset(args.path_val)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)#, collate_fn=collate_fn)
    test_data = Dataset(args.path_test)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)#, collate_fn=collate_fn)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data for motion, text, and length")
    parser.add_argument('--path_train', type=str, default=f"{os.getcwd()}/kit_numpy/train", help='Path to the training data')
    parser.add_argument('--path_val', type=str, default=f"{os.getcwd()}/kit_numpy/validation", help='Path to the validation data')
    parser.add_argument('--path_test', type=str, default=f"{os.getcwd()}/kit_numpy/test", help='Path to the test data')
    
    args = parser.parse_args()

    # Calcolo max e min globali
    dataset_paths = [args.path_train, args.path_val, args.path_test]
    maxx, minn = compute_global_max_min(dataset_paths)
    print(f"Max globale: {maxx}, Min globale: {minn}")
    
    dataset_paths = [args.path_train, args.path_val, args.path_test]
    maxx, minn, mean, std = compute_global_stats(dataset_paths)
    print(f"Max globale: {maxx}, Min globale: {minn}, Media globale: {mean}, Deviazione standard globale: {std}")

    '''
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
        
