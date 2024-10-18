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
        samples = {}
        self.motions = []
        self.lengths = []
        self.texts = []
        motions = glob.glob(os.path.join(path, "*motion.npy"))
        for m in motions: 
            self.motions.append(m)
            self.lengths.append(m.replace("motion", "length").replace("npy", "txt"))
            self.texts.append(m.replace("motion", "text").replace("npy", "txt"))
        
        self.len = len(self.motions)

        self.maxx, self.minn = 6675.25, -6442.60
        # print(f"max {self.maxx}, min {self.minn}")

    def __getitem__(self, index):
        motion = torch.from_numpy(np.load(self.motions[index]))
        normalized_motion = (motion - self.minn) / (self.maxx - self.minn)
        length = motion.shape[0]
        with open(self.texts[index], 'r') as f:
            text = f.read()

        return {'motion': normalized_motion,
                'length': length,
                'text': text
                }

    def __len__(self):
        return self.len

def collate_fn(batch):
    max_length = max([item['length'] for item in batch])
    for item in batch:
        motion, length = item['motion'], item['length']
        padding_length = max_length - length
        padding = torch.zeros((padding_length, motion.shape[1], motion.shape[2]))
        item['motion'] = torch.cat((motion, padding), dim=0)
        item['length'] = max_length
    return batch

def get_dataloaders(args):
    dataset = {}
    train_data = Dataset(args.path_train)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    valid_data = Dataset(args.path_val)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    test_data = Dataset(args.path_test)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data for motion, text, and length")
    parser.add_argument('--path_train', type=str, default="tesi/LSTM_skeleton/kit_numpy/train", help='Path to the training data')
    parser.add_argument('--path_val', type=str, default="tesi/LSTM_skeleton/kit_numpy/validation", help='Path to the validation data')
    parser.add_argument('--path_test', type=str, default="tesi/LSTM_skeleton/kit_numpy/test", help='Path to the test data')
    
    args = parser.parse_args()
    
    dataloaders = get_dataloaders(args)
    
    max_frames = 0
    min_frames = 10000
    for batch in tqdm(dataloaders['test']):
        max_frames = max_frames if max_frames > batch[0]["motion"].shape[0] else batch[0]["motion"].shape[0]
        min_frames = min_frames if min_frames < batch[0]["motion"].shape[0] else batch[0]["motion"].shape[0]
        # print(batch)
        # print(len(batch))
        # print(batch[0]["motion"].size())
        # print(batch[1]["motion"].size())
        # print(batch[0]["length"])
        # print(batch[1]["length"])
        # print(batch[0]["motion"][0])
    print("max frames", max_frames)
    print("min frames", min_frames)
        