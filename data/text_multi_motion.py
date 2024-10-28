import os
import codecs as cs
import json

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion
from .text_motion import load_annotations, read_split
import torch


class TextMultiMotionDataset(Dataset):
    def __init__(
        self,
        name: str,
        motion_loader,
        text_encoder,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = False,
        tiny: bool = False,
        # only during training
        drop_motion_perc: float = 0.0,
        drop_cond: float = 0.0,
        drop_trans: float = 0.0,
    ):
        if tiny:
            split = split + "_tiny"
        path = f"/andromeda/personal/lmandelli/stmc/datasets/annotations/{name}"
        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)

        # self.text_encoder = text_encoder
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        # Just for debugging
        # load_annotations(path)
        self.annotations = load_annotations(path, name="annotations.json") 

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split: 
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = "train" in split
        self.drop_motion_perc = drop_motion_perc
        self.drop_cond = drop_cond
        self.drop_trans = drop_trans

        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))

        annotation = annotations["annotations"][index]
        text = annotation["text"]

        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
        )

        x = motion_x_dict["x"]
        length = motion_x_dict["length"]

        output = {
            "x": x,
            "text": text,
            "keyid": keyid,
            "length": length,
        }
        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            path = val["path"]

            # remove humanact12
            # buggy left/right + no SMPL
            if "humanact12" in path:
                continue

            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
