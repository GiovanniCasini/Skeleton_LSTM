import json
import numpy as np
import os
import time

import torch
import torch.nn.functional as F

from model_lstm import Method, SkeletonLSTM
from model_transformer import SkeletonFormer
from text_encoder import *

from generation import load_input, normalize_output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def T(x):
    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))

def load_model(model_class, model_path, name, feature_size=63):    
    # Carica il checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    hidden_size = checkpoint["hidden_size"]
    method = Method("current_frame") if "_m1" in name else (Method("output") if "_m2" in name else 0)
    text_encoder = CLIP if "CLIP" in name else (Bert if "Bert" in name else Bart) 
    text_encoder = text_encoder(device=device)
    model = model_class(text_encoder=text_encoder, hidden_size=hidden_size, feature_size=feature_size, name=name, method=method)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

times = {}

def generate_output(model, motion, text, length):
    # Genera l'output dal modello
    with torch.no_grad():
        start = time.time()
        output = model.predict(motion, text)
        duration = time.time() - start
        if motion.shape[1] not in times:
            times[motion.shape[1]] = []
        times[motion.shape[1]].append(duration)
        print(f"Tempo di esecuzione: {duration} per {motion.shape[1]} frames")
        # output = model(motion, text)
    return output


def generate(model, motion, text, length, index, output_dir, name, dataset, test_id, y_is_z_axis=False, data_format="Joints", out_format="Joints"):

    output = generate_output(model, motion, text, length)
    output = normalize_output(output=output, data_format=data_format, out_format=out_format) # [num frames, num joints, 3]
    
    save_path = os.path.join(output_dir, f"{test_id}.npy")
    
    # np.save(save_path, output)
    print(f"Output salvato in: {save_path}")

if __name__ == "__main__":

    name = "SkeletonFormer_LossRec_KitML_m1_bs1_h256_textEmbCLIP_DataSmpl__4l"
    model_class = SkeletonFormer if "SkeletonFormer" in name else SkeletonLSTM
    model_path = f"{os.getcwd()}/checkpoints/{name}.ckpt"
    dataset = "humanml3d" if "HumML" in name else "kitml"
    data_format = "Smpl" if "Smpl" in name else "Joints"
    y_is_z_axis = True if data_format=="Smpl" else False
    feature_size = 63 if data_format=="Joints" else 205
    # Specifica se salvare in file (nframes, 6890, 3) o (nframes, 205) o (nframes, 24,3)
    # Joints serve per fare l'evaluate
    # Vertices per fare il render con Smpl
    out_format = "Vertices" # "Vertices", "Smpl", "Joints"

    model = load_model(model_class=model_class, model_path=model_path, name=name, feature_size=feature_size)
    model.to(device)

    if data_format == "Joints":
        if dataset == "kitml":
            testset_path = f"{os.getcwd()}/kit_numpy/test"
            test_files = [f for f in os.listdir(testset_path) if f.endswith('_motion.npy')]
            test_ids = [f.split('_')[0] for f in test_files]
    else:
        testset_file_path = f"/andromeda/personal/lmandelli/stmc/datasets/annotations/{dataset}/splits/test.txt"
        with open(testset_file_path, 'r') as f:
            test_ids = [line.strip() for line in f if line.strip()]
        annotations = json.load(open(f"/andromeda/personal/lmandelli/stmc/datasets/annotations/{dataset}/annotations.json"))
        tmp = []
        for idx in test_ids:
            path = annotations[idx]["path"]
            if "humanact12" not in path:
                tmp.append(idx)
            else:
                continue
        test_ids = tmp

    output_dir = os.path.join(os.getcwd(), "outputs", name, out_format)
    os.makedirs(output_dir, exist_ok=True)

    for index, test_id in enumerate(test_ids):
        print(f"Elaborazione dell'elemento {index + 1}/{len(test_ids)}: ID={test_id}")

        motion, text, length = load_input(test_id, dataset=dataset, data_format=data_format)

        generate(model=model,
                 motion=motion,
                 text=text,
                 length=length,
                 index=index,
                 output_dir=output_dir,            
                 name=name,
                 dataset=dataset,
                 y_is_z_axis=y_is_z_axis, 
                 test_id=test_id, 
                 data_format=data_format,
                 out_format=out_format)
        
    with open(f"{os.getcwd()}/times.json", "w") as outfile: 
        json.dump(times, outfile)
    mean_time = np.mean([np.mean(times[l]) for l in times.keys()])
    print(f"Tempo medio per frame: {mean_time}")