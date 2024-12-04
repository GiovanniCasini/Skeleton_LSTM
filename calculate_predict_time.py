import json
import numpy as np
import torch
import os
from tools.smpl_layer import SMPLH
from visualize import *
from compare_visualizations import *
import torch.nn.functional as F
import pathlib
import matplotlib.pyplot as plt

from model_lstm import Method, SkeletonLSTM
from tools.extract_joints import extract_joints
from model_transformer import SkeletonFormer
from text_encoder import *
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def T(x):
    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))

def load_model(model_class, model_path, name, feature_size=63):    
    # Carica il checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    feature_size = feature_size
    hidden_size = checkpoint["hidden_size"]
    method = Method("current_frame") if "_m1" in name else (Method("output") if "_m2" in name else 0)
    text_encoder = CLIP if "CLIP" in name else (Bert if "Bert" in name else Bart) 
    text_encoder = text_encoder(device=device)
    model = model_class(text_encoder=text_encoder, hidden_size=hidden_size, feature_size=feature_size, name=name, method=method)  
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_input(idx, dataset="kitml", target_length=100, data_format="Joints", AMASS_path="/andromeda/personal/lmandelli/stmc/datasets/motions/AMASS_20.0_fps_nh_smplrifke/"):

    if data_format == "Joints":
        if dataset == "kitml":
            testset_path = f"{os.getcwd()}/kit_numpy/validation"
            motion = torch.from_numpy(np.load(f"{testset_path}/{idx}_motion.npy")).reshape(-1, 63)
            #maxx, minn = 6675.25, -6442.60
            std, mean = 790.71, 357.44
            motion = (motion - mean) / (std)
            motion = torch.stack([motion])
            length = motion.shape[0]
        
            motion = motion.to(device)
            text_file = f"{testset_path}/{idx}_text.txt"
            length_file = f"{testset_path}/{idx}_length.txt"

            # Load the text
            with open(text_file, 'r') as f:
                text = f.read().strip()

            # Load the duration (set to target_length)
            length = target_length
        else:
            raise NotImplementedError("HumanML3D with joints data not implemented")
        
    elif data_format == "Smpl":
        fps = 20
        annotations = json.load(open(f"/andromeda/personal/lmandelli/stmc/datasets/annotations/{dataset}/annotations.json"))
        text = annotations[idx]["annotations"][0]["text"]
        length_s = annotations[idx]["annotations"][0]["end"] - annotations[idx]["annotations"][0]["start"]
        length = int(fps * float(length_s))
        path = annotations[idx]["path"]
        
        if "humanact12" in path:
            raise NotImplementedError()
        
        start = int(annotations[idx.strip()]["annotations"][0]["start"]*fps)
        end = int(annotations[idx.strip()]["annotations"][0]["end"]*fps)
        full_motion = np.load(f"{AMASS_path}{path}.npy")
        motion = full_motion[start:end] 
        if motion.shape[-1] == 205:
            motion = torch.from_numpy(motion.reshape(1, motion.shape[0], motion.shape[1])).to(device=device).float()
    
    return motion, text, length

   
def generate_output(time_predictions, lengths, model, motion, text, length, plot):
    if plot:
        lengths.append(length)
        with torch.no_grad():
            start_time = time.time()
            output = model.predict(motion, text, length)
            end_time = time.time()
            time_predictions.append(end_time-start_time)
        return output
    else:
        with torch.no_grad():
            output = model.predict(motion, text, length)


def generate(time_predictions, lengths, index, model, feature_size, model_path, id, name, dataset, y_is_z_axis=False, connections=True, data_format="Joints", out_format="Joints", plot=False):

    motion, text, length = load_input(id, dataset=dataset, data_format=data_format)

    # Genera il movimento (output)
    #output = generate_output(time_predictions, lengths, model, motion, text, length)
    output = generate_output(time_predictions, lengths, model, motion, text, index, plot)

def plot_inference_time(lengths, time_predictions):
    sorted_data = sorted(zip(lengths, time_predictions))
    sorted_lengths, sorted_times = zip(*sorted_data)  
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_lengths, sorted_times, width=0.8, edgecolor='black', alpha=0.5)
    plt.xlabel('Sequence Length')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Inference Time vs. Sequence Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    filename = "InferenceTimeHistogram5.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close()  



if __name__ == "__main__":

    name = "SkeletonFormer_LossRec_KitML_m1_bs1_h512_textEmbCLIP_DataSmpl__4l"
    model_class = SkeletonFormer if "SkeletonFormer" in name else SkeletonLSTM
    model_path = f"{os.getcwd()}/checkpoints/{name}.ckpt"
    dataset = "humanml3d" if "HumML" in name else "kitml"
    data_format = "Smpl" if "Smpl" in name else "Joints"
    y_is_z_axis = True if (data_format=="Smpl") else False
    feature_size = 63 if data_format=="Joints" else 205
    model = load_model(model_class=model_class, model_path=model_path, name=name, feature_size=feature_size)
    model.to(device)
    
    ids = []
    annotations = json.load(open(f"/andromeda/personal/lmandelli/stmc/datasets/annotations/{dataset}/annotations.json"))
    directory_path = "/andromeda/personal/lmandelli/stmc/datasets/motions/AMASS_20.0_fps_nh_smplrifke/"
    for idx, key in enumerate(annotations.keys()):
        annotation = annotations[key]
        ids.append(key)

    time_predictions = []
    lengths = []
    plot = False
    for index,id in enumerate(ids[:400]):
        print(f"{index+1}/{len(ids)}")
        if index < 3:
            continue
        else:
            if index == 3:
                generate(time_predictions, lengths, index+1, model=model, 
                    feature_size=feature_size, 
                    model_path=model_path, id=id, 
                    name=name, 
                    dataset=dataset, 
                    y_is_z_axis=y_is_z_axis, 
                    data_format=data_format, plot=plot)
                plot = True
                
            generate(time_predictions, lengths, index+1, model=model, 
                    feature_size=feature_size, 
                    model_path=model_path, id=id, 
                    name=name, 
                    dataset=dataset, 
                    y_is_z_axis=y_is_z_axis, 
                    data_format=data_format,
                    plot=plot 
            )   

    time_predictions_mean = np.mean(time_predictions)
    print(f"mean prediction times: {time_predictions_mean}")  
    plot_inference_time(lengths, time_predictions)