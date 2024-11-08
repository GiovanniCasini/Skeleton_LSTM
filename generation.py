import json
import numpy as np
import torch
import os
from tools.smpl_layer import SMPLH
from visualize import *
from compare_visualizations import *
import torch.nn.functional as F
import pathlib

from model_lstm import Method, SkeletonLSTM
from tools.extract_joints import extract_joints
from model_transformer import SkeletonFormer
from text_encoder import *


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
            testset_path = f"{os.getcwd()}/kit_numpy/test"
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

   
def generate_output(model, motion, text, length):
    # Genera l'output dal modello
    with torch.no_grad():
        output = model.predict(motion, text)
        #output = model(motion, text)
    return output


def save_output(output, id):
    """Salva l'output come file {id}.npy"""
    output_file = f"{id}.npy"
    np.save(output_file, output.cpu().numpy())  # Salva come numpy array

def normalize_output(output, data_format="Joints"):

    if data_format=="Joints":
        #maxx, minn = 6675.25, -6442.60
        #output = minn + output * (maxx - minn)
        std, mean = 790.71, 357.44
        output = (output * std) + mean
        output = output.view(1, -1, 21, 3)
        output = output.squeeze(0)
        output = output.cpu()
        output = output.numpy()

    elif data_format=="Smpl":
        output = output.cpu() #.numpy()

        output = output[0]
        smplh = SMPLH(
            path="/andromeda/personal/lmandelli/stmc/deps/smplh",
            jointstype="both",
            input_pose_rep="axisangle",
            gender="male",
        )
        output_ = extract_joints(
                output.cpu(),
                #motion[0].cpu(),
                "smplrifke",
                fps=20,
                value_from="smpl",
                smpl_layer=smplh,
        )
        joints = output_["joints"]     # (num frames, 24, 3)
        vertices = output_["vertices"] # (num frames, 6890, 3)
        smpldata = output_["smpldata"] # (num frames, 6890, 3)
        output = joints

    return output 


def generate(model_class, feature_size, model_path, id, name, dataset, y_is_z_axis=False, connections=True, data_format="Joints"):

    model = load_model(model_class=model_class, model_path=model_path, name=name, feature_size=feature_size)
    model.to(device)

    motion, text, length = load_input(id, dataset=dataset, data_format=data_format)

    # Genera il movimento (output)
    output = generate_output(model, motion, text, length)

    output = normalize_output(output=output, data_format=data_format) #
    
    save_dir = f"{os.getcwd()}/visualizations/{name}/"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True) 
    save_path = f"{save_dir}/{id}.mp4"

    if output.shape[-1] != 205:
        testset_path = f"{os.getcwd()}/kit_numpy/test"
        # np_data1 = np.load(f"{testset_path}/{id}_motion.npy")
        numpy_to_video(output, save_path, connections=connections, text=text)

    elif data_format=="Smpl":

        if y_is_z_axis:
            x, mz, my = T(joints)
            joints = T(np.stack((x, my, mz), axis=0))

        numpy_to_video(joints, save_path, connections=connections, body_connections="guoh3djoints", text=text) 


if __name__ == "__main__":
    # Specifica il percorso del modello salvato e l'id dell'elemento di test
    name = "SkeletonFormer_LossRec_KitML_m1_bs1_h256_textEmbCLIP_DataSmpl__4l"
    ids = ["00003","00029","00069","00507","00003","00915","00971","01260","01321"]
    #ids = ["000000","000004","000009","000013","000021","000015","000019","000032"]

    for id in ids:
        
        model_class = SkeletonFormer if "SkeletonFormer" in name else SkeletonLSTM
        model_path = f"{os.getcwd()}/checkpoints/{name}.ckpt"
        dataset = "humanml3d" if "HumML" in name else "kitml"
        data_format = "Smpl" if "Smpl" in name else "Joints"
        y_is_z_axis = True if data_format=="Smpl" else False
        feature_size = 63 if data_format=="Joints" else 205

        generate(model_class=model_class, feature_size=feature_size, model_path=model_path, id=id, name=name, dataset=dataset, y_is_z_axis=y_is_z_axis, data_format=data_format)     