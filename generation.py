import json
import numpy as np
import torch
import os
from tools.smpl_layer import SMPLH
from visualize import *
from compare_visualizations import *
import torch.nn.functional as F

from model_lstm import Method, SkeletonLSTM
from tools.extract_joints import extract_joints
from model_transformer import SkeletonFormer

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
    model = model_class(hidden_size=hidden_size, feature_size=feature_size, name=name, method=method)  
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_input(idx, dataset="kitml", target_length=100):
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
        fps = 20
        annotations = json.load(open(f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"))
        text = annotations[idx]["annotations"][0]["text"]
        length_s = annotations[idx]["annotations"][0]["end"] - annotations[idx]["annotations"][0]["start"]
        length = int(fps * float(length_s))
        path = annotations[idx]["path"]
        start = int(annotations[idx.strip()]["annotations"][0]["start"]*fps)
        end = int(annotations[idx.strip()]["annotations"][0]["end"]*fps)
        full_motion = np.load(f"/andromeda/personal/lmandelli/stmc/datasets/motions/AMASS_20.0_fps_nh_smplrifke/{path}.npy")
        motion = full_motion[start:end] 
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

def normalize_output(output, dataset):
    if dataset=="kitml":
        #maxx, minn = 6675.25, -6442.60
        #output = minn + output * (maxx - minn)
        std, mean = 790.71, 357.44
        output = (output * std) + mean
        output = output.view(1, -1, 21, 3)
        output = output.squeeze(0)
        output = output.cpu()
        output = output.numpy()
    elif dataset=="humanml3d":
        output = output.cpu()#.numpy()

    return output


def generate(model_class, feature_size, model_path, id, name, dataset, y_is_z_axis=False, connections=True):

    model = load_model(model_class=model_class, model_path=model_path, name=name, feature_size=feature_size)
    model.to(device)

    motion, text, length = load_input(id, dataset=dataset)

    # Genera il movimento (output)
    output = generate_output(model, motion, text, length)

    output = normalize_output(output=output, dataset=dataset)
    
    save_path = f"{os.getcwd()}/visualizations/{name}_id{id}.mp4"
    if output.shape[-1] != 205:
        testset_path = f"{os.getcwd()}/kit_numpy/test"
        # np_data1 = np.load(f"{testset_path}/{id}_motion.npy")
        numpy_to_video(output, save_path, connections=connections, text=text)

    elif output.shape[-1] == 205:
        output = output[0]
        smplh = SMPLH(
            path="/andromeda/personal/lmandelli/stmc/deps/smplh",
            jointstype="both",
            input_pose_rep="axisangle",
            gender="male",
        )
        output_ = extract_joints(
                output,
                "smplrifke",
                fps=20,
                value_from="smpl",
                smpl_layer=smplh,
        )
        joints = output_["joints"]     # (num frames, 24, 3)
        vertices = output_["vertices"] # (num frames, 6890, 3)
        smpldata = output_["smpldata"] # (num frames, 6890, 3)

        if y_is_z_axis:
            x, mz, my = T(joints)
            joints = T(np.stack((x, my, mz), axis=0))

        numpy_to_video(joints, save_path, connections=connections, body_connections="guoh3djoints", text=text) 


if __name__ == "__main__":
    # Specifica il percorso del modello salvato e l'id dell'elemento di test
    name = "SkeletonFormer_LossRec_HumML_m1_bs512_h64_"

    model_class = SkeletonFormer if "SkeletonFormer" in name else SkeletonLSTM
    model_path = f"{os.getcwd()}/checkpoints/{name}.ckpt"
    dataset = "humanml3d" if "HumML" in name else "kitml"
    test_id = "000019" if dataset=="humanml3d" else "00029"
    y_is_z_axis = True if dataset=="humanml3d" else False
    feature_size = 205 if dataset=="humanml3d" else 63 

    generate(model_class=model_class, feature_size=feature_size, model_path=model_path, id=test_id, name=name, dataset=dataset, y_is_z_axis=y_is_z_axis)
