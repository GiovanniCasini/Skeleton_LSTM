import json
import numpy as np
import torch
import os
from tools.smpl_layer import SMPLH
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
        std, mean = 790.71, 357.44
        motion = (motion - mean) / std
        motion = torch.stack([motion])
        length = motion.shape[0]
       
        motion = motion.to(device)
        text_file = f"{testset_path}/{idx}_text.txt"
        length_file = f"{testset_path}/{idx}_length.txt"

        # Carica il testo
        with open(text_file, 'r') as f:
            text = f.read().strip()

        # Carica la durata (impostata su target_length)
        length = target_length

    else:
        fps = 20
        annotations = json.load(open(f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"))
        text = annotations[idx]["annotations"][0]["text"]
        length_s = annotations[idx]["annotations"][0]["end"] - annotations[idx]["annotations"][0]["start"]
        length = int(fps * float(length_s))
        path = annotations[idx]["path"]
        start = int(annotations[idx]["annotations"][0]["start"] * fps)
        end = int(annotations[idx]["annotations"][0]["end"] * fps)
        full_motion = np.load(f"/andromeda/personal/lmandelli/stmc/datasets/motions/AMASS_20.0_fps_nh_smplrifke/{path}.npy")
        motion = full_motion[start:end] 
        motion = torch.from_numpy(motion.reshape(1, motion.shape[0], motion.shape[1])).to(device=device).float()

    return motion, text, length

def generate_output(model, motion, text, length):
    # Genera l'output dal modello
    with torch.no_grad():
        output = model.predict(motion, text)
        # output = model(motion, text)
    return output

def normalize_output(output, dataset):
    if dataset == "kitml":
        std, mean = 790.71, 357.44
        output = (output * std) + mean
        output = output.view(1, -1, 21, 3)
        output = output.squeeze(0)
        output = output.cpu().numpy()
    elif dataset == "humanml3d":
        output = output.cpu().numpy()

    return output

def generate(model, motion, text, length, index, output_dir, name, dataset, y_is_z_axis=False):

    output = generate_output(model, motion, text, length)
    output = normalize_output(output=output, dataset=dataset)
    
    save_path = os.path.join(output_dir, f"output_{index}.npy")
    
    np.save(save_path, output)
    print(f"Output salvato in: {save_path}")

if __name__ == "__main__":

    name = "SkeletonFormer_LossRec_HumML_m1_bs1_h64_"
    model_class = SkeletonFormer if "SkeletonFormer" in name else SkeletonLSTM
    model_path = f"{os.getcwd()}/checkpoints/{name}.ckpt"
    dataset = "humanml3d" if "HumML" in name else "kitml"
    y_is_z_axis = True if dataset == "humanml3d" else False
    feature_size = 205 if dataset == "humanml3d" else 63 

    model = load_model(model_class=model_class, model_path=model_path, name=name, feature_size=feature_size)
    model.to(device)

    if dataset == "kitml":
        testset_path = f"{os.getcwd()}/kit_numpy/test"
        test_files = [f for f in os.listdir(testset_path) if f.endswith('_motion.npy')]
        test_ids = [f.split('_')[0] for f in test_files]
    else:
        testset_file_path = "/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/splits/test.txt"
        with open(testset_file_path, 'r') as f:
            test_ids = [line.strip() for line in f if line.strip()]
        annotations = json.load(open(f"/andromeda/personal/lmandelli/stmc/datasets/annotations/humanml3d/annotations.json"))
        tmp = []
        for idx in test_ids:
            path = annotations[idx]["path"]
            if "humanact12" not in path:
                tmp.append(idx)
            else:
                continue
        test_ids = tmp

    output_dir = os.path.join(os.getcwd(), "outputs", name)
    os.makedirs(output_dir, exist_ok=True)

    for index, test_id in enumerate(test_ids):
        print(f"Elaborazione dell'elemento {index + 1}/{len(test_ids)}: ID={test_id}")

        motion, text, length = load_input(test_id, dataset=dataset)

        generate(model=model,motion=motion,text=text,length=length,index=index,output_dir=output_dir,
            name=name,dataset=dataset,y_is_z_axis=y_is_z_axis)

