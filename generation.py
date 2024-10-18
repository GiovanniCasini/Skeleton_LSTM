import json
import numpy as np
import torch
import os
from visualize import *
from compare_visualizations import *

from model_lstm import Method, SkeletonLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_model(model_path, name):    
    # Carica il checkpoint
    checkpoint = torch.load(model_path)
    feature_size = checkpoint["model_state_dict"]["lin1.weight"].shape[-1]
    hidden_size = checkpoint["hidden_size"]
    method = Method("current_frame") if "method1" in name else (Method("output") if "method2" in name else 0)
    model = SkeletonLSTM(hidden_size=hidden_size, feature_size=feature_size, name=name, method=method)  
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_input(idx, dataset="kitml"):
    if dataset == "kitml":
        testset_path = f"{os.getcwd()}/kit_numpy/test"
        motion = torch.from_numpy(np.load(f"{testset_path}/{idx}_motion.npy"))
        maxx, minn = 6675.25, -6442.60
        motion = (motion - minn) / (maxx - minn)
        motion = torch.stack([motion])
        motion = motion.flatten(2, 3)
        motion = motion.to(device)
        text_file = f"{testset_path}/{idx}_text.txt"
        length_file = f"{testset_path}/{idx}_length.txt"

        # Carica il testo
        with open(text_file, 'r') as f:
            text = f.read().strip()

        # Carica la durata
        with open(length_file, 'r') as f:
            length = f.read().strip()
        
    else:
        fps = 20
        annotations = json.load(open(f"datasets/annotations/humanml3d/annotations.json"))
        text = annotations[idx]["annotations"][0]["text"]
        length_s = annotations[idx]["annotations"][0]["end"] - annotations[idx]["annotations"][0]["start"]
        length = int(fps * float(length_s))
        path = annotations[idx]["path"]
        start = int(annotations[idx.strip()]["annotations"][0]["start"]*fps)
        end = int(annotations[idx.strip()]["annotations"][0]["end"]*fps)
        full_motion = np.load(f"{os.getcwd()}/datasets/motions/AMASS_20.0_fps_nh_smplrifke/{path}.npy")
        motion = full_motion[start:end] 
        motion = torch.from_numpy(motion.reshape(1, motion.shape[0], motion.shape[1])).to(device=device).float()

    return motion, text, length

   
def generate_output(model, motion, text, length):
    # Genera l'output dal modello
    with torch.no_grad():
        output = model(motion, text)

    return output

def save_output(output, id):
    """Salva l'output come file {id}.npy"""
    output_file = f"{id}.npy"
    np.save(output_file, output.cpu().numpy())  # Salva come numpy array

def normalize_output(output, dataset):
    if dataset=="kitml":
        maxx, minn = 6675.25, -6442.60
        output = minn + output * (maxx - minn)
        output = output.view(1, -1, 21, 3)
        output = output.squeeze(0)
        output = output.cpu()
        output = output.numpy()
    elif dataset=="humanml3d":
        output = output.cpu().numpy()

    return output


def generate(model_path, id, name, dataset):

    model = load_model(model_path, name=name)
    model.to(device)

    motion, text, length = load_input(id, dataset=dataset)

    # Genera il movimento (output)
    output = generate_output(model, motion, text, length)

    output = normalize_output(output=output, dataset=dataset)

    # Salva l'output nel file {id}.npy
    #save_output(output, id)

    testset_path = f"{os.getcwd()}/kit_numpy/test"
    # np_data1 = np.load(f"{testset_path}/{id}_motion.npy")
    save_path = f"{os.getcwd()}/visualizations/{name}_id{id}.mp4"
    numpy_to_video(output, save_path)
    #compare_numpy_to_video(np_data1, output, save_path)


if __name__ == "__main__":
    # Specifica il percorso del modello salvato e l'id dell'elemento di test
    name = "LossVel_method1_bs4_H"
    model_path = f"{os.getcwd()}/checkpoints/{name}.ckpt"
    dataset = "humanml3d" if "H" in name else "kitml"
    test_id = "000000" if dataset=="humanml3d" else "03902"

    generate(model_path=model_path, id=test_id, name=name, dataset=dataset)
