import numpy as np
import torch
import os
from visualize import *
from compare_visualizations import *

from model_lstm import SkeletonLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_model(model_path):    
    # Carica il checkpoint
    checkpoint = torch.load(model_path)
    model = SkeletonLSTM(checkpoint["hidden_size"])  
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_input(id):
    testset_path = "/home/gcasini/tesi/LSTM_skeleton/kit_numpy/test"
    motion = torch.from_numpy(np.load(f"{testset_path}/{id}_motion.npy"))
    maxx, minn = 6675.25, -6442.60
    motion = (motion - minn) / (maxx - minn)
    motion = torch.stack([motion])
    motion = motion.flatten(2, 3)
    motion = motion.to(device)
    text_file = f"{testset_path}/{id}_text.txt"
    length_file = f"{testset_path}/{id}_length.txt"

    # Carica il testo
    with open(text_file, 'r') as f:
        text = f.read().strip()

    # Carica la durata
    with open(length_file, 'r') as f:
        length = f.read().strip()

    return motion, text, length

def generate_output(model, motion, text, length):
    # Genera l'output dal modello
    with torch.no_grad():
        output = model(motion, text)
        maxx, minn = 6675.25, -6442.60
        output = minn + output * (maxx - minn)

    return output

def save_output(output, id):
    """Salva l'output come file {id}.npy"""
    output_file = f"{id}.npy"
    np.save(output_file, output.cpu().numpy())  # Salva come numpy array

def main(model_path, id):

    model = load_model(model_path)
    model.to(device)

    motion, text, length = load_input(id)

    # Genera il movimento (output)
    output = generate_output(model, motion, text, length)
    output = output.view(1, -1, 21, 3)
    output = output.squeeze(0)
    output = output.cpu()
    output = output.numpy()

    # Salva l'output nel file {id}.npy
    #save_output(output, id)

    testset_path = "/home/gcasini/tesi/LSTM_skeleton/kit_numpy/test"
    np_data1 = np.load(f"{testset_path}/{id}_motion.npy")
    base_dir = os.getcwd()
    save_path = f"{base_dir}/tesi/LSTM_skeleton/visualizations/{id}/{id}_velh32_m1batch32ini0_norelu_lr4.mp4"
    numpy_to_video(output, save_path)
    #compare_numpy_to_video(np_data1, output, save_path)


if __name__ == "__main__":
    # Specifica il percorso del modello salvato e l'id dell'elemento di test
    test_id = "00029"
    model_path = f"/home/gcasini/tesi/LSTM_skeleton/checkpoints/velh32_m1batch32ini0_norelu_lr4.ckpt"
    
    main(model_path, test_id)
