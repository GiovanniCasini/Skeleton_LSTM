import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from kit_dataloader import get_dataloaders
import os
from model_lstm import Method
from data.motion import AMASSMotionLoader
from data.text_multi_motion import TextMultiMotionDataset
from torch.utils import data
from model_transformer import *
import wandb
from text_encoder import *
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Funzione di training
def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs):
    valid_loss = 100
    for e in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_id, batch in pbar:
            optimizer.zero_grad()
           
            motions = batch["x"].to(device)
            texts =  batch["text"]
            
            motions = motions[:, 1:, :] - motions[:, :-1, :]

            outputs = model(motions, texts)
            loss = criterion(outputs[:, :-1, :], motions[:, 1:, :])
            #loss = criterion(outputs, motions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description("Epoch {} Train Loss {:.7f}".format((e+1), running_loss/(batch_id+1)))

        # Log training loss to wandb
        wandb.log({"train_loss": running_loss/(batch_id+1), "epoch": e+1})

    
        if (e + 1) % 1 == 0: 
            model.eval()  
            running_loss = 0.0
            pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

            with torch.no_grad():  
                for batch_id, batch in pbar:                    
        
                    motions = batch["x"].to(device)
                    texts =  batch["text"]
                    
                    motions = motions[:, 1:, :] - motions[:, :-1, :]

                    outputs = model(motions, texts)

                    loss = criterion(outputs[:, :-1, :], motions[:, 1:, :])
                    #loss = criterion(outputs, motions)


                    running_loss += loss.item()
                    pbar.set_description("Epoch {} Valid Loss {:.7f}".format((e+1), running_loss/(batch_id+1)))
                    wandb.log({"val_loss": running_loss/(batch_id+1), "epoch": e+1})

                avg_loss = running_loss/(batch_id+1)
                scheduler.step(avg_loss)
                if avg_loss < valid_loss:
                    valid_loss = avg_loss
                    save_checkpoint(model, optimizer, num_epochs, model.save_path)
        

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    print("Saving model...")
    checkpoint = {
        "feature_size": model.feature_size,
        "name":model.name,
        "hidden_size":model.hidden_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    

def rec_loss(predictions, target, loss_fn=nn.MSELoss()):
    loss = loss_fn(predictions, target)
    return loss

def velocity_loss(predictions, target, loss_fn=nn.MSELoss()):
    prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
    target_shift = target[:, 1:, :] - target[:, :-1, :]

    v_loss = torch.mean(loss_fn(prediction_shift, target_shift))# * 10
    r_loss = rec_loss(predictions, target, loss_fn)
    loss = r_loss + v_loss
    return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    print(f"Using device: {device}")

    criterion = rec_loss
    method = Method("current_frame")
    dataset_name = "kitml" # "kitml" or "humanml3d"
    model_class = SkeletonFormer # SkeletonFormer or SkeletonLSTM
    text_encoder = CLIP # Bert | Bart | CLIP
    extra_text = "_4l"
    data_format = "Smpl" # "Joints" | "Smpl"

    # Iperparametri
    hidden_size = 1024
    num_epochs = 500
    bs = 1
    lr = 0.0001

    criterion_name = "Vel" if criterion == velocity_loss else "Rec"
    method_name = "1" if method.value == "current_frame" else "2"
    dataset_sigla = "HumML" if dataset_name == "humanml3d" else "KitML"
    feature_size = 63 if data_format == "Joints" else 205
    name = f"{model_class.__name__}_Loss{criterion_name}_{dataset_sigla}_m{method_name}_bs{bs}_h{hidden_size}_textEmb{text_encoder.__name__}_Data{data_format}_{extra_text}"

    print(f"name: {name}")

    wandb.init(project="Skeleton_LSTM", name=name)

    print(f"Start training - name: {name} - bs {bs} - lr {lr} - epochs {num_epochs} - hidden size {hidden_size}")

    # Inizializzazione del modello, della funzione di perdita e dell'ottimizzatore
    text_encoder = text_encoder(device=device)
    model = model_class(hidden_size=hidden_size, feature_size=feature_size, name=name, method=method, text_encoder=text_encoder)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    if data_format == "Joints":
        if dataset_name == "kitml":
            # Parser degli argomenti
            parser = argparse.ArgumentParser(description="Load data for motion, text, and length")
            parser.add_argument('--path_train', type=str, default=f"{os.getcwd()}/kit_numpy/train", help='Path to the training data')
            parser.add_argument('--path_val', type=str, default=f"{os.getcwd()}/kit_numpy/validation", help='Path to the validation data')
            parser.add_argument('--path_test', type=str, default=f"{os.getcwd()}/kit_numpy/test", help='Path to the test data')
            parser.add_argument('--info', type=str, default="", help='Experiment info')
            args = parser.parse_args()

            # Caricamento dei dati
            dataset = get_dataloaders(args, bs=bs)
            train_loader = dataset["train"]
            valid_loader = dataset["valid"]
            test_loader = dataset["test"]
        else:
            raise NotImplementedError("HumanML3D with joints data not implemented")

    elif data_format == "Smpl":
        motion_loader = AMASSMotionLoader(fps=20, base_dir="/andromeda/personal/lmandelli/stmc/datasets/motions/AMASS_20.0_fps_nh_smplrifke")
        train_dataset = TextMultiMotionDataset(name=dataset_name, text_encoder=None, motion_loader=motion_loader, split="train")
        val_dataset = TextMultiMotionDataset(name=dataset_name, text_encoder=None, motion_loader=motion_loader, split="val")

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, collate_fn=train_dataset.collate_fn)
        valid_loader = data.DataLoader(dataset=val_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, collate_fn=train_dataset.collate_fn)
        
    train(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs)