import os
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils import data
import torch 
import wandb

from data.motion import AMASSMotionLoader
from data.text import TextEmbeddings
from data.text_multi_motion import TextMultiMotionDataset
from model_lstm import Method, SkeletonLSTM, save_checkpoint, velocity_loss, rec_loss
import torch.optim as optim
from tqdm import tqdm


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    criterion = rec_loss
    method = Method("output")

    # Iperparametri
    hidden_size = 32
    num_epochs = 50
    bs = 4

    criterion_name = "Vel" if criterion == velocity_loss else "Rec"
    method_name = "1" if method.value == "current_frame" else "2"
    name = f"Loss{criterion_name}_method{method_name}_bs{bs}_H"

    wandb.init(project="skeleton_lstm_gpu", name=name)
    wandb.config.update({
        "hidden_size": hidden_size,
        "learning_rate": 0.0001,
        "epochs": num_epochs
    })

    motion_loader = AMASSMotionLoader(fps=20, base_dir="datasets/motions/AMASS_20.0_fps_nh_smplrifke")
    train_dataset = TextMultiMotionDataset(name="humanml3d", text_encoder=None, motion_loader=motion_loader, split="train")
    val_dataset = TextMultiMotionDataset(name="humanml3d", text_encoder=None, motion_loader=motion_loader, split="me/test")

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, collate_fn=train_dataset.collate_fn)

    model = SkeletonLSTM(hidden_size=hidden_size, feature_size=205, name=name, method=method)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    valid_loss = 100

    for e in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for batch_id, batch in pbar:
            optimizer.zero_grad()
            #motions = sample['motion'].flatten(2,3).to(device)
            #texts = sample['text'].to(device)
            
            motions = batch["x"].to(device)
            texts =  batch["text"]

            outputs = model(motions, texts)

            loss = criterion(outputs, motions)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description("Epoch {} Train Loss {:.5f}".format((e+1), running_loss/(batch_id+1)))
            #print(f"Epoch {e} - loss: {running_loss/i}")
            #train_loss.append(running_loss / len(train_loader))

            # Log training loss to wandb
        wandb.log({"train_loss": running_loss/(batch_id+1), "epoch": e+1})

        if (e + 1) % 1 == 0: 
            model.eval()  
            running_loss = 0.0
            pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

            with torch.no_grad():  
                for batch_id, batch in pbar:                    
                    motions = batch["x"].to(device)
                    texts =  batch["text"]
                   
                    outputs = model(motions, texts)

                    loss = criterion(outputs, motions)

                    running_loss += loss.item()
                    pbar.set_description("Epoch {} Valid Loss {:.5f}".format((e+1), running_loss/(batch_id+1)))

                avg_loss = running_loss/(batch_id+1)
                wandb.log({"valid_loss": avg_loss, "epoch": e+1})
                if avg_loss < valid_loss:
                    valid_loss = avg_loss
                    save_checkpoint(model, optimizer, num_epochs, model.save_path)



if __name__ == "__main__":
    train()