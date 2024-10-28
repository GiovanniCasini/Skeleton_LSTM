import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
from kit_dataloader import get_dataloaders
import os
import enum
from data.motion import AMASSMotionLoader
from data.text import TextEmbeddings
from data.text_multi_motion import TextMultiMotionDataset
from torch.utils import data
from model_transformer import *
# from losses import robust_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Method(enum.Enum):
    method_1 = 'current_frame'
    method_2 = 'output'


class SkeletonLSTM(nn.Module):
    def __init__(self, device, method: Method=None, hidden_size=32, feature_size=63, name="model_name"):
        super(SkeletonLSTM, self).__init__()

        # self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        # BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.text_encoder = BertModel.from_pretrained('bert-base-cased')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.lin_text = nn.Linear(768, self.hidden_size)

        # Motion encoder
        self.lin1 = nn.Linear(feature_size, self.hidden_size)

        # LSTM cell
        #self.lstm_cell1 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=hidden_size)
        #self.lstm_cell2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=hidden_size)
        #self.lstm_cell3 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=hidden_size)

        # Motion decoder
        #self.lin2 = nn.Linear(self.hidden_size, feature_size) 
        #self.lin3 = nn.Linear(self.hidden_size, self.hidden_size) 
        self.lin4 = nn.Linear(self.hidden_size*2, feature_size) 
        
        self.combination = nn.MultiheadAttention(self.hidden_size, 4, batch_first=True)
        
        nn.init.constant_(self.lin4.weight, 0)
        nn.init.constant_(self.lin4.bias, 0)
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.save_path = f"{os.getcwd()}/checkpoints/{name}.ckpt" 
        self.method = method
        self.feature_size = feature_size
        self.name = name
        if self.method is None: 
            self.method = Method("current_frame")
            
        self.lstm = nn.LSTM(input_size=self.hidden_size, bidirectional=True, hidden_size=self.hidden_size, num_layers=5, batch_first=True)


    def forward(self, motions, texts):
        batch_size, seq_length, _ = motions.shape
        
        

        # Embedding del testo BERT
        text_tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
        input_ids = text_tokens.input_ids
        mask = text_tokens.attention_mask
        last_hidden_state, text_embedding = self.text_encoder(input_ids=input_ids, attention_mask=mask,return_dict=False) # (bs, 768)
        text_embedding = self.lin_text(text_embedding).unsqueeze(1)#.expand(batch_size, seq_length, self.hidden_size) # (bs, hideen_size/2)

        motion_frame = motions[:, 0, :] # primo frame (bs, 63)
        
        outputs = []
        
        motion_emb = self.lin1(motion_frame).unsqueeze(1) #(bs, 128)
        
        combination = self.combination(motion_emb, text_embedding, text_embedding)[0].expand(batch_size, seq_length, self.hidden_size)
        output = self.lstm(combination)[0]
        
        #output = self.lin2(output)
        #output = self.dropout(output)
        #output = self.lin3(output)
        #output = self.dropout(output)
        outputs = self.lin4(output) + motion_frame.unsqueeze(1)
        
        #hx = torch.zeros(batch_size, self.hidden_size).to(self.device)
        #cx = torch.zeros(batch_size, self.hidden_size).to(self.device)
        
        # Iterazione su ogni frame della sequenza di movimento
        #for t in range(seq_length):
        #    motion_encoding = self.lin1(motion_frame) #(bs, 128)
            
            #combined_input = torch.cat((motion_encoding, text_embedding), dim=-1) #(bs, 256)
            
        #    combined_input = self.combination(motion_encoding.unsqueeze(1), text_embedding.unsqueeze(1), text_embedding.unsqueeze(1))[0].squeeze(1)

        #    hx, cx = self.lstm_cell1(combined_input, (hx, cx))
            #lstm_output = self.lstm_cell2(lstm_output)[0]
            #lstm_output = self.lstm_cell3(lstm_output)[0]

        #    output = self.lin2(hx) #(bs, 63)

        #    new_frame = motion_frame + output
        #    outputs.append(new_frame)
        #    motion_frame = new_frame
 
        #outputs = torch.stack(outputs, dim=1)

        return outputs

# Funzione di training
def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
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
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description("Epoch {} Train Loss {:.7f}".format((e+1), running_loss/(batch_id+1)))
    
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

                avg_loss = running_loss/(batch_id+1)
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
    dataset_name = "humanml" # "kitml" or "humanml"
    model_class = SkeletonFormer # SkeletonFormer or SkeletonLSTM

    # Iperparametri
    hidden_size = 64
    num_epochs = 400
    bs = 1
    lr = 0.0001

    criterion_name = "Vel" if criterion == velocity_loss else "Rec"
    method_name = "1" if method.value == "current_frame" else "2"
    dataset_sigla = "K" if dataset_name == "kitml" else "H"
    name = f"New_Dataset_ModelTF_No_Displacements_Final{model_class.__name__}_Loss{criterion_name}_dataset{dataset_sigla}_method{method_name}_bs{bs}_h{hidden_size}"
    feature_size = 63 if dataset_name == "kitml" else 205

    print(f"name: {name}")

    print(f"Start training - name: {name} - bs {bs} - lr {lr} - epochs {num_epochs} - hidden size {hidden_size}")

    # Inizializzazione del modello, della funzione di perdita e dell'ottimizzatore
    #model = SkeletonLSTM(device, hidden_size=hidden_size, feature_size=feature_size, name=name, method=method)
    model = SkeletonFormer(hidden_size=hidden_size, feature_size=feature_size, name=name, method=method)
    #model = model_class(hidden_size=hidden_size, feature_size=feature_size, name=name, method=method)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Num parameters: {count_parameters(model)}")

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
    elif dataset_name == "humanml":
        motion_loader = AMASSMotionLoader(fps=20, base_dir="datasets/motions/AMASS_20.0_fps_nh_smplrifke")
        train_dataset = TextMultiMotionDataset(name="humanml3d", text_encoder=None, motion_loader=motion_loader, split="train")
        val_dataset = TextMultiMotionDataset(name="humanml3d", text_encoder=None, motion_loader=motion_loader, split="me/test")

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, collate_fn=train_dataset.collate_fn)
        valid_loader = data.DataLoader(dataset=val_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, collate_fn=train_dataset.collate_fn)


    train(model, train_loader, valid_loader, criterion, optimizer, num_epochs)

