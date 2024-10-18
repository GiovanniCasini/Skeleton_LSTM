import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
from kit_dataloader import get_dataloaders
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkeletonLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(SkeletonLSTM, self).__init__()

        # self.num_layers = num_layers
        self.hidden_size = hidden_size

        # BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.text_encoder = BertModel.from_pretrained('bert-base-cased')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.lin_text = nn.Linear(768, int(self.hidden_size/2))

        # Motion encoder
        self.lin1 = nn.Linear(63, int(self.hidden_size/2))

        # LSTM cell
        self.lstm_cell = nn.LSTMCell(input_size=self.hidden_size, hidden_size=hidden_size)

        # Motion decoder
        self.lin2 = nn.Linear(self.hidden_size, 63) 
        nn.init.constant_(self.lin2.weight, 0)
        nn.init.constant_(self.lin2.bias, 0)

        self.save_path = f"/home/gcasini/tesi/LSTM_skeleton/checkpoints/velh32_m1batch32ini0_norelu_lr4.ckpt" 


    def forward(self, motions, texts):
        batch_size, seq_length, _ = motions.size()

        # Embedding del testo
        # print(texts)
        text_tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device) # padding='max_length', max_length = self.max_length
        # print(text_tokens)
        input_ids = text_tokens.input_ids
        mask = text_tokens.attention_mask
        # BERT
        last_hidden_state, text_embedding = self.text_encoder(input_ids=input_ids, attention_mask=mask,return_dict=False)
        # print(text_embedding.shape) # (8, 768)
        text_embedding = self.lin_text(text_embedding)
        # print(text_embedding.shape) # (8, 128)
        outputs = []

        motion_frame = motions[:, 0, :] # primo frame
        # print(motion_frame.size()) #(8, 63)
        
        # Iterazione su ogni frame della sequenza di movimento
        for t in range(seq_length):
            # Passaggio attraverso il motion encoder
            motion_encoding = self.lin1(motion_frame) #(8, 128)
            # print(motion_encoding.size()) 

            # Concatenazione dell'embedding del testo con l'output del motion encoding
            combined_input = torch.cat((motion_encoding, text_embedding), dim=-1)
            # print(combined_input.size()) #(8, 256)

            # Passaggio attraverso l'LSTM cell
            lstm_output = self.lstm_cell(combined_input)[0]

            # Passaggio attraverso il motion decoder
            output = self.lin2(lstm_output)
            # print(output.shape) #(8, 63)

            # metodo 1: somma frame corrente e predizione
            
            new_frame = motion_frame + output
            outputs.append(new_frame)
            motion_frame = new_frame
            '''
            # metodo 2: output direttamente nuovo frame
            outputs.append(output)
            motion_frame = output
            '''

        outputs = torch.stack(outputs, dim=1)

        return outputs

# Funzione di training
def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    valid_loss = 100
    for e in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch, sample in pbar:
            optimizer.zero_grad()
            #motions = sample['motion'].flatten(2,3).to(device)
            #texts = sample['text'].to(device)
            
            motions = torch.stack([el["motion"] for el in sample])
            motions = motions.flatten(2, 3) #(8, seq_len, 21, 3) -> (8, seq_len, 63)
            texts = [el["text"] for el in sample]

            motions = motions.to(device)
            texts = [text for text in texts]
            '''
            for k in range(1,motions.shape[1]):
                outputs = model(motions[:,:k], texts)
                loss = criterion(outputs, motions[:,:k])
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
            '''
            outputs = model(motions, texts)
            loss = criterion(outputs, motions)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description("Epoch {} Train Loss {:.5f}".format((e+1), running_loss/(batch+1)))

            # Log training loss to wandb
        wandb.log({"train_loss": running_loss/(batch+1), "epoch": e+1})

    
        if (e + 1) % 1 == 0: 
            model.eval()  
            running_loss = 0.0
            pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

            with torch.no_grad():  
                for batch, sample in pbar:                    
                    motions = torch.stack([el["motion"] for el in sample])
                    motions = motions.flatten(2, 3) #(8, seq_len, 21, 3) -> (8, seq_len, 63)
                    texts = [el["text"] for el in sample]

                    motions = motions.to(device)
                    texts = [text for text in texts]

                    outputs = model(motions, texts)

                    loss = criterion(outputs, motions)

                    running_loss += loss.item()
                    pbar.set_description("Epoch {} Valid Loss {:.5f}".format((e+1), running_loss/(batch+1)))

                avg_loss = running_loss/(batch+1)
                wandb.log({"valid_loss": avg_loss, "epoch": e+1})
                if avg_loss < valid_loss:
                    valid_loss = avg_loss
                    save_checkpoint(model, optimizer, num_epochs, model.save_path)
        

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    print("Saving model...")
    checkpoint = {
        "hidden_size":model.hidden_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    wandb.save(filename)


def rec_loss(predictions, target, loss_fn=nn.MSELoss()):
    loss = loss_fn(predictions, target)
    return loss

def velocity_loss(predictions, target, loss_fn=nn.MSELoss()):
    prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
    target_shift = target[:, 1:, :] - target[:, :-1, :]

    v_loss = torch.mean(loss_fn(prediction_shift, target_shift))
    r_loss = rec_loss(predictions, target, loss_fn)
    loss = r_loss + 100 * v_loss
    return loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Iperparametri
    hidden_size = 32
    num_epochs = 20

    # Initialize wandb and log hyperparameters
    wandb.init(project="skeleton_lstm_gpu", name = "velh32_m1batch32ini0_norelu_lr4")
    wandb.config.update({
        "hidden_size": hidden_size,
        "learning_rate": 0.0001,
        "epochs": num_epochs
    })

    # Inizializzazione del modello, della funzione di perdita e dell'ottimizzatore
    model = SkeletonLSTM(hidden_size)
    model.to(device)
    criterion = velocity_loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Parser degli argomenti
    parser = argparse.ArgumentParser(description="Load data for motion, text, and length")
    parser.add_argument('--path_train', type=str, default="/home/gcasini/tesi/LSTM_skeleton/kit_numpy/train", help='Path to the training data')
    parser.add_argument('--path_val', type=str, default="/home/gcasini/tesi/LSTM_skeleton/kit_numpy/validation", help='Path to the validation data')
    parser.add_argument('--path_test', type=str, default="/home/gcasini/tesi/LSTM_skeleton/kit_numpy/test", help='Path to the test data')
    args = parser.parse_args()

    # Caricamento dei dati
    dataset = get_dataloaders(args)
    train_loader = dataset["train"]
    valid_loader = dataset["valid"]
    test_loader = dataset["test"]

    train(model, train_loader, valid_loader, criterion, optimizer, num_epochs)

    # Finalize wandb
    wandb.finish()