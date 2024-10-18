import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
from kit_dataloader import get_dataloaders

class SkeletonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SkeletonLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Text encoder
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.text_encoder = BartModel.from_pretrained('facebook/bart-large')

        # Motion encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1))

        # LSTM cell
        self.lstm_input_size = 16 * 5 * 5  # Modificato in base all'output dei layer conv

        self.lstm = nn.LSTM(input_size=self.lstm_input_size + self.text_encoder.config.hidden_size,
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)

        # Motion decoder
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)  


    def forward(self, motion_sequence, batch_size, text_input):
        seq_length, _, _ = motion_sequence.size()

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        inputs = self.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs.input_ids
        self.text_embedding = self.text_encoder(input_ids=input_ids, return_dict=True)['last_hidden_state']

        outputs = []

        # Iterazione su ogni frame della sequenza di movimento
        for t in range(seq_length):
            # Estrazione del frame corrente
            motion_frame = motion_sequence[t, :, :]
            print(motion_frame.size()) #(21,3)
            motion_frame = motion_frame.unsqueeze(0).unsqueeze(0)
            print(motion_frame.size()) #(1,1,21,3)

            # Passaggio attraverso il motion encoder
            motion_encoding = self.conv1(motion_frame)
            print(motion_encoding.size()) #(1,3,21,3)
            motion_encoding = F.relu(self.pool(motion_encoding))
            print(motion_encoding.size()) #(1,3,10,1)
            motion_encoding = F.relu(self.conv2(motion_encoding))
            print(motion_encoding.size()) #(1,16,10,1)
            motion_encoding = motion_encoding.view(batch_size, -1)

            # Concatenazione dell'embedding del testo con l'output del motion encoding
            combined_input = torch.cat((motion_encoding, self.text_embedding[:, t, :]), dim=-1)
            print(combined_input.size())

            # Passaggio attraverso l'LSTM
            lstm_output, (h0, c0) = self.lstm(combined_input.unsqueeze(1), (h0, c0))

            # Passaggio attraverso il motion decoder
            output = F.relu(self.fc1(lstm_output.squeeze(1)))
            output = F.relu(self.fc2(output))
            output = self.fc3(output)

            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)

        return outputs

# Funzione di training
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch, batch_size in train_loader:
        for item in batch:
            motion = item['motion']
            text = item['text']

            optimizer.zero_grad()

            outputs = model(motion, batch_size, text)
            loss = criterion(outputs, motion)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return running_loss / len(train_loader)


# Funzione di validazione
def validate(model, valid_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            motion = batch['motion']
            text_tokens = batch['text_tokens']
            
            outputs = model(motion, text_tokens)
            loss = criterion(outputs, motion)
            
            running_loss += loss.item()
    
    return running_loss / len(valid_loader)


if __name__ == '__main__':
    
    # Iperparametri
    input_size = 21 * 3  # Numero totale di coordinate
    hidden_size = 256
    num_layers = 2
    output_size = 21 * 3  # 21 giunti, 3 coordinate (x, y, z)

    # Inizializzazione del modello, della funzione di perdita e dell'ottimizzatore
    model = SkeletonLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Parser degli argomenti
    parser = argparse.ArgumentParser(description="Load data for motion, text, and length")
    parser.add_argument('--path_train', type=str, required=True, help='Path to the training data')
    parser.add_argument('--path_val', type=str, required=True, help='Path to the validation data')
    parser.add_argument('--path_test', type=str, required=True, help='Path to the test data')
    args = parser.parse_args()

    # Caricamento dei dati
    dataset = get_dataloaders(args)
    train_loader = dataset["train"]
    valid_loader = dataset["valid"]

    # Ciclo di addestramento
    num_epochs = 1

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        valid_loss = validate(model, valid_loader, criterion)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')