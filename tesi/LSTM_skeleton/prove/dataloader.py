import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import glob
import argparse
from transformers import BartTokenizer

class CustomDataset(Dataset):
    """Custom dataset class."""

    def __init__(self, path):
        """
        Inizializza il dataset.

        Args:
            path (str): Percorso della cartella contenente i dati.
        """
        self.path = path
        # Trova tutti i file di movimento, lunghezze e testo nella cartella specificata
        self.motions = glob.glob(os.path.join(path, "*motion.npy"))
        self.lengths = glob.glob(os.path.join(path, "*length*"))
        self.texts = glob.glob(os.path.join(path, "*text*"))

        self.len = len(self.motions)
        # Inizializza il tokenizer BART pre-addestrato per il testo
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    def __getitem__(self, index):
        """
        Ottiene un singolo elemento dal dataset.

        Args:
            index (int): Indice dell'elemento nel dataset.

        Returns:
            dict: Dizionario contenente il movimento, la lunghezza, il testo, i token del testo e la maschera di attenzione.
        """
        # Carica il movimento dal file .npy
        motion = torch.from_numpy(np.load(self.motions[index])).float()
        # Legge la lunghezza del movimento dal file di lunghezza
        with open(self.lengths[index], 'r') as f:
            length = int(f.read())
        # Legge il testo dalla corrispondente file di testo
        with open(self.texts[index], 'r') as f:
            text = f.read()

        # Tokenizza il testo utilizzando il tokenizer BART
        text_tokens = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        return {'motion': motion,
                'length': length,
                'text': text,
                'text_tokens': text_tokens['input_ids'].squeeze(),
                'text_attention_mask': text_tokens['attention_mask'].squeeze()
                }

    def __len__(self):
        """
        Restituisce la lunghezza del dataset.

        Returns:
            int: Lunghezza del dataset.
        """
        return self.len

def collate_fn(batch):
    """
    Funzione di collate personalizzata per combinare batch di dati.

    Args:
        batch (list): Lista di dizionari, ciascuno contenente dati di un singolo esempio.

    Returns:
        dict: Dizionario contenente i tensori paddati e altre informazioni del batch.
    """
    # Estrae le informazioni dai singoli esempi nel batch
    motions = [item['motion'] for item in batch]
    lengths = [item['length'] for item in batch]
    texts = [item['text'] for item in batch]
    text_tokens = [item['text_tokens'] for item in batch]
    text_attention_masks = [item['text_attention_mask'] for item in batch]

    # Calcola la lunghezza massima delle sequenze di movimento nel batch
    max_length = max(lengths)

    # Crea un tensore vuoto per contenere i movimenti paddati
    padded_motions = torch.zeros((len(batch), max_length, 21, 3)) 
    # Popola il tensore paddato con i dati di movimento
    for i, motion in enumerate(motions):
        end = lengths[i]
        padded_motions[i, :end, :, :] = motion[:, :21, :] 

    # Converte le liste di tensori in tensori singoli
    text_tokens = torch.stack(text_tokens)
    text_attention_masks = torch.stack(text_attention_masks)

    return {'motion': padded_motions,
            'length': torch.tensor(lengths),
            'text': texts,
            'text_tokens': text_tokens,
            'text_attention_mask': text_attention_masks}

def get_dataloaders(args):
    """
    Ottiene i DataLoader per il training, la validazione e il test.

    Args:
        args: Argomenti passati dalla riga di comando.

    Returns:
        dict: Dizionario contenente i DataLoader per i tre set di dati.
    """
    dataset = {}
    # Crea il DataLoader per il set di dati di training
    train_data = CustomDataset(args.path_train)
    dataset["train"] = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    # Crea il DataLoader per il set di dati di validazione
    valid_data = CustomDataset(args.path_val)
    dataset["valid"] = DataLoader(dataset=valid_data, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    # Crea il DataLoader per il set di dati di test
    test_data = CustomDataset(args.path_test)
    dataset["test"] = DataLoader(dataset=test_data, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data for motion, text, and length")
    parser.add_argument('--path_train', type=str, required=True, help='Path to the training data')
    parser.add_argument('--path_val', type=str, required=True, help='Path to the validation data')
    parser.add_argument('--path_test', type=str, required=True, help='Path to the test data')
    
    args = parser.parse_args()

    '''
    index = 1
    texts = glob.glob(os.path.join("kit_numpy/train", "*text*"))
    with open(texts[index], 'r') as f:
        text = f.read()
    print(text)    
    lengths = glob.glob(os.path.join("kit_numpy/validation", "*length*"))
    with open(lengths[index], 'r') as f:
        length = int(f.read())
    print(length)
    motions = glob.glob(os.path.join("kit_numpy/train", "*motion.npy"))
    motion = torch.from_numpy(np.load(motions[index])).float()
    print(motion.size())
    #print(motion)
    #'''

    dataloaders = get_dataloaders(args)
    
    for batch in tqdm(dataloaders['train']):
        #print(batch)
        #motion_batch = batch['motion']
        #print("Dimensione del batch di motion:", motion_batch.size())
        pass

