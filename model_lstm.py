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
from data.text_multi_motion import TextMultiMotionDataset
from torch.utils import data
from model_transformer import *
import wandb


class Method(enum.Enum):
    method_1 = 'current_frame'
    method_2 = 'output'

class SkeletonLSTM(nn.Module):
    def __init__(self, device, method: Method=None, hidden_size=32, feature_size=63, name="model_name"):
        super(SkeletonLSTM, self).__init__()

        self.method = method

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
        
        outputs = self.lin4(output) + motion_frame.unsqueeze(1)

        return outputs