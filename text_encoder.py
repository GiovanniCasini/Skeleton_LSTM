from turtle import forward
from transformers import BertTokenizer, BertModel
from transformers import BartTokenizer, BartModel
import torch.nn as nn
import os
import torch
from torch import Tensor
from typing import Dict, List
import torch.nn.functional as F
import clip


class Bert(nn.Module):
    def __init__(self, device):
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.text_encoder = BertModel.from_pretrained('bert-base-cased')
        self.device = device
        self.out_dim = 768

    @torch.no_grad()
    def forward(self, texts):
         # Embedding del testo BERT
        text_tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        input_ids = text_tokens.input_ids
        mask = text_tokens.attention_mask
        last_hidden_state, text_embedding = self.text_encoder(input_ids=input_ids, attention_mask=mask,return_dict=False) # (bs, 768)
        return text_embedding


class Bart(nn.Module):
    def __init__(self, device):
        super(Bart, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.text_encoder = BartModel.from_pretrained("facebook/bart-large")
        self.device = device
        self.out_dim = 1024

    @torch.no_grad()
    def forward(self, texts):
        text_tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        input_ids = text_tokens.input_ids
        mask = text_tokens.attention_mask
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=mask, return_dict=False)
        last_hidden_state = outputs[0]  # (batch_size, seq_length_text, hidden_size)
        # Use the first token's hidden state as the pooled embedding
        text_embedding = last_hidden_state[:, 0, :] 
        return text_embedding


class CLIP(nn.Module):
    def __init__(self, modelname: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.out_dim = 512

        model, preprocess = clip.load(modelname, device)
        self.tokenizer = clip.tokenize
        self.clip_model = model.eval()

        # Freeze the weights just in case
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> nn.Module:
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Dict:
        device = device if device is not None else self.device
        tokens = self.tokenizer(texts, truncate=True).to(device)
        return self.clip_model.encode_text(tokens).float() # (bs, 512)