import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SkeletonFormer(nn.Module):
    def __init__(self, text_encoder, hidden_size=32, feature_size=63, name="model_name", method=None):
        super(SkeletonFormer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.name = name
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.period = 2000
        self.device = device
        self.method = method

        self.text_encoder = text_encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.lin_text = nn.Linear(self.text_encoder.out_dim, int(self.hidden_size))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        # motion encoder
        self.motion_encoder = nn.Linear(feature_size, self.hidden_size)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(self.hidden_size, period = self.period)
        # temporal bias
        
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 2000, period=self.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=4, dim_feedforward=2*self.hidden_size, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # motion decoder
        self.motion_decoder = nn.Linear(self.hidden_size, self.feature_size)

        nn.init.constant_(self.motion_decoder.weight, 0)
        nn.init.constant_(self.motion_decoder.bias, 0)

        self.save_path = f"{os.getcwd()}/checkpoints/{name}.ckpt" 

    def forward(self, motions, texts):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        batch_size, seq_length, _ = motions.shape

        text_embedding = self.text_encoder(texts) 

        text_embedding = self.lin_text(text_embedding).unsqueeze(1).expand(batch_size, seq_length, self.hidden_size) # (bs, hideen_size/2)
        
        text_embedding = self.transformer_encoder(self.PPE(text_embedding))

        motions_emb = self.PPE(self.motion_encoder(motions))
        
        tgt_mask = self.biased_mask[:, :motions.shape[1], :motions.shape[1]].clone().detach().to(device=self.device).unsqueeze(0).expand(batch_size, 4, seq_length, seq_length).reshape(batch_size*4, seq_length, seq_length) 
        memory_mask = enc_dec_mask(self.device, motions.shape[1], text_embedding.shape[1])
        motion_out = self.transformer_decoder(motions_emb, text_embedding, tgt_mask=tgt_mask, memory_mask=memory_mask)
        outputs = self.motion_decoder(motion_out)# + motions[:, 0, :].unsqueeze(1)
            
        return outputs
    
    def predict(self, motions, texts):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        batch_size, seq_length, _ = motions.shape
        
        text_embedding = self.text_encoder(texts)
        
        text_embedding = self.lin_text(text_embedding).unsqueeze(1).expand(batch_size, seq_length, self.hidden_size) # (bs, hideen_size/2)
        outputs = []

        text_embedding = self.transformer_encoder(self.PPE(text_embedding))
        
        motion_frame_disp = motions[:, 1, :] - motions[:, 0, :]# primo frame (bs, 63)
        motion_frame = motions[:, 0, :] # primo frame (bs, 63)
        
        for i in range(seq_length):
            if i==0:
                motion_emb = self.motion_encoder(motion_frame_disp).unsqueeze(1)
                motion_input = self.PPE(motion_emb)
            else:
                motion_input = self.PPE(motion_emb)
            tgt_mask = self.biased_mask[:, :motion_input.shape[1], :motion_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, motion_input.shape[1], text_embedding.shape[1])
            motion_out = self.transformer_decoder(motion_input, text_embedding, tgt_mask=tgt_mask, memory_mask=memory_mask)
            motion_out = self.motion_decoder(motion_out) #+ motion_frame_
            motion_disp = motion_out[:,-1,:].unsqueeze(1)
            motion_frame = motion_disp + motion_frame
            outputs.append(motion_frame)
            new_output = self.motion_encoder(motion_disp)#.unsqueeze(1)
            motion_emb = torch.cat((motion_emb, new_output), 1)

        outputs = torch.stack(outputs, dim=1).squeeze(2)
        
        return outputs