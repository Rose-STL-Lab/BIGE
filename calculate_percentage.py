from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from write_mot import write_muscle_activations
import torch.optim as optim
import sys
from glob import glob
import numpy as np
import pandas as pd

class TransformerLayer(nn.Module):
    def __init__(self, timestep_vector_dim: int, num_heads: int, dim_feedforward: int, dropout: float):
        super(TransformerLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=timestep_vector_dim, num_heads=num_heads
        )
        self.feedforward = nn.Sequential(
            nn.Linear(timestep_vector_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, timestep_vector_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(timestep_vector_dim)
        self.norm2 = nn.LayerNorm(timestep_vector_dim)

    def forward(self, x: torch.Tensor):
        # Multihead self-attention
        x = x.float()
        attn_output, _ = self.multihead_attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # Feedforward neural network
        ff_output = self.feedforward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)

        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, num_heads: int, dim_feedforward: int, dropout: float):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 196, input_dim))

        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(input_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Fully Connected Output Layer
        self.fc = nn.Linear(input_dim, output_dim)

        # Sigmoid for output normalization
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # Add positional encoding
        x = x.float()
        x = x + self.positional_encoding

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Generate output predictions
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        
        print(f"Model loaded from {path}")
    
    def calculate_loss(self, predictions, targets):
        return F.smooth_l1_loss(predictions, targets)

if __name__ == "__main__":
    window_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model = TransformerModel(input_dim=33, output_dim=80, num_layers=3, num_heads=3, dim_feedforward=128, dropout=0.1).to(device)
    
    save_path = "transformer_surrogate_model_v2.pth"
    
    model.load_model(save_path)
    
    root_dir = "output_GPT_Final/"
    folders = ["HighSurrogateFootSlding", "LowSurrogateFootSlding", "MidSurrogateFootSlding"]
    
    low = [0.15,0.25]
    mid = [0.25,0.35]
    high = [0.35,0.45]
    
    for folder in tqdm(folders):
        result_dict = {"path":[], "low":[], "mid":[], "high":[]}
        
        root_path = root_dir + folder + "/" + "mot_visualization/*/*.mot"
        files = glob(root_path)
        
        for file in files:
            with open(file,'r') as f:
                file_data = f.read().split('\n')
                # print(file_data)
                data = {'info':'', 'poses': []}
                read_header = False
                read_rows = 0
                
                for line in file_data:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    
                    if not read_header:
                        if line == 'endheader':
                            read_header = True
                            continue
                        if '=' not in line:
                            data['info'] += line + '\n'
                        else:
                            k,v = line.split('=')
                            if v.isnumeric():
                                data[k] = int(v)
                            else:
                                data[k] = v
                    else:
                        rows = line.split()
                        if read_rows == 0:
                            data['headers'] = rows
                        else:
                            rows = [float(row) for row in rows]
                            data['poses'].append(rows)

                        read_rows += 1
            motion = np.array(data['poses'])[:,1:]
            output = model(torch.tensor(motion).to(device))
            output = output.squeeze(0).cpu().detach().numpy()
            
            idx_to_keep = [-1,-2,-3,-7,-41,-42,-43,-47]
            out = output[:,idx_to_keep]
            # print(out.shape)
            
            low_count = []
            mid_count = []
            high_count = []
            
            for i in range(out.shape[1]):
                feature = out[:,i]
                feature = feature.astype(float)
                
                low_c = np.sum((feature >= low[0]) & (feature < low[1]))
                mid_c = np.sum((feature >= mid[0]) & (feature < mid[1]))
                high_c = np.sum((feature >= high[0]) & (feature < high[1]))
                
                total_c = len(feature)
                
                low_count.append((low_c/total_c)*100)
                mid_count.append((mid_c/total_c)*100)
                high_count.append((high_c/total_c)*100)
            
            low_count = np.array(low_count)
            mid_count = np.array(mid_count)
            high_count = np.array(high_count)
            
            # print(low_count.shape)
            # print(low_count)
            result_dict['path'].append(file)
            result_dict['low'].append(low_count)
            result_dict['mid'].append(mid_count)
            result_dict['high'].append(high_count)
            
        df = pd.DataFrame(result_dict)
        df.to_csv(folder + ".csv", index = False)