from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from write_mot import write_muscle_activations
import torch.optim as optim

import sys

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
        
    # Example usage
    window_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model = TransformerModel(input_dim=33, output_dim=80, num_layers=3, num_heads=3, dim_feedforward=128, dropout=0.1).to(device)
    
    # Save path for the model
    save_path = "transformer_surrogate_model_v2.pth"
    
    assert os.path.exists(save_path), f"Model not found at {save_path}" 
    
    model.load_model(save_path)
    
    from dataset import dataset_MOT_segmented_surrogate
        
    final_test_loader = dataset_MOT_segmented_surrogate.DATALoader(
    "mcs",
    batch_size=1,  # Process one sample at a time for final predictions
    window_size=window_size,
    unit_length=4,
    mode="test")
    
    exp_name = "testing_surrogate"

    # Save predictions
    save_dir = os.path.join(final_test_loader.dataset.data_dir, f"{exp_name}_activations")
    os.makedirs(save_dir, exist_ok=True)



    # Prepare to store predictions
    collate_predictions = {}
    OUT_D = 80  # Ensure this matches your model's output dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Process and save predictions
    for inputs, lengths, _, name in tqdm(final_test_loader, desc="Generating Reconstructions"):
        # try:
            # Prepare inputs
        # print(lengths, name)
        inputs = inputs.float().to(device)  # Already a single sample
        # lengths = lengths[0]  # Extract [start, end] from the list
        start, end = int(lengths[0]), int(lengths[1])
        name = name[0]  # Get the name of the file

        # Model inference
        outputs = model(inputs)  # Shape: (1, seq_len, OUT_D)
        outputs = outputs.squeeze(0).detach().cpu()  # Remove batch dimension

        # Determine motion length and initialize prediction storage
        motion_length = end
        if name not in collate_predictions:
            collate_predictions[name] = torch.zeros((motion_length, OUT_D))

        # Store predictions
        block_sz = min(end - start, outputs.shape[0])
        collate_predictions[name][start : start + block_sz] = outputs[:block_sz]

        # except Exception as e:
        #     print(f"Error processing final output for {name}: {e}")


    for name in collate_predictions:
        session_id = name.split("/")[-5]
        trial = name.split("/")[-2]
        act_name = f"{session_id}-{trial}.mot"

        save_path = os.path.join(save_dir, act_name)
        print(f"Saving to {save_path}")
        write_muscle_activations(save_path, collate_predictions[name].numpy())