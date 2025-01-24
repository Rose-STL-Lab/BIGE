from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dataset import dataset_MOT_segmented_surrogate
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
    
# Hyperparameters
input_dim = 33  # Number of input features per timestep
output_dim = 80  # Number of output features per timestep
batch_size = 40
window_size = 64

train_loader = dataset_MOT_segmented_surrogate.DATALoader("mcs",
                                        batch_size,
                                        window_size=window_size,
                                        unit_length=4)

test_loader = dataset_MOT_segmented_surrogate.DATALoader("mcs",
                                        batch_size,
                                        window_size=window_size,
                                        unit_length=4,
                                        mode='test')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_dim=33, output_dim=80, num_layers=3, num_heads=3, dim_feedforward=128, dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)
# criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
criterion = nn.L1Loss()  # Mean Squared Error for regression tasks

# Training loop
def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    train_squat_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for motion, len_motion, activations, name in pbar:
        # Move data to the appropriate device
        motion = motion.to(device).float()
        activations = activations.to(device).float()
        
        # Forward pass
        predictions = model(motion.float()).float()


        # Compute loss
        loss = criterion(predictions, activations)
        squat_loss = criterion(predictions[:, :, -6:], activations[:, :, -6:])
        loss += 100*squat_loss

        # # Create mask for the motion length
        # motion_start, motion_end = len_motion[0].to(device), len_motion[1].to(device)
        # # Create mask based on motion length
        # mask = torch.arange(motion.size(1), device=device).expand(motion_end.size(0), motion.size(1)) < motion_end.unsqueeze(1)
        # mask = mask.unsqueeze(-1).expand_as(activations).float()

        # # Apply mask
        # masked_predictions = predictions * mask
        # masked_activations = activations * mask
        
        # # Compute loss
        # loss = criterion(masked_predictions, masked_activations)
        # squat_loss = criterion(masked_predictions[:, :, -6:], masked_activations[:, :, -6:])
        # loss += 100 * squat_loss



        
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress
        train_loss += loss.item()
        train_squat_loss += squat_loss.item()
        pbar.set_postfix(loss=loss.item(), squat_loss=squat_loss.item())
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_squat_loss = train_squat_loss / len(train_loader)
    print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f} Squat Loss: {avg_train_squat_loss:.4f}")
    return avg_train_loss, avg_train_squat_loss

# Testing loop
def test_model(model, test_loader, criterion):
    
    model.eval()
    test_loss = 0
    test_squat_loss = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for motion, len_motion, activations, name in pbar:
            # Move data to the appropriate device
            motion = motion.to(device).float()
            activations = activations.to(device).float()
            
            # Forward pass
            predictions = model(motion.float())
        
            # Compute loss
            loss = criterion(predictions.float(), activations)
            
            squat_loss = criterion(predictions[:, :, -6:], activations[:, :, -6:])

            
            # Update progress
            test_loss += loss.item()
            test_squat_loss += squat_loss.item()
            pbar.set_postfix(loss=loss.item(), squat_loss=squat_loss.item())

    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_squat_loss = test_squat_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f} Squat Loss: {avg_test_squat_loss:.4f}")
    return avg_test_loss, avg_test_squat_loss

# Save path for the model
save_path = "transformer_surrogate_model_physics_consistency.pth"

# Initialize the best test accuracy
best_test_accuracy = float("inf")

# Main loop
num_epochs = 1000
# num_epochs = 1
for epoch in range(1, num_epochs + 1):
    train_loss, train_squat_loss = train_model(model, train_loader, optimizer, criterion, epoch)

    if epoch % 10 == 0:
        print(f"Running Testing at Epoch {epoch}")
        test_loss, test_squat_loss = test_model(model, test_loader, criterion)

        # Save the model if test accuracy improves
        if test_loss < best_test_accuracy:
            best_test_accuracy = test_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with Test Loss: {test_loss:.4f} Squat Loss: {test_squat_loss:.4f}")
    
final_test_loader = dataset_MOT_segmented_surrogate.DATALoader(
    "mcs",
    batch_size=1,  # Process one sample at a time for final predictions
    window_size=window_size,
    unit_length=4,
    mode="test"
)

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

exp_name = sys.argv[1] if len(sys.argv) > 1 else "transformer_surrogate_linear_momentum"

# Save predictions
save_dir = os.path.join(final_test_loader.dataset.data_dir, f"{exp_name}_activations")
os.makedirs(save_dir, exist_ok=True)

for name in collate_predictions:
    session_id = name.split("/")[-5]
    trial = name.split("/")[-2]
    act_name = f"{session_id}-{trial}.mot"

    save_path = os.path.join(save_dir, act_name)
    print(f"Saving to {save_path}")
    write_muscle_activations(save_path, collate_predictions[name].numpy())
    
    
final_train_loader = dataset_MOT_segmented_surrogate.DATALoader(
    "mcs",
    batch_size=1,  # Process one sample at a time for final predictions
    window_size=window_size,
    unit_length=4,
    mode="train"
)

# Prepare to store predictions
collate_predictions = {}
OUT_D = 80  # Ensure this matches your model's output dimension
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Process and save predictions
for inputs, lengths, _, name in tqdm(final_train_loader, desc="Generating Reconstructions"):
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

# Save predictions
save_dir = os.path.join(final_test_loader.dataset.data_dir, f"{exp_name}_activations")
os.makedirs(save_dir, exist_ok=True)

for name in collate_predictions:
    session_id = name.split("/")[-5]
    trial = name.split("/")[-2]
    act_name = f"{session_id}-{trial}.mot"

    save_path = os.path.join(save_dir, act_name)
    print(f"Saving to {save_path}")
    write_muscle_activations(save_path, collate_predictions[name].numpy())
    
    
    
    
    


import os
reporter_path = os.path.join(os.getcwd(), "..", "UCSD-OpenCap-Fitness-Dataset")
os.chdir(reporter_path)
report_name = os.path.join(os.getcwd(), "MCS-Surrogate.pdf")

print(f"Running command: python src/surrogate/create_report.py --surrogates transformer_surrogate_v3_activations {save_dir} --name {report_name}")
os.system(f"python src/surrogate/create_report.py --surrogates transformer_surrogate_v3_activations {save_dir} --name {report_name} > report.log")
os.system(f"cat report.log | tail -n 40")