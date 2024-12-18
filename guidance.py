import os
from collections import OrderedDict
import random
import torch 
import torch.nn.functional as F

from dataset import dataset_MOT_segmented
from osim_sequence import GetLowestPointLayer

val_loader = dataset_MOT_segmented.DATALoader('mcs',
                                        1,
                                        window_size=64,
                                        unit_length=2**2,
                                        mode='limo')

    
# squat_muscles_indices = [val_loader.dataset.headers2indices[k] for k in val_loader.dataset.headers2indices if 'vaslat' in k or 'vasmed' in k] # Thigh muscles, (left and right) Vasus lateralis, medialis, intermedius 
squat_muscles_indices = [val_loader.dataset.headers2indices[k] for k in val_loader.dataset.headers2indices if 'vasmed' in k] # Thigh muscles, (left and right) Vasus lateralis, medialis, intermedius 
print("Squat muscles indices:",squat_muscles_indices)
pelvis_tilt_index = val_loader.dataset.headers2indices['pelvis_tilt']
print("Pelvis tilt index:",pelvis_tilt_index)



# Symmetry conditions 
symm_left_indices = ['hip_flexion_l', 'knee_angle_l', 'ankle_angle_l']
symm_left_indices = [val_loader.dataset.headers2indices[k] for k in symm_left_indices]
symm_right_indices = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
symm_right_indices = [val_loader.dataset.headers2indices[k] for k in symm_right_indices]


from surrogate import TransformerModel
window_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
surrogate = TransformerModel(input_dim=33, output_dim=80, num_layers=3, num_heads=3, dim_feedforward=128, dropout=0.1).to(device)

# Save path for the model
save_path = "transformer_surrogate_model_v2.pth"

assert os.path.exists(save_path), f"Model not found at {save_path}" 

surrogate.load_model(save_path)
surrogate.eval()


def constained_optimization(x, low, high):

    # Compute the three expressions
    expr1 = x - high
    expr2 = low - x
    expr3 = torch.zeros_like(x)
    
    # Compute the element-wise maximum of the three expressions
    result = torch.max(torch.max(expr1, expr2), expr3)
    return result        


def get_guidance_score(pred_motion,framerate=60):
    pred_motion = pred_motion.to(device)

    com_acc_laplacian = torch.tensor([[1, -2, 1]], dtype=torch.float32)
    com_acc_laplacian = com_acc_laplacian.view(1, 1, -1)  # Shape: (1, 1, 3)
    com_acc_laplacian = com_acc_laplacian.repeat(3, 1, 1)  # Shape: (3, 1, 3)
    com_acc_laplacian = com_acc_laplacian.to(device)

    # Relevant indices
    relevant_indices = symm_left_indices + symm_right_indices + [pelvis_tilt_index] 
    # loss_temp = torch.tensor([0.0],device=device)
    ang_vel = pred_motion[:,1:,relevant_indices]-pred_motion[:,:-1,relevant_indices]
    ang_vel = ang_vel / (1/framerate)
    loss_temp = torch.mean(ang_vel**2).sqrt()

    # Symmetry loss
    # loss_symm = torch.tensor([0.0],device=device)        
    loss_symm = torch.mean((pred_motion[:,:,symm_left_indices] - pred_motion[:,:,symm_right_indices])**2).sqrt()        


    # Reduce jitter in the translation
    # loss_temp_trans = torch.tensor([0.0],device=device)
    
    ## Test ball trajectory loss_temp_trans == 10
    ## ball_trajectory =  torch.tile(torch.stack([torch.zeros(60),torch.zeros(60), torch.arange(0,1,1/60)*10]).permute(1,0).unsqueeze(0), (20,1,1))
    vel = pred_motion[:,1:,3:6]-pred_motion[:,:-1,3:6]
    vel = vel / (1/framerate)
    vel = vel.norm(dim=2) # Norm
    loss_temp_trans = torch.mean(vel**2).sqrt() # RMSE

    # Test ball trajectory loss_temp_acc == 10 
    # ball_trajectory =  torch.tile(torch.stack([torch.zeros(60),torch.zeros(60), 0.5*(torch.arange(0,1,1/60)**2)*10]).permute(1,0).unsqueeze(0), (20,1,1))

    com_acc = F.conv1d(
        input=pred_motion[:,:,3:6].permute(0,2,1),  # Pad to maintain sequence length
        weight=com_acc_laplacian,
        groups=3).permute(0,2,1)  # Shape: (batch_size, seq_len, 3)
    
    com_acc = com_acc / (1/framerate)**2  # Convert to acceleration
    
    com_acc = com_acc.norm(dim=2)  # Shape: (batch_size, seq_len)
    
    # loss_temp_com = torch.tensor([0.0],device=device) # RMSE
    loss_temp_com = torch.mean(com_acc**2).sqrt() # RMSE  
    # Surrogate model loss
    pred_muscle_activations = surrogate(pred_motion)        
    surrogate_muscle_activation , peak_timestep = torch.max(pred_muscle_activations[:,:,squat_muscles_indices],dim=1)
    
    # constrain_loss = torch.tensor([0.0],device=device)
    constrain_loss = constained_optimization(surrogate_muscle_activation,low=0.35,high=0.45)

    constrain_loss = torch.mean(constrain_loss)


    peak_timestep = peak_timestep[torch.arange(peak_timestep.size(0)),torch.max(surrogate_muscle_activation,dim=1)[1]] 

    # Lumbar extension constraint
    loss_tilt = torch.mean(pred_motion[:,peak_timestep,pelvis_tilt_index])
    # loss_tilt = torch.tensor([0.0],device=device)
    # increase = True
    # if increase:
    #     surrogate_muscle_activation *= -1

    surrogate_muscle_activation = torch.mean(surrogate_muscle_activation,dim=0)
    
    loss_dict = OrderedDict([ ["Pelvis tilt(degs)", loss_tilt], ["Asymmetry(degs)", loss_symm], \
        ["\omega (degs/s)", loss_temp], ["COM vel.(m/s)", loss_temp_trans], ["COM acc.(m/s^2)", loss_temp_com],\
        ["Constrain (0-1)", constrain_loss]])


    # loss_dict = OrderedDict([["proximity", 0.001*loss_proximity], \
    #     ["tilt", 0.001*loss_tilt], ["symmetry", loss_symm], \
    #     ["foot", foot_loss*0.1], ["foot_sliding", 0.1*foot_sliding_loss], \
    #     ["temporal", 0.5*loss_temp], ["temporal_trans", 50*loss_temp_trans], ["com_acc", 100*loss_temp_com],\
    #     ["constrain", constrain_loss]])
    
    
    return loss_dict, surrogate_muscle_activation
