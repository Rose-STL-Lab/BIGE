import torch
import torch.nn as nn
import nimblephysics as nimble 

from osim_sequence import load_osim, groundConstraint, GetLowestPointLayer


class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, mode='together'):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        self.mode = mode
        
    def forward(self, motion_pred, motion_gt) : 
        # loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        if self.mode == 'together':
            loss = self.Loss(motion_pred, motion_gt)
            return loss
        elif self.mode == 'separate':
            rot_indices = list(range(0, 33))
            for idx in [3, 4, 5]:
                rot_indices.remove(idx)  # Exclude indices 3, 4, 5 for rotational

            trans_indices = [3, 4, 5]  # Translational components

            # Extract rotational and translational components
            rot_pred = motion_pred[:, :, rot_indices]  # Rotational components from prediction
            rot_gt = motion_gt[:, :, rot_indices]      # Rotational components from ground truth

            trans_pred = motion_pred[:, :, trans_indices]  # Translational components from prediction
            trans_gt = motion_gt[:, :, trans_indices]      # Translational components from ground truth

            # Compute separate losses
            rot_loss = self.Loss(rot_pred, rot_gt)
            trans_loss = self.Loss(trans_pred, trans_gt)

            # Apply different weights
            rot_weight = 0.7  # Weight for rotational loss
            trans_weight = 20  # Weight for translational loss

            # Combine weighted losses
            loss = rot_weight * rot_loss + trans_weight * trans_loss
            return loss, rot_loss, trans_loss
    
    def forward_vel(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss
    
    
    def penetration_loss(self, motion_pred, osim, num_samples=1, device=torch.device('cuda')):
        
        
        batch_size = motion_pred.shape[0]
        
        foot_loss = torch.tensor([0.0],device=device)
        
        j_list = torch.randint(0, motion_pred.shape[1]-1, (batch_size, num_samples))
        
        for i in range(motion_pred.shape[0]):
            # m = motion[i]
            # m_tensor = torch.tensor(m, dtype=torch.float32, device=device, requires_grad=True)
            
            # for j in range(1,motion_pred.shape[1],3):
            for rand_j in range(num_samples):
                
                j = j_list[i,rand_j]
                
                nimble_input_motion = motion_pred[i,j].clone()
                nimble_input_motion[6:] = torch.deg2rad(nimble_input_motion[6:])
                nimble_input_motion[:3] = 0 # Set pelvis rotation to 0
                nimble_input_motion = nimble_input_motion.cpu()
                x = GetLowestPointLayer.apply(osim.skeleton, nimble_input_motion).to(device)
                foot_loss += x**2
                
        return foot_loss / (batch_size * num_samples)
    


# How to load osim file: 
# assert os.path.isdir(args.subject), "Location to subject info does not exist"
# osim_path = os.path.join(args.subject,'OpenSimData','Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim')

# assert os.path.exists(osim_path), f"Osim file:{osim_path} does not exist"
# osim_geometry_dir = os.path.join("/data/panini/MCS_DATA",'OpenCap_LaiArnoldModified2017_Geometry')

# assert os.path.exists(osim_geometry_dir), f"Osim geometry path:{osim_geometry_dir} does not exist"from osim_sequence import load_osim, groundConstraint, GetLowestPointLayer
# assert os.path.isdir(args.subject), "Location to subject info does not exist"
# osim_path = os.path.join(args.subject,'OpenSimData','Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim')

# assert os.path.exists(osim_path), f"Osim file:{osim_path} does not exist"
# osim_geometry_dir = os.path.join("/data/panini/MCS_DATA",'OpenCap_LaiArnoldModified2017_Geometry')
# assert os.path.exists(osim_geometry_dir), f"Osim geometry path:{osim_geometry_dir} does not exist"