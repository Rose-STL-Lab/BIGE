import numpy as np
import os
import argparse
from dataset import dataset_MOT_MCS, dataset_MOT_segmented
from tqdm import tqdm
from glob import glob
from scipy.stats import entropy
import torch


from guidance import get_guidance_score


# Step 1: Load training data (assuming it's already loaded or available)
def load_training_data():
    """
    Load your real training data. For this example, I'm generating random data.
    You should replace this with actual code to load the real data.

    Returns:
    - training_data (numpy array): real training data
    """
    real_data = []
    data_loader = dataset_MOT_segmented.DATALoader(
        dataset_name='mcs',
        batch_size=1,
        window_size=64,
        mode='limo'
    )
    data_loader.dataset.mode2 = 'metrics'
    for i,batch in tqdm(enumerate(data_loader)):
        motion, _,_ = batch
        real_data.extend(motion)
    
    real_data = np.array([np.array(x) for x in real_data])
    real_data = np.squeeze(real_data, axis=1)
    return real_data

def load_mocap_data(file_type, folder_path):
    """
    Args:
    - folder_path (str): path to the folder containing files

    Returns:
    - generated_data (numpy array): combined generated data
    """
    generated_data_list = []
    
    
    # path = "/home/ubuntu/data/MCS_DATA/Data/*/OpenSimData/Dynamics/*_segment_*/kinematics_activations_*_muscle_driven.mot"
    # path = "/home/ubuntu/data/MCS_DATA/Data/*/OpenSimData/Kinematics/*.mot"
    path = folder_path + "/OpenSimData/Kinematics/*.mot"
    if len(glob(path)) == 0:
        return None

    mot_files = [file for file in glob(path) if file.endswith(".mot") and ("sqt" in file or "SQT" in file or "Sqt" in file) and ("segment" not in file)]
    # print("Loading data from:", mot_files)
    for file in glob(path):
        if file.endswith(".mot") and ("sqt" in file or "SQT" in file or "Sqt" in file) and ("segment" not in file):
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
                        
                            
            data['poses'] = np.array(data['poses']) #[:,1:34]
            current_format = data['headers']
            required_format = ["pelvis_tilt","pelvis_list","pelvis_rotation","pelvis_tx","pelvis_ty","pelvis_tz","hip_flexion_l","hip_adduction_l","hip_rotation_l","hip_flexion_r","hip_adduction_r","hip_rotation_r","knee_angle_l","knee_angle_r","ankle_angle_l","ankle_angle_r","subtalar_angle_l","subtalar_angle_r","mtp_angle_l","mtp_angle_r","lumbar_extension","lumbar_bending","lumbar_rotation","arm_flex_l","arm_add_l","arm_rot_l","arm_flex_r","arm_add_r","arm_rot_r","elbow_flex_l","elbow_flex_r","pro_sup_l","pro_sup_r"]

            mapping_indices = [current_format.index(name) for name in required_format]
            # print(mapping_indices)
            data['poses'] = data['poses'][:,mapping_indices]

            segmentation_file = os.path.join(os.path.dirname(os.path.dirname(folder_path)), "squat-segmentation-data", os.path.basename(folder_path) + ".npy")
            print(segmentation_file)
            segments = np.load(segmentation_file,allow_pickle=True).item()
            # print(segments)
            print("Number of segments:", len(segments))
            if os.path.basename(file.replace(".mot","")) not in segments:
                continue
            segments = segments[os.path.basename(file.replace(".mot",""))]

            for segment in segments:
                poses = data['poses'][segment[0]:segment[1]]
                if poses.shape[0] < 196:
                    rc = (196+poses.shape[0]-1)//poses.shape[0]
                    poses = np.tile(poses, (rc,1))[:196]
                generated_data_list.append(poses)
    generated_data = np.array(generated_data_list)
    # print("Shape of generated data:", generated_data.shape)
    return generated_data




# Step 2: Load generated data from .npy files
def load_generated_data(file_type, folder_path, baseline='bige'):
    """
    Load generated data from .npy files in the specified folder.
    Args:
    - folder_path (str): path to the folder containing .npy files

    Returns:
    - generated_data (numpy array): combined generated data
    """
    generated_data_list = []
    
    if file_type == 'npy':
    
        # for file in os.listdir(folder_path):
        print(len(glob(folder_path + "/*.npy")))
        for file in glob(folder_path + "/*.npy"):
            if file.endswith(".npy"):
                # file_path = os.path.join(folder_path, file)
                # print(file)
                data = np.load(file)
                if data.shape[0]!=196:
                    continue
                generated_data_list.append(data)
        
        # Combine all the loaded generated data into a single numpy array
        # generated_data = np.concatenate(generated_data_list, axis=0)
        generated_data = np.array(generated_data_list)
    elif file_type == 'mot':
        
        glob_files = glob(folder_path + "/*.mot")

        if baseline == 't2m':
            glob_files = [ file for file in glob_files if "degrees" in file ] 

        # print(glob_files) 
        if len(glob_files) == 0:
            return None
        for file in glob(folder_path + "/*.mot"):
            if file.endswith(".mot"):
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
                data['poses'] = np.array(data['poses'])[:,1:34]
                if data["poses"].shape[0] < 196:
                    rc = (196+data["poses"].shape[0]-1)//data["poses"].shape[0]
                    data["poses"] = np.tile(data["poses"], (rc,1))[:196]
                generated_data_list.append(data["poses"])
        generated_data = np.array(generated_data_list)
    return generated_data


# Step 3: Flatten the data into a single dimension
def flatten_data(data):
    """
    Flatten the time series data into a single dimension.
    Args:
    - data (numpy array): shape (n_samples, n_timesteps, n_features)

    Returns:
    - flattened_data (numpy array): shape (n_samples, )
    """
    return data.reshape(data.shape[0], -1)

# Step 4: Compute mean and variance
def aggregate_mean_and_variance(data):
    """
    Aggregates flattened data by computing the mean and variance.
    Args:
    - data (numpy array): flattened data

    Returns:
    - mean (float): mean of aggregated data
    - std (float): standard deviation of aggregated data
    """
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

# Step 5: Compute the 2-Wasserstein distance
def wasserstein_distance_mean_variance(real_data, generated_data):
    """
    Calculate the 2-Wasserstein distance between two datasets using mean and variance.
    Args:
    - real_data (numpy array): real flattened data
    - generated_data (numpy array): generated flattened data

    Returns:
    - wasserstein_distance (float): the 2-Wasserstein distance
    """
    # Aggregate data by computing mean and variance
    mean_real, std_real = aggregate_mean_and_variance(real_data)
    mean_generated, std_generated = aggregate_mean_and_variance(generated_data)

    # Compute mean and variance terms
    mean_diff_squared = (mean_real - mean_generated) ** 2
    std_diff_squared = (std_real - std_generated) ** 2

    # Compute the 2-Wasserstein distance using the formula
    wasserstein_distance = mean_diff_squared + std_diff_squared
    return wasserstein_distance

def calculate_entropy(data, num_bins=10):
    """
    Calculate Shannon entropy for a dataset.
    
    Args:
    - data (numpy array): flattened data array (each row is a sample)
    - num_bins (int): number of bins to discretize the data into
    
    Returns:
    - entropies (numpy array): array of entropies for each sample
    """
    entropies = []
    for sample in data:
        # Create a histogram (binning) for the data
        hist, bin_edges = np.histogram(sample, bins=num_bins, density=True)
        # Calculate the entropy (adding epsilon to avoid log(0))
        sample_entropy = entropy(hist + np.finfo(float).eps)
        entropies.append(sample_entropy)
    
    return np.array(entropies)

def entropy_difference(real_data, generated_data, num_bins=10):
    """
    Compute the difference in entropy between real and generated data.
    
    Args:
    - real_data (numpy array): real flattened data
    - generated_data (numpy array): generated flattened data
    - num_bins (int): number of bins to discretize the data into
    
    Returns:
    - entropy_diff (float): absolute difference in entropy between real and generated data
    """
    # Calculate entropy for real and generated data
    real_entropy = calculate_entropy(real_data, num_bins=num_bins)
    # print("Training data entropy:", real_entropy.mean())
    generated_entropy = calculate_entropy(generated_data, num_bins=num_bins)
    print("Generated data entropy:", generated_entropy.mean())
    
    # Calculate the absolute difference between the entropies
    # entropy_diff = np.abs(real_entropy.mean() - generated_entropy.mean())
    entropy_diff = np.abs(generated_entropy.mean())
    
    return entropy_diff


# Step 6: Main script function
def main(file_type, folder_path,baseline='bige'):
 
    if file_type == 'npy':
        all_folders = [z[0] for z in os.walk(folder_path)][1:]
    elif file_type == 'mot':
        all_folders = [folder_path + name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    elif file_type == 'mocap':
        mcs_sessions = ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]
        all_folders = [os.path.join(folder_path,session) for session in mcs_sessions]
    all_folders = sorted(all_folders)
    print(all_folders)
    
    
    # Load test data 
    real_data = load_training_data()
    pred_motion = torch.from_numpy(real_data).float()
    real_data_loss_dict, real_data_muscle_activations = get_guidance_score(pred_motion)    
    real_data_loss_dict = {k: real_data_loss_dict[k].item() for k in real_data_loss_dict if isinstance(real_data_loss_dict[k], torch.Tensor)}    
    
    
    losses_dict = {}
    
    for folder in tqdm(all_folders):
        # print("Going through folder:", folder)
        # Load generated data from .npy files
        if file_type == 'mocap':
            generated_data = load_mocap_data(file_type, folder)
        else:            
            generated_data = load_generated_data(file_type, folder,baseline=baseline)
        if generated_data is None:
            print(f"Empty folder:{folder}")
            continue
        # print("Shape of generated data:", generated_data.shape)
        pred_motion = torch.from_numpy(generated_data).float()

        loss_dict, muscle_activations = get_guidance_score(pred_motion)
        
        loss_dict = {k: loss_dict[k].item() for k in loss_dict if isinstance(loss_dict[k], torch.Tensor)}
        
        # print(muscle_activations)
        
        for k in loss_dict:
            if k not in losses_dict:
                losses_dict[k] = []
            losses_dict[k].append(loss_dict[k])
    
    losses_dict = {k: np.array(losses_dict[k]) for k in losses_dict}
    print("Losses dict:", losses_dict)
    mean_loss = {k: np.mean(losses_dict[k]) for k in losses_dict}
    std_loss = {k: np.std(losses_dict[k]) for k in losses_dict}
    
    
    
    
    
    distribution = {k: (losses_dict[k].mean(), losses_dict[k].std()) for k in losses_dict}
    print("Distribution:", distribution)
    print(" & ".join([ k + "$\\rightarrow$"   for k in  distribution.keys()]))
    
    
    
    print(f"mot sim", end=' &  ')
    for k in real_data_loss_dict:
        print("${:.2f}^{{\pm0.0}}$ ".format(real_data_loss_dict[k]), end=' &  ')

    print(f'\\\\  % {args.folder_path}')
    
    
    print(f"{args.file_type} {args.baseline}", end=' &  ')
    for k in distribution:
        print("${:.2f}^{{\pm{:.2f}}}$ ".format(distribution[k][0], distribution[k][1]), end=' &  ')

    print(f'\\\\  % {args.folder_path}')
    
    

# Step 7: Argument parser to pass folder path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute 2-Wasserstein Distance between real and generated data.")
    parser.add_argument("--file_type", type=str)
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing .npy files of generated data")
    parser.add_argument("--baseline", type=str, default='bige', help="Baseline to use for guidance")
    args = parser.parse_args()

    # Run the main function with the provided folder path
    main(args.file_type, args.folder_path, args.baseline)
