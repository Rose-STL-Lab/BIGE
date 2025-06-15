import numpy as np
import os
import argparse
from dataset import dataset_MOT_MCS, dataset_MOT_segmented
from tqdm import tqdm
from glob import glob
from scipy.stats import entropy, shapiro, kstest, anderson

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

        print(glob_files)
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
                generated_data_list.append(data['poses'])
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
    
    
    ISNormal = {}
    # KS Test 
    ISNormal['KS'] = {}
    ISNormal['KS']['result'] = True
    ISNormal['KS']['params'] = {}
    ISNormal['KS']['params']['statistic'], ISNormal['KS']['params']['pvalue'] = kstest(data.reshape(-1), 'norm', args=(mean, std))
    if ISNormal['KS']['params']['pvalue'] < 0.05:
        ISNormal['KS']['result'] = False
    
    
    # Shapiro-Wilk Normality Test
    ISNormal['SW'] = {}
    ISNormal['SW']['result'] = True
    ISNormal['SW']['params'] = {}
    ISNormal['SW']['params']['statistic'], ISNormal['SW']['params']['pvalue'] = shapiro(data)
    if ISNormal['SW']['params']['pvalue'] < 0.05:
        ISNormal['SW']['result'] = False
    
    return mean, std, ISNormal

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
    mean_real, std_real, IsNormal = aggregate_mean_and_variance(real_data)
    
    for test in IsNormal:
        if IsNormal[test]['result'] == False:
            print(f"[{test} failed]:Real data is not normal. Params: {IsNormal[test]['params']}")

    
    mean_generated, std_generated, IsNormal = aggregate_mean_and_variance(generated_data)
    for test in IsNormal:
        if IsNormal[test]['result'] == False:
            print(f"[{test} failed]:Generated data is not normal. Params: {IsNormal[test]['params']}")



    print("Mean of real data:", mean_real, " Std dev of real data:", std_real)
    print("Mean of generated data:", mean_generated, " Std dev of generated data:", std_generated)

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
def main(file_type, folder_path,baseline):
    # Load real training data
    real_data = load_training_data()
    print("Shape of training data:", real_data.shape)
    
    if file_type == 'npy':
        all_folders = [z[0] for z in os.walk(folder_path)][1:]
    elif file_type == 'mot':
        all_folders = [folder_path + name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    
    entropies = []
    wasserstein = []
    all_folders = sorted(all_folders)
    print(all_folders)
    
    
    for folder in tqdm(all_folders):
        # print("Going through folder:", folder)
        # Load generated data from .npy files
        generated_data = load_generated_data(file_type, folder,baseline=baseline)
        if generated_data is None:
            print("Empty folder")
            continue
        print("Shape of generated data:", generated_data.shape)

        # Flatten both real and generated data
        real_data_flattened = flatten_data(real_data)
        generated_data_flattened = flatten_data(generated_data)
        print("Shape of flattened training data:", real_data_flattened.shape)
        print("Shape of flattened generated data:", generated_data_flattened.shape)

        # Calculate the 2-Wasserstein distance
        wasserstein_dist = wasserstein_distance_mean_variance(real_data_flattened, generated_data_flattened)
        
        # Calculate the entropy difference
        entropy_diff = entropy_difference(real_data_flattened, generated_data_flattened)
        print(f"Entropy Difference: {entropy_diff}")
        print(f"2-Wasserstein Distance: {wasserstein_dist}")
        entropies.append(entropy_diff)
        wasserstein.append(wasserstein_dist)
    
    print("Mean of wasserstein metric:", np.mean(wasserstein)," Std dev of wasserstein distance:", np.std(wasserstein))
    print("Mean of entropy:", np.mean(entropies), " Std dev of entropy:", np.std(entropies))
    
    # Compute means and standard deviations
    mean_wasserstein = np.mean(wasserstein)
    std_wasserstein = np.std(wasserstein)
    mean_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)

    
    
    print(" - & ${:.2f}^{{\pm{:.2f}}}$ & ${:.2f}^{{\pm{:.2f}}}$ \\\\".format(mean_wasserstein, std_wasserstein, mean_entropy, std_entropy))    

# Step 7: Argument parser to pass folder path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute 2-Wasserstein Distance between real and generated data.")
    parser.add_argument("--file_type", type=str)
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing .npy files of generated data")
    parser.add_argument("--baseline", type=str, default='bige' , help="Path to the folder containing .npy files of generated data")
    args = parser.parse_args()

    # Run the main function with the provided folder path
    main(args.file_type, args.folder_path, args.baseline)
