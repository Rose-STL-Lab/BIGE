import os
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
import random
from glob import glob

class Athletes183Dataset(data.Dataset):
    def __init__(self, window_size=64, unit_length=4, mode='train', data_dir='/home/mnt/Datasets/183_retargeted'):
        self.window_size = window_size
        self.unit_length = unit_length
        self.data_dir = data_dir
        self.mode = mode
        
        # Get all .npz files in the directory
        self.npz_files = glob(os.path.join(data_dir, '*.npz'))
        if not self.npz_files:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        # Filter out summary files
        self.npz_files = [f for f in self.npz_files if not f.endswith('processing_summary.npz')]
        
        self.motion_data = []
        self.motion_lengths = []
        self.motion_names = []
        self.motion_fps = []
        self.athlete_names = []
        self.movement_types = []
        self.athlete_metadata = {}
        self.athlete_biomech = {}
        
        # Split athletes based on mode
        total_athletes = len(self.npz_files)
        if mode == 'train':
            # Use first 80% for training
            athlete_files = self.npz_files[:int(0.8 * total_athletes)]
        elif mode == 'test':
            # Use last 20% for testing
            athlete_files = self.npz_files[int(0.8 * total_athletes):]
        elif mode == 'all':
            athlete_files = self.npz_files
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        print(f"Loading {len(athlete_files)} athletes in {mode} mode...")
        
        for npz_file in tqdm(athlete_files, desc=f"Loading {mode} data"):
            try:
                self._load_athlete_data(npz_file)
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
                continue
        
        print(f"Total number of motions loaded: {len(self.motion_data)}")
        
    def _load_athlete_data(self, npz_file):
        """Load data for a single athlete from NPZ file"""
        data = np.load(npz_file, allow_pickle=True)
        
        athlete_name = str(data['athlete_name'])
        num_trials = int(data['num_trials'])
        
        # Store athlete metadata if not already stored
        if athlete_name not in self.athlete_metadata:
            self.athlete_metadata[athlete_name] = {
                'source': str(data.get('source', '183_athletes')),
                'has_validation_data': bool(data.get('has_validation_data', False)),
                'reference_markers': list(data.get('reference_markers', [])),
                'marker_mapping': data.get('marker_mapping', {}).item() if 'marker_mapping' in data else {}
            }
        
        # Compute biomechanical features for this athlete
        if athlete_name not in self.athlete_biomech:
            self.athlete_biomech[athlete_name] = self._compute_athlete_biomech_features(data, num_trials)
        
        # Load each trial
        for trial_idx in range(num_trials):
            try:
                motion_key = f'trial_{trial_idx}_motion'
                label_key = f'trial_{trial_idx}_label'
                metadata_key = f'trial_{trial_idx}_metadata'
                
                if motion_key not in data:
                    continue
                    
                motion = data[motion_key]
                label = str(data[label_key]) if label_key in data else f"trial_{trial_idx}"
                metadata = data[metadata_key].item() if metadata_key in data else {}
                
                # Skip if motion is too short or invalid
                if len(motion) < self.window_size:
                    continue
                
                # Extract fps from metadata
                fps = metadata.get('fps', 100)  # Default to 100 fps
                
                self.motion_data.append(motion)
                self.motion_lengths.append(len(motion))
                self.motion_names.append(f"{athlete_name}::{label}")
                self.motion_fps.append(fps)
                self.athlete_names.append(athlete_name)
                self.movement_types.append(label)
                
            except Exception as e:
                print(f"Error loading trial {trial_idx} from {athlete_name}: {e}")
                continue
        
        data.close()
    
    def _compute_athlete_biomech_features(self, data, num_trials):
        """Compute biomechanical features for an athlete across all trials"""
        all_positions = []
        all_velocities = []
        all_accelerations = []
        
        for trial_idx in range(num_trials):
            motion_key = f'trial_{trial_idx}_motion'
            if motion_key in data:
                motion = data[motion_key]
                if len(motion) > 1:
                    # Compute velocities and accelerations
                    velocities = np.diff(motion, axis=0)
                    accelerations = np.diff(velocities, axis=0)
                    
                    all_positions.append(motion)
                    all_velocities.append(velocities)
                    all_accelerations.append(accelerations)
        
        if not all_positions:
            return self._get_default_biomech_features()
        
        # Concatenate all data
        positions = np.concatenate(all_positions, axis=0)
        velocities = np.concatenate(all_velocities, axis=0) if all_velocities else np.array([])
        accelerations = np.concatenate(all_accelerations, axis=0) if all_accelerations else np.array([])
        
        # Compute statistics
        biomech = {
            "mean_marker_velocities": float(np.mean(np.linalg.norm(velocities, axis=-1))) if len(velocities) > 0 else 0.0,
            "max_marker_velocities": float(np.max(np.linalg.norm(velocities, axis=-1))) if len(velocities) > 0 else 0.0,
            "mean_marker_accelerations": float(np.mean(np.linalg.norm(accelerations, axis=-1))) if len(accelerations) > 0 else 0.0,
            "max_marker_accelerations": float(np.max(np.linalg.norm(accelerations, axis=-1))) if len(accelerations) > 0 else 0.0,
            "mean_marker_pos": float(np.mean(positions)),
            "var_marker_pos": float(np.var(positions)),
            "marker_range": float(np.max(positions) - np.min(positions)),
            "movement_volume": float(np.prod(np.max(positions, axis=0) - np.min(positions, axis=0))),
        }
        
        return biomech
    
    def _get_default_biomech_features(self):
        """Return default biomechanical features when computation fails"""
        return {
            "mean_marker_velocities": 0.0,
            "max_marker_velocities": 0.0,
            "mean_marker_accelerations": 0.0,
            "max_marker_accelerations": 0.0,
            "mean_marker_pos": 0.0,
            "var_marker_pos": 0.0,
            "marker_range": 0.0,
            "movement_volume": 0.0,
        }
    
    def __len__(self):
        return len(self.motion_data)
    
    def __getitem__(self, item):
        motion = self.motion_data[item]
        len_motion = min(len(motion), self.window_size)
        name = self.motion_names[item]
        athlete_name = self.athlete_names[item]
        movement_type = self.movement_types[item]
        biomech = self.athlete_biomech[athlete_name]
        
        # Crop or pad to window_size
        if len(motion) >= self.window_size:
            # Randomly sample a window
            idx = random.randint(0, len(motion) - self.window_size)
            motion = motion[idx:idx + self.window_size]
        else:
            # Repeat motion to fill window
            repeat_count = (self.window_size + len(motion) - 1) // len(motion)
            motion = np.tile(motion, (repeat_count, 1, 1))[:self.window_size]
        
        # Convert to tensor
        motion = torch.from_numpy(motion).float()
        
        return motion, len_motion, name, athlete_name, biomech, movement_type
    
    def get_movement_types(self):
        """Get all unique movement types in the dataset"""
        return list(set(self.movement_types))
    
    def get_athlete_names(self):
        """Get all unique athlete names in the dataset"""
        return list(set(self.athlete_names))
    
    def compute_sampling_prob(self):
        """Compute sampling probabilities for balanced training"""
        movement_counts = {}
        for movement_type in self.movement_types:
            movement_counts[movement_type] = movement_counts.get(movement_type, 0) + 1
        
        # Inverse frequency weighting
        total_samples = len(self.movement_types)
        weights = []
        for movement_type in self.movement_types:
            weight = total_samples / (len(movement_counts) * movement_counts[movement_type])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)


def athletes183_data_loader(window_size=64, unit_length=4, batch_size=1, num_workers=4, 
                           mode='train', data_dir='/home/mnt/Datasets/183_retargeted', 
                           balanced_sampling=False):
    """Create data loader for 183 Athletes dataset"""
    dataset = Athletes183Dataset(
        window_size=window_size, 
        unit_length=unit_length, 
        mode=mode, 
        data_dir=data_dir
    )
    
    # Use balanced sampling if requested
    sampler = None
    shuffle = True
    
    if balanced_sampling and mode == 'train':
        prob = dataset.compute_sampling_prob()
        sampler = torch.utils.data.WeightedRandomSampler(
            prob, 
            num_samples=len(dataset) * 3,  # Oversample for balance
            replacement=True
        )
        shuffle = False  # Don't shuffle when using sampler
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True
    )
    return loader


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    motions, lengths, names, athletes, biomechs, movement_types = zip(*batch)
    
    # Stack motions (already padded to window_size)
    motions = torch.stack(motions)
    lengths = torch.tensor(lengths)
    
    return motions, lengths, names, athletes, biomechs, movement_types


if __name__ == "__main__":
    # Test the dataset
    print("Testing Athletes183Dataset...")
    
    # Test loading
    dataset = Athletes183Dataset(window_size=64, mode='train', data_dir='/home/mnt/Datasets/183_retargeted')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test getting an item
        motion, length, name, athlete, biomech, movement_type = dataset[0]
        print(f"Motion shape: {motion.shape}")
        print(f"Length: {length}")
        print(f"Name: {name}")
        print(f"Athlete: {athlete}")
        print(f"Movement type: {movement_type}")
        print(f"Biomech features: {list(biomech.keys())}")
        
        # Test data loader
        loader = athletes183_data_loader(batch_size=2, mode='train', data_dir='/home/mnt/Datasets/183_retargeted')
        batch = next(iter(loader))
        motions, lengths, names, athletes, biomechs, movement_types = batch
        print(f"Batch motion shape: {motions.shape}")
        print(f"Batch lengths: {lengths}")
        print(f"Movement types in batch: {movement_types}")
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Unique athletes: {len(dataset.get_athlete_names())}")
        print(f"Unique movement types: {len(dataset.get_movement_types())}")
        print(f"Movement types: {dataset.get_movement_types()}")
    else:
        print("No data loaded. Check if .npz files exist in the data directory.")