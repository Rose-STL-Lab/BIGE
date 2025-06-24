import codecs as cs
import nimblephysics as nimble
import numpy as np
import os
import random
import torch
from glob import glob
from os.path import join as pjoin
from torch.utils import data
from tqdm import tqdm

class AddBiomechanicsDataset(data.Dataset):
    def __init__(self, window_size=64, unit_length=4, mode='train', data_dir='addb_dataset_publication'):
        self.window_size = window_size
        self.unit_length = unit_length
        self.data_dir = data_dir
        self.mode = mode

        # Define subdirectories for each paper
        if mode == 'train':
            paper_dirs = [
                "train/No_Arm/Falisse2017_Formatted_No_Arm",
                "test/No_Arm/Falisse2017_Formatted_No_Arm",
                "train/No_Arm/Han2023_Formatted_No_Arm",
                "test/No_Arm/Han2023_Formatted_No_Arm",
                "test/No_Arm/Uhlrich2023_Formatted_No_Arm",
                "train/No_Arm/Wang2023_Formatted_No_Arm",
                "test/No_Arm/Wang2023_Formatted_No_Arm",
            ]
        elif mode == 'test':
            paper_dirs = [
                "test/No_Arm/Falisse2017_Formatted_No_Arm",
                "test/No_Arm/Uhlrich2023_Formatted_No_Arm",
                "test/No_Arm/Wang2023_Formatted_No_Arm",
                "test/No_Arm/Han2023_Formatted_No_Arm",
            ]
        elif mode == 'all':
            paper_dirs = [
                "train/No_Arm/Camargo2021_Formatted_No_Arm",
                "train/No_Arm/Hamner2013_Formatted_No_Arm",
                "train/No_Arm/Tan2021_Formatted_No_Arm",
                "train/No_Arm/vanderZee2022_Formatted_No_Arm",
                "train/No_Arm/Carter2023_Formatted_No_Arm",
                "train/No_Arm/Han2023_Formatted_No_Arm",
                "train/No_Arm/Tan2022_Formatted_No_Arm",
                "train/No_Arm/Falisse2017_Formatted_No_Arm",
                "train/No_Arm/Moore2015_Formatted_No_Arm",
                "train/No_Arm/Tiziana2019_Formatted_No_Arm",
                "train/No_Arm/Fregly2012_Formatted_No_Arm",
                "train/No_Arm/Santos2017_Formatted_No_Arm",
                "test/No_Arm/Camargo2021_Formatted_No_Arm",
                "test/No_Arm/Hamner2013_Formatted_No_Arm",
                "test/No_Arm/Tan2021_Formatted_No_Arm",
                "test/No_Arm/vanderZee2022_Formatted_No_Arm",
                "test/No_Arm/Carter2023_Formatted_No_Arm",
                "test/No_Arm/Han2023_Formatted_No_Arm",
                "test/No_Arm/Tan2022_Formatted_No_Arm",
                "test/No_Arm/Falisse2017_Formatted_No_Arm",
                "test/No_Arm/Moore2015_Formatted_No_Arm",
                "test/No_Arm/Tiziana2019_Formatted_No_Arm",
                "test/No_Arm/Fregly2012_Formatted_No_Arm",
                "test/No_Arm/Santos2017_Formatted_No_Arm",
                "test/No_Arm/Uhlrich2023_Formatted_No_Arm",
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Collect all .b3d files from the specified subdirectories
        self.b3d_file_paths = []
        for paper_dir in paper_dirs:
            search_path = os.path.join(data_dir, paper_dir, '**', '*.b3d')
            files = glob(search_path, recursive=True)
            self.b3d_file_paths.extend(files)

        self.motion_data = []
        self.motion_lengths = []
        self.motion_names = []
        self.motion_fps = []
        self.subject_names = []
        self.subject_metadata = {}
        self.subject_biomech = {}

        def extract_subject_name(b3d_file):
            parts = b3d_file.split(os.sep)
            for i in range(len(parts)-1, 1, -1):
                if parts[i].endswith("_Formatted_No_Arm") or parts[i].endswith("_Formatted_With_Arm"):
                    paper = parts[i].replace("_Formatted_No_Arm", "").replace("_Formatted_With_Arm", "")
                    subj_folder = parts[i+1] if i+1 < len(parts) else ""
                    subj = subj_folder.split("_split")[0]
                    return f"{paper}/{subj}"
            paper = parts[-2].replace("_Formatted_No_Arm", "").replace("_Formatted_With_Arm", "")
            subj = parts[-1].split("_split")[0]
            return f"{paper}/{subj}"

        for b3d_file in tqdm(self.b3d_file_paths):
            try:
                if os.path.getsize(b3d_file) == 0:
                    continue
                subject = nimble.biomechanics.SubjectOnDisk(b3d_file)
                num_trials = subject.getNumTrials()
                subject_name = extract_subject_name(b3d_file)
                if subject_name not in self.subject_metadata:
                    self.subject_metadata[subject_name] = {
                        "age": subject.getAgeYears(),
                        "height": subject.getHeightM(),
                        "mass": subject.getMassKg(),
                        "sex": subject.getBiologicalSex()
                    }
                # Gather biomechanical features only once per subject
                if subject_name not in self.subject_biomech:
                    velocities = []
                    velocities_max = []
                    grf_mags = []
                    joint_centers = []
                    com_pos = []
                    com_vel = []
                    com_acc = []
                    for trial in range(num_trials):
                        trial_length = subject.getTrialLength(trial)
                        frames = subject.readFrames(
                            trial=trial,
                            startFrame=0,
                            numFramesToRead=trial_length,
                            includeSensorData=False,
                            includeProcessingPasses=True
                        )
                        for frame in frames:
                            v = np.abs(frame.processingPasses[0].vel)
                            velocities.append(v.mean())
                            velocities_max.append(v.max())
                            grf = getattr(frame.processingPasses[0], "groundContactForce", None)
                            if grf is not None and len(grf) > 0:
                                grf = np.array(grf).reshape(-1, 3)
                                grf_mags.append(np.linalg.norm(grf, axis=1).sum())
                            jc = getattr(frame.processingPasses[0], "jointCenters", None)
                            if jc is not None and len(jc) > 0:
                                jc = np.array(jc)
                                joint_centers.append(jc)
                            cp = getattr(frame.processingPasses[0], "comPos", None)
                            if cp is not None:
                                com_pos.append(np.array(cp))
                            cv = getattr(frame.processingPasses[0], "comVel", None)
                            if cv is not None:
                                com_vel.append(np.array(cv))
                            ca = getattr(frame.processingPasses[0], "comAcc", None)
                            if ca is not None:
                                com_acc.append(np.array(ca))
                    # Compute stats
                    biomech = {
                        "mean_joint_velocities": float(np.mean(velocities)) if velocities else 0.0,
                        "max_joint_velocities": float(np.max(velocities_max)) if velocities_max else 0.0,
                        "mean_grf": float(np.mean(grf_mags)) if grf_mags else 0.0,
                        "max_grf": float(np.max(grf_mags)) if grf_mags else 0.0,
                        "mean_joint_center": float(np.mean(np.concatenate(joint_centers, axis=0))) if joint_centers else 0.0,
                        "var_joint_center": float(np.var(np.concatenate(joint_centers, axis=0))) if joint_centers else 0.0,
                        "mean_com_pos": float(np.mean(np.stack(com_pos))) if com_pos else 0.0,
                        "var_com_pos": float(np.var(np.stack(com_pos))) if com_pos else 0.0,
                        "mean_com_vel": float(np.mean(np.stack(com_vel))) if com_vel else 0.0,
                        "var_com_vel": float(np.var(np.stack(com_vel))) if com_vel else 0.0,
                        "mean_com_acc": float(np.mean(np.stack(com_acc))) if com_acc else 0.0,
                        "var_com_acc": float(np.var(np.stack(com_acc))) if com_acc else 0.0,
                    }
                    self.subject_biomech[subject_name] = biomech
                for trial in range(num_trials):
                    trial_length = subject.getTrialLength(trial)
                    if trial_length < self.window_size:
                        continue
                    frames = subject.readFrames(
                        trial=trial,
                        startFrame=0,
                        numFramesToRead=trial_length,
                        includeSensorData=False,
                        includeProcessingPasses=True
                    )
                    if not frames:
                        continue
                    kin_passes = [frame.processingPasses[0] for frame in frames]
                    positions = np.array([kp.pos for kp in kin_passes])
                    seconds_per_frame = subject.getTrialTimestep(trial)
                    fps = int(round(1.0 / seconds_per_frame)) if seconds_per_frame > 0 else 0
                    target_fps = 100
                    if fps > target_fps:
                        step = int(round(fps / target_fps))
                        positions = positions[::step]
                        fps = int(round(fps / step))
                    elif fps < target_fps:
                        continue
                    if len(positions) < self.window_size:
                        continue
                    self.motion_data.append(positions)
                    self.motion_lengths.append(len(positions))
                    self.motion_names.append(f"{b3d_file}::trial{trial}")
                    self.motion_fps.append(fps)
                    self.subject_names.append(subject_name)
            except Exception as e:
                print(f"Skipping file {b3d_file} due to error: {e}")

        print("Total number of motions:", len(self.motion_data))

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, item):
        motion = self.motion_data[item]
        len_motion = len(motion) if len(motion) <= self.window_size else self.window_size
        name = self.motion_names[item]
        subject_name = self.subject_names[item]
        biomech = self.subject_biomech[subject_name]

        # Crop or pad to window_size (no downsampling here)
        if len(motion) >= self.window_size:
            idx = random.randint(0, len(motion) - self.window_size)
            motion = motion[idx:idx + self.window_size]
        else:
            repeat_count = (self.window_size + len(motion) - 1) // len(motion)
            motion = np.tile(motion, (repeat_count, 1))[:self.window_size]

        return motion, len_motion, name, subject_name, biomech
    

def addb_data_loader(window_size=64, unit_length=4, batch_size=1, num_workers=4, mode='train', data_dir='addb_dataset_publication'):
    dataset = AddBiomechanicsDataset(window_size=window_size, unit_length=unit_length, mode=mode, data_dir=data_dir)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return loader

def main():
    dataloader = addb_data_loader(window_size=64, unit_length=4, batch_size=1, mode='train')

if __name__ == "__main__":
    main() 
    