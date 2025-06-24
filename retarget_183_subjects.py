import os
import sys
import time
import argparse
import logging
import numpy as np
import scipy.io
import nimblephysics as nimble
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Retarget183")

class Retarget183Subjects:
    def __init__(self, skeleton_path, output_dir):
        """
        Initialize the retargeter with a reference skeleton.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.subject_on_disk = None

        # Load reference skeleton
        logger.info(f"Loading reference skeleton: {skeleton_path}")
        ref_subject = nimble.biomechanics.SubjectOnDisk(skeleton_path)
        self.skeleton = ref_subject.readSkel(processingPass=0, ignoreGeometry=True)

    def retarget_subject(self, subject_id, data_dir):
        """
        Retarget all trials for a given subject to AddBiomechanics format,
        fitting each trial's joint centers onto the reference skeleton.
        """
        import scipy.io

        subject_path = os.path.join(data_dir, str(subject_id))
        if not os.path.isdir(subject_path):
            logger.warning(f"Subject folder not found: {subject_path}")
            return False

        # Create an empty SubjectOnDisk to accumulate trials
        logger.info(f"Preparing subject {subject_id}")
        header = nimble.biomechanics.SubjectOnDiskHeader()
        header.mSkeleton = self.skeleton
        self.subject_on_disk = nimble.biomechanics.SubjectOnDisk(header)

        # For example, define the skeleton joints in the same order as your data’s joint-centers array
        # Adapt these to match your skeleton's actual joint names
        bodyJoints = [
            self.skeleton.getJoint("hip_l"),
            self.skeleton.getJoint("hip_r"),
            self.skeleton.getJoint("knee_l"),
            self.skeleton.getJoint("knee_r"),
            # ... add more joints as applicable ...
        ]

        mat_files = [f for f in os.listdir(subject_path) if f.endswith('.mat')]
        if not mat_files:
            logger.warning(f"No .mat files found for subject {subject_id}")
            return False

        for mat_file in mat_files:
            trial_path = os.path.join(subject_path, mat_file)
            logger.info(f"Retargeting trial: {mat_file}")

            try:
                data = scipy.io.loadmat(trial_path)
            except Exception as e:
                logger.error(f"Failed to load {trial_path}: {e}")
                continue

            # Example: assume the joint center data is in data["joint_centers"] with shape [frames, joints, 3]
            if "joint_centers" not in data:
                logger.warning(f"No 'joint_centers' found in {mat_file}, skipping.")
                continue
            
            joint_centers = data["joint_centers"]  # shape (num_frames, len(bodyJoints), 3)
            num_frames = joint_centers.shape[0]

            # Fit each frame of data to the skeleton
            frames_poses = []
            for frame_idx in range(num_frames):
                # Extract the positions for this frame (list of 3D vectors)
                frame_pos = [joint_centers[frame_idx, j, :] for j in range(len(bodyJoints))]
                # Solve IK to match the skeleton’s joints to these positions
                self.skeleton.fitJointsToWorldPositions(bodyJoints, frame_pos, scaleBodies=False)
                # Store the resulting pose
                frames_poses.append(self.skeleton.getPositions())

            # Add this trial’s retargeted poses to our SubjectOnDisk
            self.subject_on_disk.addTrial(poses=np.array(frames_poses))

        # Write out a single .b3d file for all trials under this subject
        output_b3d = os.path.join(self.output_dir, f"{subject_id}.b3d")
        self.subject_on_disk.writeB3D(output_b3d)
        logger.info(f"Subject {subject_id} saved to {output_b3d}")
        return True

    def _generate_dummy_poses(self, num_frames=100):
        """
        A placeholder that generates random joint data.
        Replace with actual retargeting logic using nimble fitMarkers or fitJoints.
        """
        n_dofs = self.skeleton.getNumDofs()
        poses = np.random.randn(num_frames, n_dofs)
        return poses

def main():
    parser = argparse.ArgumentParser(description="Retarget 183 subjects dataset to AddBiomechanics format.")
    parser.add_argument("--dataset_dir", required=True, help="Path to the dataset root directory.")
    parser.add_argument("--reference_skeleton", required=True, help="Path to the reference skeleton (.b3d).")
    parser.add_argument("--output_dir", required=True, help="Output directory for retargeted .b3d files.")
    args = parser.parse_args()

    start_time = time.time()
    retarget = Retarget183Subjects(args.reference_skeleton, args.output_dir)

    # Identify subject folders (e.g., if they're numeric subfolders).
    subject_folders = [d for d in os.listdir(args.dataset_dir)
                       if os.path.isdir(os.path.join(args.dataset_dir, d))]

    success_count = 0
    for subject_id in tqdm(subject_folders, desc="Retargeting Subjects"):
        success = retarget.retarget_subject(subject_id, args.dataset_dir)
        if success:
            success_count += 1

    elapsed = time.time() - start_time
    logger.info(f"Successfully retargeted {success_count}/{len(subject_folders)} subjects.")
    logger.info(f"Completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()